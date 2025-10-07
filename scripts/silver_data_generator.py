import os
import gc
import pandas as pd
import ta
import talib
import numba
import numpy as np
from tqdm import tqdm

# --- CONFIGURATION ---
SMA_PERIODS = [20, 50, 100, 200]
EMA_PERIODS = [8, 13, 21, 50]
BBANDS_PERIOD, BBANDS_STD_DEV = 20, 2.0
RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
ATR_PERIOD = 14
ADX_PERIOD = 14
SR_LOOKBACK = 200
PAST_LOOKBACKS = [3, 5, 10, 20, 50]

# --- NEW: Define the warmup period ---
INDICATOR_WARMUP_PERIOD = 200 # Rows to remove from the start of raw data
# Process the huge bronze file in chunks of this size
BRONZE_CHUNK_SIZE = 1_000_000

# --- UTILITY & FEATURE FUNCTIONS (Unchanged) ---
def downcast_dtypes(df):
    """Downcast numeric dtypes to save RAM."""
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
    return df

def robust_read_csv(filepath):
    """
    Reads a RAW data CSV file, assuming it has NO header.
    Assigns standard OHLCV column names.
    """
    df = pd.read_csv(filepath, sep=None, engine="python", header=None)
    
    # Assign column names based on the number of columns found
    if df.shape[1] > 5:
        df = df.iloc[:, :6]
        df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
    elif df.shape[1] == 5:
        df.columns = ['time', 'open', 'high', 'low', 'close']
        df['volume'] = 1 # Add volume if it's missing
    else:
        raise ValueError(f"Raw data file '{filepath}' has fewer than 5 columns.")

    # --- Proceed with data processing ---
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Drop rows with parsing errors
    df.dropna(subset=['time'] + numeric_cols, inplace=True)
    return df.sort_values('time').reset_index(drop=True)

def add_all_features(df):
    for p in SMA_PERIODS:
        df[f"SMA_{p}"] = ta.trend.SMAIndicator(df["close"], p).sma_indicator()
    for p in EMA_PERIODS:
        df[f"EMA_{p}"] = ta.trend.EMAIndicator(df["close"], p).ema_indicator()

    bb = ta.volatility.BollingerBands(df["close"], BBANDS_PERIOD, BBANDS_STD_DEV)
    df["BB_upper"], df["BB_lower"], df["BB_width"] = bb.bollinger_hband(), bb.bollinger_lband(), bb.bollinger_wband()

    df[f"RSI_{RSI_PERIOD}"] = ta.momentum.RSIIndicator(df["close"], RSI_PERIOD).rsi()
    df["MACD_hist"] = ta.trend.MACD(df["close"], MACD_SLOW, MACD_FAST, MACD_SIGNAL).macd_diff()
    
    atr_indicator = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], ATR_PERIOD)
    df[f"ATR_{ATR_PERIOD}"] = atr_indicator.average_true_range()
    
    # 1️⃣ --- CORE IDEA: ADDED ATR-BASED DYNAMIC LEVELS ---
    # These create dynamic price channels based on recent volatility.
    atr_series = df[f"ATR_{ATR_PERIOD}"]
    df["ATR_level_up_1x"] = df["close"] + atr_series
    df["ATR_level_down_1x"] = df["close"] - atr_series
    df["ATR_level_up_2x"] = df["close"] + (atr_series * 2)
    df["ATR_level_down_2x"] = df["close"] - (atr_series * 2)

    df["ADX"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], ADX_PERIOD).adx()
    df["MOM_10"] = ta.momentum.ROCIndicator(df["close"], window=10).roc()
    df["CCI_20"] = ta.trend.CCIIndicator(df["high"], df["low"], df["close"], window=20).cci()

    if "volume" in df.columns:
        df["OBV"] = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()

    patterns_data = {};
    for p in talib.get_function_groups().get("Pattern Recognition", []):
        try:
            patterns_data[p] = getattr(talib, p)(df["open"], df["high"], df["low"], df["close"])
        except:
            continue
    df = pd.concat([df, pd.DataFrame(patterns_data, index=df.index)], axis=1)
    
    df = add_support_resistance(df)
    
    df['session'] = df['time'].dt.hour.map(lambda h: 'London' if 7<=h<12 else 'London_NY_Overlap' if 12<=h<16 else 'New_York' if 16<=h<21 else 'Asian')
    df['hour'] = df['time'].dt.hour; df['weekday'] = df['time'].dt.weekday
    df['is_bullish'] = (df['close'] > df['open']).astype(int); df['body_size'] = (df['close'] - df['open']).abs()

    new_cols = {}
    for n in PAST_LOOKBACKS:
        new_cols[f'bullish_ratio_last_{n}'] = df['is_bullish'].rolling(n).mean()
        new_cols[f'avg_body_last_{n}'] = df['body_size'].rolling(n).mean()
        new_cols[f'avg_range_last_{n}'] = (df['high'] - df['low']).rolling(n).mean()
        new_cols[f'close_SMA20_ratio_{n}'] = df['close'] / df[f'SMA_{n}' if f'SMA_{n}' in df.columns else 'SMA_20']
        new_cols[f'EMA8_EMA21_ratio_{n}'] = df['EMA_8'] / df['EMA_21']
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    
    df.drop(columns=['is_bullish', 'body_size'], inplace=True)
    df['trend_regime'] = np.where(df['ADX'] > 25, 'trend', 'range')
    df['vol_regime'] = np.where(df[f'ATR_{ATR_PERIOD}'] > df[f'ATR_{ATR_PERIOD}'].rolling(50).mean(), 'high_vol', 'low_vol')

    return df

@numba.njit
def _calculate_s_r_numba(lows, highs, lookback=SR_LOOKBACK):
    n = len(lows)
    supports = np.full(n, np.nan, dtype=np.float32)
    resistances = np.full(n, np.nan, dtype=np.float32)
    last_support = np.nan
    last_resistance = np.nan
    for i in range(lookback + 2, n - 2):
        # --- Step 1: Detect local pivot supports/resistances ---
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            last_support = lows[i]
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            last_resistance = highs[i]

        # --- Step 2: Validation within lookback window ---
        # Support remains valid only if not broken in lookback range
        if not np.isnan(last_support):
            valid = True
            for j in range(max(0, i - lookback), i):
                if lows[j] < last_support:
                    valid = False; break
            if valid:
                supports[i] = last_support
            else:
                last_support = np.nan

        # Resistance remains valid only if not broken in lookback range
        if not np.isnan(last_resistance):
            valid = True
            for j in range(max(0, i - lookback), i):
                if highs[j] > last_resistance:
                    valid = False; break
            if valid:
                resistances[i] = last_resistance
            else:
                last_resistance = np.nan
    return supports, resistances

def add_support_resistance(df, lookback=SR_LOOKBACK):
    lows = df["low"].values.astype(np.float32)
    highs = df["high"].values.astype(np.float32)
    supports, resistances = _calculate_s_r_numba(lows, highs, lookback)
    df["support_points"] = supports
    df["resistance_points"] = resistances
    df["support"] = df["support_points"].ffill()
    df["resistance"] = df["resistance_points"].ffill()
    return df.drop(columns=["support_points", "resistance_points"])

# 1️⃣ --- CORE IDEA: NEW FUNCTION TO ADD RELATIONAL FEATURES ---
def add_relational_features(df):
    """
    Adds features that describe the normalized distance from SL/TP to key price levels.
    A positive value means the SL/TP is ABOVE the level.
    A negative value means the SL/TP is BELOW the level.
    This function must be called on the merged dataframe that contains both
    trade data (sl, tp) and indicator data (support, resistance, etc.).
    """
    # Key price levels to compare against
    levels = {
        'support': df['support'],
        'resistance': df['resistance'],
        'bb_upper': df['BB_upper'],
        'bb_lower': df['BB_lower'],
        'atr_up_1x': df['ATR_level_up_1x'],
        'atr_down_1x': df['ATR_level_down_1x'],
    }
    
    # Add all SMA and EMA lines to the levels dictionary dynamically
    for p in SMA_PERIODS:
        levels[f'sma_{p}'] = df[f'SMA_{p}']
    for p in EMA_PERIODS:
        levels[f'ema_{p}'] = df[f'EMA_{p}']
        
    sl = df['sl_price']
    tp = df['tp_price']
    # Use 'close' at entry time as the normalizing factor to make distances comparable
    close_price = df['close'] 
    
    for name, level_series in levels.items():
        # Check if the level series has valid data to avoid all-NaN columns
        if not level_series.isnull().all():
            # Calculate normalized distance from the Stop Loss to the level
            df[f'sl_dist_to_{name}_norm'] = (sl - level_series) / close_price
            
            # Calculate normalized distance from the Take Profit to the level
            df[f'tp_dist_to_{name}_norm'] = (tp - level_series) / close_price
            
    return df

def create_silver_data_chunked(bronze_file, raw_file, silver_file, htf_file=None):
    print(f"\n{'='*25}\nProcessing: {os.path.basename(raw_file)}\n{'='*25}")

    # 1. Load Raw Data to determine the valid time range
    print("Loading raw OHLCV to determine indicator warmup period...")
    raw_df = robust_read_csv(raw_file)

    if len(raw_df) < INDICATOR_WARMUP_PERIOD + 1:
        print(f" ❌ SKIPPING: Not enough data ({len(raw_df)} rows) for indicator warmup ({INDICATOR_WARMUP_PERIOD}).")
        return

    print("Calculating all historical features...")
    features_df = add_all_features(raw_df)
    del raw_df
    gc.collect()

    features_df = downcast_dtypes(features_df)
    print(" ✅ Historical features calculated and cached in memory.")

    warmup_cutoff_time = features_df['time'].iloc[INDICATOR_WARMUP_PERIOD]
    print(f"  Indicator warmup Initialized. Only processing trades on or after: {warmup_cutoff_time}")
    # Now, we only need the data *after* the warmup
    features_df = features_df[features_df['time'] >= warmup_cutoff_time].copy()
    features_df.sort_values('time', inplace=True)
    features_df.reset_index(drop=True, inplace=True) # Reset index after filtering
    print(f"  Remaining rows after warmup filter: {len(features_df)}")

    # 3. Process the massive Bronze file in chunks
    print(f"Loading and filtering bronze trade data in chunks of {BRONZE_CHUNK_SIZE}...")
    if os.path.exists(silver_file):
        os.remove(silver_file)
    
    # --- MODIFIED: Read the bronze file assuming it HAS a header (header=0 is default) ---
    bronze_reader = pd.read_csv(
        bronze_file,
        chunksize=BRONZE_CHUNK_SIZE
    )
    is_first_chunk = True
    
    for chunk in tqdm(bronze_reader, desc="Merging Bronze Chunks"):
        chunk['entry_time'] = pd.to_datetime(chunk['entry_time'], errors='coerce')
        chunk.dropna(subset=['entry_time'], inplace=True)
        
        chunk = chunk[chunk['entry_time'] >= warmup_cutoff_time].copy()
        if chunk.empty:
            continue
            
        chunk.sort_values('entry_time', inplace=True)
        
        merged_chunk = pd.merge_asof(
            chunk,
            features_df,
            left_on='entry_time',
            right_on='time',
            direction='backward'
        )
        
        # 1️⃣ --- CORE IDEA: CALL THE NEW FUNCTION HERE ---
        # This is the crucial step where context is added to each trade.
        if not merged_chunk.empty:
            # Drop rows where the merge might have failed to find indicator data
            merged_chunk.dropna(subset=['close'], inplace=True) 
            merged_chunk = add_relational_features(merged_chunk)

        merged_chunk.drop(columns=['time'], inplace=True, errors='ignore')
        merged_chunk = downcast_dtypes(merged_chunk)
        
        merged_chunk.to_csv(
            silver_file,
            mode='a',
            header=is_first_chunk,
            index=False
        )
        is_first_chunk = False
        del chunk, merged_chunk
        gc.collect()

    if not is_first_chunk:
        print(f"\n✅ Success! Rich dataset with all chunks merged saved to {os.path.basename(silver_file)}")
    else:
        print("\n⚠️ No valid trades found after filtering for indicator warmup period. Silver file not created.")

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    core_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.abspath(os.path.join(core_dir, '..', 'raw_data'))
    bronze_dir = os.path.abspath(os.path.join(core_dir, '..', 'bronze_data'))
    silver_dir = os.path.abspath(os.path.join(core_dir, '..', 'silver_data'))
    os.makedirs(silver_dir, exist_ok=True)

    bronze_files = [f for f in os.listdir(bronze_dir) if f.endswith('.csv')]
    if not bronze_files:
        print("❌ No bronze files found to process.")
    else:
        for fname in bronze_files:
            bronze_path = os.path.join(bronze_dir, fname)
            raw_path = os.path.join(raw_dir, fname)
            silver_path = os.path.join(silver_dir, fname)
            
            if not os.path.exists(raw_path):
                print(f"⚠️ SKIPPING: Raw file for {fname} not found."); continue

            htf_filepath = None
            base_symbol = fname.rstrip('0123456789.csv')
            for tf in ['240', '60']:
                candidate_path = os.path.join(raw_dir, f"{base_symbol}{tf}.csv")
                if os.path.exists(candidate_path):
                    htf_filepath = candidate_path; break
            
            try:
                create_silver_data_chunked(bronze_path, raw_path, silver_path, htf_file=htf_filepath)
            except Exception as e:
                print(f"❌ FAILED to process {fname}. Error: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "="*50 + "\n✅ All silver data generation complete.")
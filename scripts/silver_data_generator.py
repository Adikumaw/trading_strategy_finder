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
PIVOT_WINDOW = 10 
PAST_LOOKBACKS = [3, 5, 10, 20, 50]
INDICATOR_WARMUP_PERIOD = 200

# --- UTILITY & FEATURE FUNCTIONS ---
def downcast_dtypes(df):
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
        df['volume'] = 1
    else:
        raise ValueError(f"Raw data file '{filepath}' has fewer than 5 columns.")

    # --- Proceed with data processing ---
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Drop rows with parsing errors
    df.dropna(subset=['time'] + numeric_cols, inplace=True)
    return df.sort_values('time').reset_index(drop=True)

@numba.njit
def _calculate_s_r_numba(lows, highs, window):
    """
    Identifies pivot points for support and resistance.
    A low is a support pivot if it's the lowest value in a window around it.
    A high is a resistance pivot if it's the highest value in a window around it.
    """
    n = len(lows)
    support_points = np.full(n, np.nan, dtype=np.float32)
    resistance_points = np.full(n, np.nan, dtype=np.float32)

    # Iterate from the first possible pivot to the last
    for i in range(window, n - window):
        # Define the window to check
        window_lows = lows[i - window : i + window + 1]
        window_highs = highs[i - window : i + window + 1]
        
        # Check for support
        if lows[i] == np.min(window_lows):
            support_points[i] = lows[i]
            
        # Check for resistance
        if highs[i] == np.max(window_highs):
            resistance_points[i] = highs[i]
    return support_points, resistance_points

def add_support_resistance(df, window=PIVOT_WINDOW):
    """
    Calculates support and resistance levels based on pivot points.
    """
    lows = df["low"].values.astype(np.float32)
    highs = df["high"].values.astype(np.float32)
    support_points, resistance_points = _calculate_s_r_numba(lows, highs, window)
    
    # Use a temporary DataFrame to avoid fragmentation
    sr_df = pd.DataFrame(index=df.index)
    sr_df["support_points"] = support_points
    sr_df["resistance_points"] = resistance_points
    sr_df["support"] = sr_df["support_points"].ffill()
    sr_df["resistance"] = sr_df["resistance_points"].ffill()
    
    return pd.concat([df, sr_df[['support', 'resistance']]], axis=1)

# --- REFACTORED FOR PERFORMANCE ---
def add_all_market_features(df):
    """Calculates all market-based features for each candle efficiently."""
    
    # Create a list to hold all the new feature DataFrames
    new_features_list = []
    
    # --- Batch 1: Standard Indicators ---
    print("Calculating standard indicators (SMA, EMA, RSI, etc.)...")
    indicator_df = pd.DataFrame(index=df.index)
    for p in SMA_PERIODS: indicator_df[f"SMA_{p}"] = ta.trend.SMAIndicator(df["close"], p).sma_indicator()
    for p in EMA_PERIODS: indicator_df[f"EMA_{p}"] = ta.trend.EMAIndicator(df["close"], p).ema_indicator()
    bb = ta.volatility.BollingerBands(df["close"], BBANDS_PERIOD, BBANDS_STD_DEV)
    indicator_df["BB_upper"], indicator_df["BB_lower"], indicator_df["BB_width"] = bb.bollinger_hband(), bb.bollinger_lband(), bb.bollinger_wband()
    indicator_df[f"RSI_{RSI_PERIOD}"] = ta.momentum.RSIIndicator(df["close"], RSI_PERIOD).rsi()
    indicator_df["MACD_hist"] = ta.trend.MACD(df["close"], MACD_SLOW, MACD_FAST, MACD_SIGNAL).macd_diff()
    indicator_df[f"ATR_{ATR_PERIOD}"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], ATR_PERIOD).average_true_range()
    indicator_df["ADX"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], ADX_PERIOD).adx()
    indicator_df["MOM_10"] = ta.momentum.ROCIndicator(df["close"], window=10).roc()
    indicator_df["CCI_20"] = ta.trend.CCIIndicator(df["high"], df["low"], df["close"], window=20).cci()
    if "volume" in df.columns and df['volume'].nunique() > 1:
        indicator_df["OBV"] = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
    new_features_list.append(indicator_df)

    # --- Batch 2: Candlestick Patterns ---
    print("Calculating candlestick patterns...")
    patterns_data = {}
    pattern_list = talib.get_function_groups().get("Pattern Recognition", [])
    for p in tqdm(pattern_list, desc="Candlestick Patterns"):
        try:
            patterns_data[p] = getattr(talib, p)(df["open"], df["high"], df["low"], df["close"])
        except: continue
    patterns_df = pd.DataFrame(patterns_data, index=df.index)
    new_features_list.append(patterns_df)
    
    # --- Batch 3: Support & Resistance ---
    print("Calculating support and resistance...")
    df_with_sr = add_support_resistance(df.copy())
    new_features_list.append(df_with_sr[['support', 'resistance']])

    # --- Batch 4: Time-based and Price-Action Features ---
    print("Calculating time-based and price-action features...")
    price_action_df = pd.DataFrame(index=df.index)
    price_action_df['session'] = df['time'].dt.hour.map(lambda h: 'London' if 7<=h<12 else 'London_NY_Overlap' if 12<=h<16 else 'New_York' if 16<=h<21 else 'Asian')
    price_action_df['hour'] = df['time'].dt.hour
    price_action_df['weekday'] = df['time'].dt.weekday
    
    # Temporary columns for calculation
    is_bullish = (df['close'] > df['open']).astype(int)
    body_size = (df['close'] - df['open']).abs()
    
    for n in PAST_LOOKBACKS:
        price_action_df[f'bullish_ratio_last_{n}'] = is_bullish.rolling(n).mean()
        price_action_df[f'avg_body_last_{n}'] = body_size.rolling(n).mean()
        price_action_df[f'avg_range_last_{n}'] = (df['high'] - df['low']).rolling(n).mean()
        price_action_df[f'close_SMA20_ratio_{n}'] = df['close'] / indicator_df.get(f'SMA_{n}', indicator_df.get('SMA_20'))
        if f'EMA_8' in indicator_df.columns and f'EMA_21' in indicator_df.columns:
            price_action_df[f'EMA8_EMA21_ratio_{n}'] = indicator_df['EMA_8'] / indicator_df['EMA_21']
    
    price_action_df['trend_regime'] = np.where(indicator_df['ADX'] > 25, 'trend', 'range')
    if f'ATR_{ATR_PERIOD}' in indicator_df.columns:
        price_action_df['vol_regime'] = np.where(indicator_df[f'ATR_{ATR_PERIOD}'] > indicator_df[f'ATR_{ATR_PERIOD}'].rolling(50).mean(), 'high_vol', 'low_vol')
    new_features_list.append(price_action_df)

    # --- Final Assembly ---
    print("Assembling final features dataset...")
    # Concatenate all new feature DataFrames to the original one in a single, efficient operation
    return pd.concat([df] + new_features_list, axis=1)


def create_decoupled_silver_data(bronze_path, raw_path, features_path, outcomes_path):
    """Main function to create the new, decoupled silver datasets."""
    print(f"\n{'='*25}\nProcessing: {os.path.basename(raw_path)}\n{'='*25}")

    # --- STEP 1: Create the Silver Features Dataset ---
    print("STEP 1: Creating Silver Features dataset (unique per candle)...")
    raw_df = robust_read_csv(raw_path)
    if len(raw_df) < INDICATOR_WARMUP_PERIOD + 1:
        print(f"❌ SKIPPING: Not enough data ({len(raw_df)} rows) for indicator warmup.")
        return

    features_df = add_all_market_features(raw_df)
    del raw_df 
    gc.collect()
    
    features_df = features_df.iloc[INDICATOR_WARMUP_PERIOD:].reset_index(drop=True)
    
    print(f"Downcasting features data to save space...")
    features_df = downcast_dtypes(features_df)
    
    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    features_df.to_csv(features_path, index=False)
    print(f"✅ Silver Features saved to: {features_path} ({len(features_df)} rows)")
    
    first_feature_time = features_df['time'].iloc[0]
    del features_df; gc.collect()

    # --- STEP 2: Create the Silver Outcomes Dataset ---
    print("\nSTEP 2: Creating Silver Outcomes dataset (from bronze data)...")
    print(f"Will filter outcomes to start on or after: {first_feature_time}")
    
    outcomes_iterator = pd.read_csv(bronze_path, chunksize=1_000_000)
    is_first_chunk = True
    os.makedirs(os.path.dirname(outcomes_path), exist_ok=True)
    if os.path.exists(outcomes_path): os.remove(outcomes_path)

    for chunk in tqdm(outcomes_iterator, desc="Processing Bronze Chunks"):
        chunk['entry_time'] = pd.to_datetime(chunk['entry_time'])
        chunk = chunk[chunk['entry_time'] >= first_feature_time]
        
        if not chunk.empty:
            chunk.to_csv(outcomes_path, mode='a', header=is_first_chunk, index=False)
            is_first_chunk = False
            
    print(f"✅ Silver Outcomes saved to: {outcomes_path}")


if __name__ == "__main__":
    core_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.abspath(os.path.join(core_dir, '..', 'raw_data'))
    bronze_dir = os.path.abspath(os.path.join(core_dir, '..', 'bronze_data'))
    features_dir = os.path.abspath(os.path.join(core_dir, '..', 'silver_data', 'features'))
    outcomes_dir = os.path.abspath(os.path.join(core_dir, '..', 'silver_data', 'outcomes'))
    
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(outcomes_dir, exist_ok=True)

    bronze_files = [f for f in os.listdir(bronze_dir) if f.endswith('.csv')]
    if not bronze_files:
        print("❌ No bronze files found to process.")
    else:
        for fname in bronze_files:
            bronze_path = os.path.join(bronze_dir, fname)
            raw_path = os.path.join(raw_dir, fname)
            features_path = os.path.join(features_dir, fname)
            outcomes_path = os.path.join(outcomes_dir, fname)
            
            if not os.path.exists(raw_path):
                print(f"⚠️ SKIPPING: Raw file for {fname} not found."); continue
            
            try:
                create_decoupled_silver_data(bronze_path, raw_path, features_path, outcomes_path)
            except Exception as e:
                print(f"❌ FAILED to process {fname}. Error: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "="*50 + "\n✅ All silver data generation complete.")
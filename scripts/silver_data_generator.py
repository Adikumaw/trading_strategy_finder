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
    df = pd.read_csv(filepath, sep=None, engine="python", header=None)
    if df.shape[1] > 5:
        df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
    elif df.shape[1] == 5:
        df.columns = ['time', 'open', 'high', 'low', 'close']; df['volume'] = 1
    else: raise ValueError(f"File '{filepath}' has fewer than 5 columns.")
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=['time'] + numeric_cols, inplace=True)
    return df.sort_values('time').reset_index(drop=True)

@numba.njit
def _calculate_s_r_numba(lows, highs, window):
    n = len(lows)
    support_points, resistance_points = np.full(n, np.nan, dtype=np.float32), np.full(n, np.nan, dtype=np.float32)
    for i in range(window, n - window):
        window_lows = lows[i - window : i + window + 1]
        window_highs = highs[i - window : i + window + 1]
        if lows[i] == np.min(window_lows): support_points[i] = lows[i]
        if highs[i] == np.max(window_highs): resistance_points[i] = highs[i]
    return support_points, resistance_points

def add_support_resistance(df, window=PIVOT_WINDOW):
    lows, highs = df["low"].values.astype(np.float32), df["high"].values.astype(np.float32)
    support_points, resistance_points = _calculate_s_r_numba(lows, highs, window)
    sr_df = pd.DataFrame({'support_points': support_points, 'resistance_points': resistance_points}, index=df.index)
    sr_df["support"], sr_df["resistance"] = sr_df["support_points"].ffill(), sr_df["resistance_points"].ffill()
    return pd.concat([df, sr_df[['support', 'resistance']]], axis=1)

def add_all_market_features(df):
    """Calculates all market-based features for each candle efficiently."""
    new_features_list = []
    
    # --- Batch 1: Standard Indicators ---
    print("Calculating standard indicators...")
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
    
    # Add ATR Dynamic Levels to the indicator batch
    atr_series = indicator_df[f"ATR_{ATR_PERIOD}"]
    indicator_df["ATR_level_up_1x"] = df["close"] + atr_series
    indicator_df["ATR_level_down_1x"] = df["close"] - atr_series
    new_features_list.append(indicator_df)

    # --- Batch 2: Candlestick Patterns ---
    print("Calculating candlestick patterns...")
    patterns_data = {p: getattr(talib, p)(df["open"], df["high"], df["low"], df["close"]) for p in talib.get_function_groups().get("Pattern Recognition", [])}
    new_features_list.append(pd.DataFrame(patterns_data, index=df.index))
    
    # --- Batch 3: Support & Resistance ---
    print("Calculating support and resistance...")
    df_with_sr = add_support_resistance(df.copy())
    new_features_list.append(df_with_sr[['support', 'resistance']])

    # --- Batch 4: Time-based and Price-Action Features ---
    print("Calculating time-based and price-action features...")
    price_action_df = pd.DataFrame(index=df.index)
    price_action_df['session'] = df['time'].dt.hour.map(lambda h: 'London' if 7<=h<12 else 'London_NY_Overlap' if 12<=h<16 else 'New_York' if 16<=h<21 else 'Asian')
    price_action_df['hour'], price_action_df['weekday'] = df['time'].dt.hour, df['time'].dt.weekday
    is_bullish, body_size = (df['close'] > df['open']).astype(int), (df['close'] - df['open']).abs()
    for n in PAST_LOOKBACKS:
        price_action_df[f'bullish_ratio_last_{n}'] = is_bullish.rolling(n).mean()
        price_action_df[f'avg_body_last_{n}'] = body_size.rolling(n).mean()
        price_action_df[f'avg_range_last_{n}'] = (df['high'] - df['low']).rolling(n).mean()
        # Ensure SMA_20 is used as a fallback if a specific SMA period doesn't exist
        sma_col = f'SMA_{n}' if f'SMA_{n}' in indicator_df else 'SMA_20'
        price_action_df[f'close_{sma_col}_ratio'] = df['close'] / indicator_df[sma_col]
    price_action_df['trend_regime'] = np.where(indicator_df['ADX'] > 25, 'trend', 'range')
    if f'ATR_{ATR_PERIOD}' in indicator_df: price_action_df['vol_regime'] = np.where(indicator_df[f'ATR_{ATR_PERIOD}'] > indicator_df[f'ATR_{ATR_PERIOD}'].rolling(50).mean(), 'high_vol', 'low_vol')
    new_features_list.append(price_action_df)

    return pd.concat([df] + new_features_list, axis=1)

def add_positioning_features(merged_chunk):
    """
    Calculates the distance from SL/TP to key indicator levels, expressed in basis points (bps).
    1 basis point = 0.01%.
    A positive value means the price (SL/TP) is ABOVE the indicator level.
    A negative value means the price (SL/TP) is BELOW the indicator level.
    """
    # Identify all possible price levels for positioning
    level_cols = [
        'open', 'high', 'low', 'support', 'resistance',
        'BB_upper', 'BB_lower', 'ATR_level_up_1x', 'ATR_level_down_1x'
    ]
    level_cols.extend([col for col in merged_chunk.columns if 'SMA_' in col or 'EMA_' in col])
    
    sl, tp, close = merged_chunk['sl_price'], merged_chunk['tp_price'], merged_chunk['close']
    
    for level in level_cols:
        if level in merged_chunk.columns and not merged_chunk[level].isnull().all():
            # Calculate distance as a ratio, then scale by 10,000 to get basis points
            merged_chunk[f'sl_dist_to_{level}_bps'] = ((sl - merged_chunk[level]) / close) * 10000
            merged_chunk[f'tp_dist_to_{level}_bps'] = ((tp - merged_chunk[level]) / close) * 10000
            
    return merged_chunk

def create_enriched_silver_data(bronze_path, raw_path, features_path, outcomes_path):
    print(f"\n{'='*25}\nProcessing: {os.path.basename(raw_path)}\n{'='*25}")

    # --- STEP 1: Create and Save the Silver Features Dataset ---
    print("STEP 1: Creating Silver Features dataset...")
    raw_df = robust_read_csv(raw_path)
    if len(raw_df) < INDICATOR_WARMUP_PERIOD + 1:
        print(f"❌ SKIPPING: Not enough data for indicator warmup."); return

    features_df = add_all_market_features(raw_df)
    del raw_df; gc.collect()
    
    features_df = features_df.iloc[INDICATOR_WARMUP_PERIOD:].reset_index(drop=True)
    features_df = downcast_dtypes(features_df)
    
    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    features_df.to_csv(features_path, index=False)
    print(f"✅ Silver Features saved to: {features_path}")
    
    # --- STEP 2: Create the Enriched Silver Outcomes Dataset ---
    print("\nSTEP 2: Creating ENRICHED Silver Outcomes dataset...")
    bronze_iterator = pd.read_csv(bronze_path, chunksize=500_000, parse_dates=['entry_time'])
    is_first_chunk = True
    os.makedirs(os.path.dirname(outcomes_path), exist_ok=True)
    if os.path.exists(outcomes_path): os.remove(outcomes_path)

    # Define the columns to keep. All other feature columns will be dropped after merging.
    # We keep OHLC for context and all indicator levels needed for positioning calculations.
    indicator_levels = [
        'support', 'resistance', 'BB_upper', 'BB_lower', 'ATR_level_up_1x', 'ATR_level_down_1x'
    ] + [f"SMA_{p}" for p in SMA_PERIODS] + [f"EMA_{p}" for p in EMA_PERIODS]
    
    cols_to_keep = ['time', 'open', 'high', 'low', 'close'] + indicator_levels

    for chunk in tqdm(bronze_iterator, desc="Enriching Bronze Chunks"):
        chunk = chunk[chunk['entry_time'] >= features_df['time'].min()]
        if chunk.empty: continue
        
        # Merge only the necessary columns from the features_df to save memory
        merged_chunk = pd.merge_asof(
            chunk.sort_values('entry_time'), 
            features_df[cols_to_keep], 
            left_on='entry_time', 
            right_on='time', 
            direction='backward'
        )
        
        enriched_chunk = add_positioning_features(merged_chunk)
        
        # Drop the indicator level columns, as their information is now encoded in the '_bps' features
        enriched_chunk.drop(columns=indicator_levels + ['time'], inplace=True, errors='ignore')
        
        if not enriched_chunk.empty:
            enriched_chunk = downcast_dtypes(enriched_chunk)
            enriched_chunk.to_csv(outcomes_path, mode='a', header=is_first_chunk, index=False)
            is_first_chunk = False
            
    del features_df; gc.collect()
    print(f"✅ Enriched Silver Outcomes saved to: {outcomes_path}")

if __name__ == "__main__":
    core_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir, bronze_dir = [os.path.abspath(os.path.join(core_dir, '..', d)) for d in ['raw_data', 'bronze_data']]
    features_dir = os.path.abspath(os.path.join(core_dir, '..', 'silver_data', 'features'))
    outcomes_dir = os.path.abspath(os.path.join(core_dir, '..', 'silver_data', 'outcomes'))
    
    os.makedirs(features_dir, exist_ok=True); os.makedirs(outcomes_dir, exist_ok=True)

    bronze_files = [f for f in os.listdir(bronze_dir) if f.endswith('.csv')]
    if not bronze_files: print("❌ No bronze files found to process.")
    else:
        for fname in bronze_files:
            bronze_path, raw_path = [os.path.join(d, fname) for d in [bronze_dir, raw_dir]]
            features_path, outcomes_path = [os.path.join(d, fname) for d in [features_dir, outcomes_dir]]
            
            if not os.path.exists(raw_path):
                print(f"⚠️ SKIPPING: Raw file for {fname} not found."); continue
            
            # Simple check to skip already processed files
            if os.path.exists(outcomes_path):
                print(f"✅ SKIPPING: Silver outcomes file for {fname} already exists."); continue
            
            try:
                create_enriched_silver_data(bronze_path, raw_path, features_path, outcomes_path)
            except Exception as e:
                print(f"❌ FAILED to process {fname}. Error: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "="*50 + "\n✅ All silver data generation complete.")
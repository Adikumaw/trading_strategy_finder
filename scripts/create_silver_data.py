import os
import pandas as pd
import ta
import talib
import numba
import numpy as np
from tqdm import tqdm

# --- CONFIGURATION (Unchanged) ---
SMA_PERIODS = [20, 50, 100, 200]
EMA_PERIODS = [8, 13, 21, 50]
BBANDS_PERIOD, BBANDS_STD_DEV = 20, 2.0
RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
ATR_PERIOD = 14
ADX_PERIOD = 14
SR_LOOKBACK = 200
PAST_LOOKBACKS = [3, 5, 10, 20, 50]

# ### --- OPTIMIZATION 1: MEMORY REDUCTION HELPER --- ###
def downcast_dtypes(df):
    """Reduces DataFrame memory usage by downcasting numeric columns."""
    print("  Downcasting data types for memory optimization...")
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        if df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    end_mem = df.memory_usage().sum() / 1024**2
    print(f"  Memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB")
    return df

# --- FEATURE CALCULATION FUNCTIONS (Now More Efficient) ---

def add_indicators(df):
    print("  Calculating standard indicators...")
    # This function is already efficient
    for period in SMA_PERIODS: df[f"SMA_{period}"] = ta.trend.SMAIndicator(df["close"], window=period).sma_indicator()
    for period in EMA_PERIODS: df[f"EMA_{period}"] = ta.trend.EMAIndicator(df["close"], window=period).ema_indicator()
    bb = ta.volatility.BollingerBands(df["close"], window=BBANDS_PERIOD, window_dev=BBANDS_STD_DEV)
    df["BB_upper"], df["BB_lower"], df["BB_width"] = bb.bollinger_hband(), bb.bollinger_lband(), bb.bollinger_wband()
    df[f"RSI_{RSI_PERIOD}"] = ta.momentum.RSIIndicator(df["close"], window=RSI_PERIOD).rsi()
    macd = ta.trend.MACD(df["close"], window_slow=MACD_SLOW, window_fast=MACD_FAST, window_sign=MACD_SIGNAL)
    df["MACD_hist"] = macd.macd_diff()
    df[f"ATR_{ATR_PERIOD}"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=ATR_PERIOD).average_true_range()
    df["ADX"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=ADX_PERIOD).adx()
    return df

def add_candlestick_patterns(df):
    print("  Calculating all TA-Lib candlestick patterns...")
    candle_names = talib.get_function_groups()["Pattern Recognition"]
    # ### OPTIMIZATION 2: FIX FRAGMENTATION ###
    # Create all pattern columns in a dictionary first
    patterns_data = {}
    for candle_func in candle_names:
        patterns_data[candle_func] = getattr(talib, candle_func)(df["open"], df["high"], df["low"], df["close"])
    
    # Convert to a DataFrame and join in one operation
    patterns_df = pd.DataFrame(patterns_data, index=df.index)
    return pd.concat([df, patterns_df], axis=1)

@numba.njit
def _calculate_s_r_numba(lows, highs, support_indices, support_values, resistance_indices, resistance_values, lookback):
    n = len(lows)
    supports, resistances = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        start_idx = max(0, i - lookback); current_low, current_high = lows[i], highs[i]
        max_s, min_r = -np.inf, np.inf
        for j in range(len(support_indices)):
            s_idx, s_val = support_indices[j], support_values[j]
            if s_idx >= start_idx and s_idx < i and s_val < current_low and s_val > max_s: max_s = s_val
        if max_s != -np.inf: supports[i] = max_s
        for j in range(len(resistance_indices)):
            r_idx, r_val = resistance_indices[j], resistance_values[j]
            if r_idx >= start_idx and r_idx < i and r_val > current_high and r_val < min_r: min_r = r_val
        if min_r != np.inf: resistances[i] = min_r
    return supports, resistances

def add_support_resistance(df, lookback=SR_LOOKBACK):
    print(f"  Calculating support & resistance...")
    lows, highs = df['low'].values, df['high'].values
    is_support = (lows < np.roll(lows, 1)) & (lows < np.roll(lows, 2)) & (lows < np.roll(lows, -1)) & (lows < np.roll(lows, -2))
    is_resistance = (highs > np.roll(highs, 1)) & (highs > np.roll(highs, 2)) & (highs > np.roll(highs, -1)) & (highs > np.roll(highs, -2))
    support_indices, resistance_indices = np.where(is_support)[0], np.where(is_resistance)[0]
    support_values, resistance_values = lows[support_indices], highs[resistance_indices]
    supports, resistances = _calculate_s_r_numba(lows, highs, support_indices, support_values, resistance_indices, resistance_values, lookback)
    df['support'], df['resistance'] = supports, resistances
    return df

def get_market_session(timestamp):
    hour = timestamp.hour
    if (hour >= 7 and hour < 12): return 'London'
    if (hour >= 12 and hour < 16): return 'London_NY_Overlap'
    if (hour >= 16 and hour < 21): return 'New_York'
    if (hour >= 21 or hour < 7): return 'Asian'
    return 'Off-Session'

def add_sessions(df):
    print("  Identifying market sessions...")
    df['session'] = df['time'].apply(get_market_session)
    return df

def add_past_candle_features(df, lookbacks=PAST_LOOKBACKS):
    print(f"  Analyzing past candle formations...")
    df['is_bullish'] = (df['close'] > df['open']).astype(int)
    df['body_size'] = (df['close'] - df['open']).abs()
    # ### OPTIMIZATION 2: FIX FRAGMENTATION ###
    new_cols = {}
    for n in lookbacks:
        new_cols[f'bullish_ratio_last_{n}'] = df['is_bullish'].rolling(window=n).mean()
        new_cols[f'avg_body_last_{n}'] = df['body_size'].rolling(window=n).mean()
        new_cols[f'avg_range_last_{n}'] = (df['high'] - df['low']).rolling(window=n).mean()
    
    past_df = pd.DataFrame(new_cols, index=df.index)
    df = pd.concat([df, past_df], axis=1)
    return df.drop(columns=['is_bullish', 'body_size'])

# --- MAIN PROCESSING FUNCTION (Heavily Modified for Memory Efficiency) ---

def create_silver_data(bronze_file, raw_file, silver_file):
    print(f"Loading bronze trade data from: {os.path.basename(bronze_file)}")
    # Only load the column we absolutely need
    bronze_df = pd.read_csv(bronze_file, usecols=['entry_time', 'trade_type', 'sl_ratio', 'tp_ratio'])
    bronze_df['entry_time'] = pd.to_datetime(bronze_df['entry_time'])

    print(f"Loading raw OHLCV data from: {os.path.basename(raw_file)}")
    raw_df = pd.read_csv(raw_file, sep=None, engine="python")
    if raw_df.shape[1] > 5: raw_df = raw_df.iloc[:, :5]
    raw_df.columns = ["time", "open", "high", "low", "close"]
    raw_df["time"] = pd.to_datetime(raw_df["time"])
    raw_df[["open", "high", "low", "close"]] = raw_df[["open", "high", "low", "close"]].apply(pd.to_numeric)
    
    # --- Calculate All Features on the Raw Data ---
    features_df = raw_df
    features_df = add_indicators(features_df)
    features_df = add_candlestick_patterns(features_df)
    features_df = add_support_resistance(features_df)
    features_df = add_sessions(features_df)
    features_df = add_past_candle_features(features_df)
    features_df = downcast_dtypes(features_df)

    # ### --- OPTIMIZATION 3: FILTER BEFORE MERGING --- ###
    print("\nFiltering features to match trade entry times...")
    # Get a unique list of timestamps we need features for
    required_times = bronze_df['entry_time'].unique()
    
    # Create a much smaller features DataFrame that only contains rows for our trades
    filtered_features_df = features_df[features_df['time'].isin(required_times)]
    
    print("Merging filtered features with profitable trade data...")
    # Now the merge is much smaller and faster
    silver_df = pd.merge(
        left=bronze_df, right=filtered_features_df,
        left_on='entry_time', right_on='time',
        how='inner'
    )
    silver_df.drop(columns=['time'], inplace=True)
    
    print(f"  Original bronze trades: {len(bronze_df)}")
    print(f"  Trades after merging: {len(silver_df)}")
    
    silver_df.dropna(inplace=True)
    print(f"  Trades after removing rows with NaN values: {len(silver_df)}")

    if silver_df.empty:
        print("⚠️ No data remaining after processing.")
        return
        
    silver_df = downcast_dtypes(silver_df)
    silver_df.to_csv(silver_file, index=False)
    print(f"✅ Success! Rich dataset with {silver_df.shape[1]} features saved to {os.path.basename(silver_file)}")


# --- MAIN EXECUTION BLOCK (Unchanged) ---
if __name__ == "__main__":
    core_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.abspath(os.path.join(core_dir, '..', 'raw_data'))
    bronze_data_dir = os.path.abspath(os.path.join(core_dir, '..', 'bronze_data'))
    silver_data_dir = os.path.abspath(os.path.join(core_dir, '..', 'silver_data'))
    os.makedirs(silver_data_dir, exist_ok=True)
    
    try:
        bronze_files = [f for f in os.listdir(bronze_data_dir) if f.endswith('.csv')]
    except FileNotFoundError:
        print(f"❌ Error: The directory '{bronze_data_dir}' was not found.")
        bronze_files = []

    if not bronze_files: 
        print("❌ No CSV files found in 'bronze_data'.")
    else: 
        print(f"Found {len(bronze_files)} files to process...")

    for filename in bronze_files:
        bronze_path = os.path.join(bronze_data_dir, filename)
        raw_path = os.path.join(raw_data_dir, filename)
        silver_path = os.path.join(silver_data_dir, filename)
        
        print("\n" + "="*50 + f"\nProcessing: {filename}")
        
        if not os.path.exists(raw_path):
            print(f"⚠️ SKIPPING: Corresponding raw data file not found at '{raw_path}'")
            continue
            
        try:
            create_silver_data(bronze_file=bronze_path, raw_file=raw_path, silver_file=silver_path)
        except Exception as e:
            print(f"❌ FAILED to process {filename}. Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*50 + "\nSilver data generation complete.")
import os
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
REMOVE_INITIAL_ROWS = 200  # Remove first N rows for accurate indicators

# --- MEMORY OPTIMIZATION ---
def downcast_dtypes(df):
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        if df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    return df

# --- INDICATORS ---
def add_indicators(df):
    print("  Calculating indicators...")
    for period in SMA_PERIODS:
        print(f"    Calculating SMA_{period}...")
        df[f"SMA_{period}"] = ta.trend.SMAIndicator(df["close"], period).sma_indicator()
    for period in EMA_PERIODS:
        print(f"    Calculating EMA_{period}...")
        df[f"EMA_{period}"] = ta.trend.EMAIndicator(df["close"], period).ema_indicator()
    
    print("    Calculating Bollinger Bands...")
    bb = ta.volatility.BollingerBands(df["close"], BBANDS_PERIOD, BBANDS_STD_DEV)
    df["BB_upper"], df["BB_lower"], df["BB_width"] = bb.bollinger_hband(), bb.bollinger_lband(), bb.bollinger_wband()
    
    print("    Calculating RSI...")
    df[f"RSI_{RSI_PERIOD}"] = ta.momentum.RSIIndicator(df["close"], RSI_PERIOD).rsi()
    
    print("    Calculating MACD histogram...")
    macd = ta.trend.MACD(df["close"], MACD_SLOW, MACD_FAST, MACD_SIGNAL)
    df["MACD_hist"] = macd.macd_diff()
    
    print("    Calculating ATR & ADX...")
    df[f"ATR_{ATR_PERIOD}"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], ATR_PERIOD).average_true_range()
    df["ADX"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], ADX_PERIOD).adx()
    
    print("    Calculating additional indicators (MOM, CCI, OBV)...")
    df["MOM_10"] = ta.momentum.ROCIndicator(df["close"], window=10).roc()
    df["CCI_20"] = ta.trend.CCIIndicator(df["high"], df["low"], df["close"], window=20).cci()
    df["OBV"] = ta.volume.OnBalanceVolumeIndicator(df["close"], df.get("volume", pd.Series(1))).on_balance_volume()
    
    print("  ✅ Indicators calculation done.")
    return df

# --- CANDLESTICKS ---
def add_candlestick_patterns(df):
    print("  Calculating TA-Lib candlestick patterns...")
    patterns = talib.get_function_groups()["Pattern Recognition"]
    for p in tqdm(patterns, desc="    Candlestick patterns"):
        df[p] = getattr(talib, p)(df["open"], df["high"], df["low"], df["close"])
    print("  ✅ Candlestick patterns done.")
    return df

# --- SUPPORT / RESISTANCE ---
@numba.njit
def _calculate_s_r_numba(lows, highs, support_indices, support_values, resistance_indices, resistance_values, lookback):
    n = len(lows)
    supports, resistances = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        start_idx = max(0, i - lookback)
        current_low, current_high = lows[i], highs[i]
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

# --- SUPPORT / RESISTANCE ---
def add_support_resistance(df, lookback=SR_LOOKBACK):
    print("  Calculating support and resistance...")
    lows, highs = df['low'].values, df['high'].values
    is_support = (lows < np.roll(lows, 1)) & (lows < np.roll(lows, 2)) & (lows < np.roll(lows, -1)) & (lows < np.roll(lows, -2))
    is_resistance = (highs > np.roll(highs, 1)) & (highs > np.roll(highs, 2)) & (highs > np.roll(highs, -1)) & (highs > np.roll(highs, -2))
    support_idx, resistance_idx = np.where(is_support)[0], np.where(is_resistance)[0]
    supports, resistances = _calculate_s_r_numba(lows, highs, support_idx, lows[support_idx], resistance_idx, highs[resistance_idx], lookback)
    df['support'], df['resistance'] = supports, resistances
    print("  ✅ Support & resistance done.")
    return df

# --- SESSIONS & TEMPORALS ---
def get_market_session(ts):
    h = ts.hour
    if 7 <= h < 12: return 'London'
    if 12 <= h < 16: return 'London_NY_Overlap'
    if 16 <= h < 21: return 'New_York'
    return 'Asian'

def add_sessions(df):
    df['session'] = df['time'].apply(get_market_session)
    df['hour'] = df['time'].dt.hour
    df['weekday'] = df['time'].dt.weekday
    return df

# --- PAST FEATURES ---
def add_past_features(df, lookbacks=PAST_LOOKBACKS):
    print("  Calculating past candle features...")
    df['is_bullish'] = (df['close'] > df['open']).astype(int)
    df['body_size'] = (df['close'] - df['open']).abs()
    new_cols = {}
    for n in tqdm(lookbacks, desc="    Rolling features"):
        new_cols[f'bullish_ratio_{n}'] = df['is_bullish'].rolling(n).mean()
        new_cols[f'avg_body_{n}'] = df['body_size'].rolling(n).mean()
        new_cols[f'avg_range_{n}'] = (df['high'] - df['low']).rolling(n).mean()
        new_cols[f'close_SMA20_ratio_{n}'] = df['close'].rolling(n).mean() / df['close']
        new_cols[f'EMA8_EMA21_ratio_{n}'] = df['EMA_8'].rolling(n).mean() / df['EMA_21'].rolling(n).mean()
    print("  ✅ Past features done.")
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1).drop(columns=['is_bullish', 'body_size'])

# --- REGIME FEATURES ---
def add_regimes(df):
    df['trend_regime'] = np.where(df['ADX'] > 25, 'trend', 'range')
    df['vol_regime'] = np.where(df['ATR_14'] > df['ATR_14'].rolling(20).mean(), 'high_vol', 'low_vol')
    return df

# --- HIGHER TIMEFRAME FEATURES ---
def add_htf_features(df, htf_file=None):
    if htf_file is None:
        print("⚠️ No higher timeframe file provided. Skipping HTF features.")
        return df

    print(f"  Loading higher timeframe data: {htf_file}")
    htf_df = pd.read_csv(htf_file, sep=None, engine="python")
    if htf_df.shape[1] > 5: htf_df = htf_df.iloc[:, :5]
    htf_df.columns = ['time','open','high','low','close']
    htf_df['time'] = pd.to_datetime(htf_df['time'])
    
    # Add some HTF indicators (SMA/EMA) if not present
    htf_df = add_indicators(htf_df)

    # Keep only the columns we want to merge
    htf_df = htf_df[['time','SMA_50','EMA_21']].copy()

    # Sort by time
    htf_df = htf_df.sort_values('time')
    df = df.sort_values('time')

    # Forward-fill HTF values to lower timeframe
    htf_df = htf_df.set_index('time').reindex(df['time'], method='ffill').reset_index()
    df = pd.concat([df.reset_index(drop=True), htf_df[['SMA_50','EMA_21']].reset_index(drop=True)], axis=1)
    
    print("  ✅ HTF features merged with forward-fill.")
    return df


# --- MAIN PROCESSING ---
def create_silver_data(bronze_file, raw_file, silver_file, htf_file=None):
    print(f"\n=== Processing file: {os.path.basename(bronze_file)} ===")
    
    print("Loading bronze data...")
    bronze_df = pd.read_csv(bronze_file)
    bronze_df['entry_time'] = pd.to_datetime(bronze_df['entry_time'])
    print(f"  Bronze data loaded: {len(bronze_df)} rows")
    
    print("Loading raw OHLCV data...")
    df = pd.read_csv(raw_file, sep=None, engine='python')
    if df.shape[1] > 5: df = df.iloc[:, :5]
    df.columns = ['time','open','high','low','close']
    df['time'] = pd.to_datetime(df['time'])
    df[["open","high","low","close"]] = df[["open","high","low","close"]].apply(pd.to_numeric)
    print(f"  Raw OHLCV loaded: {len(df)} rows")
    
    print(f"Removing first {REMOVE_INITIAL_ROWS} rows for indicators...")
    df = df.iloc[REMOVE_INITIAL_ROWS:].reset_index(drop=True)
    
    # Feature calculations
    df = add_indicators(df)
    df = add_candlestick_patterns(df)
    df = add_support_resistance(df)
    df = add_sessions(df)
    df = add_past_features(df)
    df = add_regimes(df)
    df = downcast_dtypes(df)
    
    print("Merging with bronze trade times...")
    df = df.sort_values('time')
    bronze_df = bronze_df.sort_values('entry_time')
    silver_df = pd.merge_asof(
        bronze_df,
        df,
        left_on='entry_time',
        right_on='time',
        direction='backward'
    )
    silver_df.drop(columns=['time'], inplace=True)
    print(f"  Silver dataset created: {len(silver_df)} rows, {len(silver_df.columns)} columns")
    
    silver_df.to_csv(silver_file, index=False)
    print(f"✅ Silver dataset saved: {silver_file}\n")

# --- EXECUTION ---
if __name__ == "__main__":
    core_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.abspath(os.path.join(core_dir,'..','raw_data'))
    bronze_dir = os.path.abspath(os.path.join(core_dir,'..','bronze_data'))
    silver_dir = os.path.abspath(os.path.join(core_dir,'..','silver_data'))
    os.makedirs(silver_dir, exist_ok=True)
    
    bronze_files = [f for f in os.listdir(bronze_dir) if f.endswith('.csv')]
    
    for fname in bronze_files:
        bronze_path = os.path.join(bronze_dir, fname)
        raw_path = os.path.join(raw_dir, fname)
        silver_path = os.path.join(silver_dir, fname)
        create_silver_data(bronze_path, raw_path, silver_path)
    
    print("✅ Silver data generation complete!")

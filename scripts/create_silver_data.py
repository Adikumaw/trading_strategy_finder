import os
import pandas as pd
import ta
import gc
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
REMOVE_INITIAL_ROWS = 200

# --- MEMORY OPTIMIZATION ---
def downcast_dtypes(df):
    """Downcast numeric dtypes to save RAM."""
    for col in df.columns:
        try:
            if df[col].dtype == 'float64':
                df[col] = df[col].astype('float32')
            if df[col].dtype == 'int64':
                df[col] = df[col].astype('int32')
        except Exception:
            # skip columns we cannot convert
            continue
    return df

# --- INDICATORS ---
def add_indicators(df):
    print("  Calculating indicators...")
    for period in SMA_PERIODS:
        print(f"    SMA_{period}")
        df[f"SMA_{period}"] = ta.trend.SMAIndicator(df["close"], period).sma_indicator().astype('float32')
    for period in EMA_PERIODS:
        print(f"    EMA_{period}")
        df[f"EMA_{period}"] = ta.trend.EMAIndicator(df["close"], period).ema_indicator().astype('float32')

    print("    Bollinger Bands...")
    bb = ta.volatility.BollingerBands(df["close"], BBANDS_PERIOD, BBANDS_STD_DEV)
    df["BB_upper"] = bb.bollinger_hband().astype('float32')
    df["BB_lower"] = bb.bollinger_lband().astype('float32')
    df["BB_width"] = bb.bollinger_wband().astype('float32')

    print("    RSI...")
    df[f"RSI_{RSI_PERIOD}"] = ta.momentum.RSIIndicator(df["close"], RSI_PERIOD).rsi().astype('float32')

    print("    MACD hist...")
    macd = ta.trend.MACD(df["close"], MACD_SLOW, MACD_FAST, MACD_SIGNAL)
    df["MACD_hist"] = macd.macd_diff().astype('float32')

    print("    ATR & ADX...")
    df[f"ATR_{ATR_PERIOD}"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], ATR_PERIOD).average_true_range().astype('float32')
    df["ADX"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], ADX_PERIOD).adx().astype('float32')

    print("    MOM, CCI, OBV...")
    df["MOM_10"] = ta.momentum.ROCIndicator(df["close"], window=10).roc().astype('float32')
    df["CCI_20"] = ta.trend.CCIIndicator(df["high"], df["low"], df["close"], window=20).cci().astype('float32')

    # OBV: if no volume column, use NaNs (avoid artificial 1s)
    if "volume" in df.columns:
        df["OBV"] = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume().astype('float32')
    else:
        df["OBV"] = np.nan
    print("  ✅ Indicators done")
    return df

# --- CANDLESTICKS ---
def add_candlestick_patterns(df):
    print("  Calculating candlestick patterns...")
    patterns = talib.get_function_groups().get("Pattern Recognition", [])
    for p in tqdm(patterns, desc="    Patterns"):
        try:
            df[p] = getattr(talib, p)(df["open"], df["high"], df["low"], df["close"]).astype('float32')
        except Exception:
            df[p] = np.nan
    print("  ✅ Candlestick patterns done")
    return df

# --- SUPPORT / RESISTANCE (numba-accelerated) ---
@numba.njit
def _calculate_s_r_numba(lows, highs, support_indices, support_values, resistance_indices, resistance_values, lookback):
    n = len(lows)
    supports = np.full(n, np.nan)
    resistances = np.full(n, np.nan)
    for i in range(n):
        start_idx = max(0, i - lookback)
        max_s, min_r = -np.inf, np.inf
        for j in range(len(support_indices)):
            s_idx = support_indices[j]; s_val = support_values[j]
            if s_idx >= start_idx and s_idx < i and s_val < lows[i] and s_val > max_s:
                max_s = s_val
        if max_s != -np.inf:
            supports[i] = max_s
        for j in range(len(resistance_indices)):
            r_idx = resistance_indices[j]; r_val = resistance_values[j]
            if r_idx >= start_idx and r_idx < i and r_val > highs[i] and r_val < min_r:
                min_r = r_val
        if min_r != np.inf:
            resistances[i] = min_r
    return supports, resistances

def add_support_resistance(df, lookback=SR_LOOKBACK):
    print("  Calculating support/resistance...")
    lows = df['low'].values
    highs = df['high'].values
    is_support = (lows < np.roll(lows, 1)) & (lows < np.roll(lows, 2)) & (lows < np.roll(lows, -1)) & (lows < np.roll(lows, -2))
    is_resistance = (highs > np.roll(highs, 1)) & (highs > np.roll(highs, 2)) & (highs > np.roll(highs, -1)) & (highs > np.roll(highs, -2))
    support_idx = np.where(is_support)[0]
    resistance_idx = np.where(is_resistance)[0]

    # handle empty
    if support_idx.size == 0:
        supports = np.full(len(lows), np.nan)
    if resistance_idx.size == 0:
        resistances = np.full(len(highs), np.nan)

    if support_idx.size > 0 and resistance_idx.size > 0:
        supports, resistances = _calculate_s_r_numba(lows, highs, support_idx, lows[support_idx], resistance_idx, highs[resistance_idx], lookback)
    else:
        # compute each side separately (call numba with empty arrays isn't problematic but handle cleanly)
        supports, _ = _calculate_s_r_numba(lows, highs, support_idx, lows[support_idx] if support_idx.size>0 else np.array([], dtype=np.float64), np.array([], dtype=np.int64), np.array([], dtype=np.float64), lookback)
        _, resistances = _calculate_s_r_numba(lows, highs, np.array([], dtype=np.int64), np.array([], dtype=np.float64), resistance_idx, highs[resistance_idx] if resistance_idx.size>0 else np.array([], dtype=np.float64), lookback)

    df['support'] = supports.astype('float32')
    df['resistance'] = resistances.astype('float32')
    print("  ✅ Support/resistance done")
    return df

# --- SESSIONS & TEMPORALS ---
def get_market_session(ts):
    h = ts.hour
    if 7 <= h < 12: return 'London'
    if 12 <= h < 16: return 'London_NY_Overlap'
    if 16 <= h < 21: return 'New_York'
    return 'Asian'

def add_sessions(df):
    print("  Adding session & temporal features...")
    df['session'] = df['time'].apply(get_market_session)
    df['hour'] = df['time'].dt.hour.astype('int8')
    df['weekday'] = df['time'].dt.weekday.astype('int8')
    return df

# --- PAST CANDLES & RELATIONAL FEATURES ---
def add_past_features(df):
    print("  Calculating past candle relational features...")
    df['is_bullish'] = (df['close'] > df['open']).astype('int8')
    df['body_size'] = (df['close'] - df['open']).abs().astype('float32')
    new_cols = {}
    for n in PAST_LOOKBACKS:
        new_cols[f'bullish_ratio_{n}'] = df['is_bullish'].rolling(n).mean().astype('float32')
        new_cols[f'avg_body_{n}'] = df['body_size'].rolling(n).mean().astype('float32')
        new_cols[f'avg_range_{n}'] = (df['high'] - df['low']).rolling(n).mean().astype('float32')
        # relational (avoid divide-by-zero)
        sma_close = df['close'].rolling(n).mean()
        ema8 = df['EMA_8'].rolling(n).mean()
        ema21 = df['EMA_21'].rolling(n).mean()
        new_cols[f'close_SMA20_ratio_{n}'] = (sma_close / df['close']).astype('float32')
        new_cols[f'EMA8_EMA21_ratio_{n}'] = (ema8 / ema21).astype('float32')
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    df.drop(columns=['is_bullish','body_size'], inplace=True)
    print("  ✅ Past features done")
    return df

# --- REGIME FEATURES ---
def add_regimes(df):
    print("  Adding regime flags...")
    df['trend_regime'] = np.where(df['ADX'] > 25, 'trend', 'range')
    df['vol_regime'] = np.where(df['ATR_14'] > df['ATR_14'].rolling(20).mean(), 'high_vol', 'low_vol')
    return df

# --- HIGHER TIMEFRAME FEATURES (forward-fill alignment) ---
def add_htf_features(df, htf_file=None):
    if htf_file is None:
        print("⚠️ No higher timeframe file provided. Skipping HTF features.")
        return df

    print(f"  Loading higher timeframe data: {htf_file}")
    # robust read
    try:
        htf_df = pd.read_csv(htf_file, sep=None, engine="python", header=0)
    except Exception:
        htf_df = pd.read_csv(htf_file, sep=None, engine="python", header=None)

    # keep first 5 columns
    htf_df = htf_df.iloc[:, :5].copy()
    htf_df.columns = ['time','open','high','low','close']
    htf_df['time'] = pd.to_datetime(htf_df['time'], errors='coerce')
    htf_df.dropna(subset=['time','open','high','low','close'], inplace=True)
    htf_df[['open','high','low','close']] = htf_df[['open','high','low','close']].apply(pd.to_numeric, errors='coerce')
    htf_df.dropna(subset=['open','high','low','close'], inplace=True)

    print("  Calculating HTF indicators (this may take a bit)...")
    htf_df = add_indicators(htf_df)
    # choose HTF cols to merge (rename to indicate HTF)
    htf_df = htf_df[['time','SMA_50','EMA_21']].rename(columns={'SMA_50':'SMA_50_HTF','EMA_21':'EMA_21_HTF'})

    # align & forward-fill: reindex HTF on df.time and ffill
    htf_df = htf_df.sort_values('time').set_index('time')
    target_idx = pd.Index(df['time'].values)
    # reindex (creates index same size as df), then forward-fill
    htf_reindexed = htf_df.reindex(target_idx, method='ffill').reset_index()
    htf_reindexed.columns = ['time','SMA_50_HTF','EMA_21_HTF']

    # concat HTF columns to df (index aligned)
    df = df.reset_index(drop=True).sort_values('time').reset_index(drop=True)
    df = pd.concat([df, htf_reindexed[['SMA_50_HTF','EMA_21_HTF']].reset_index(drop=True)], axis=1)
    print("  ✅ HTF features merged (forward-filled).")
    return df

# --- SAFE CSV LOADING UTIL (raw/bronze) ---
def robust_read_ohlcv(filepath):
    """Read file trying header=0 first, then header=None; keep first 5 cols and coerce types."""
    print(f"  Reading file: {filepath}")
    try:
        df = pd.read_csv(filepath, sep=None, engine='python', header=0)
    except Exception:
        df = pd.read_csv(filepath, sep=None, engine='python', header=None)
    # keep only first five columns
    df = df.iloc[:, :5].copy()
    # assign column names
    df.columns = ['time','open','high','low','close']
    # parse & clean
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df[['open','high','low','close']] = df[['open','high','low','close']].apply(pd.to_numeric, errors='coerce')
    # drop invalid rows
    df.dropna(subset=['time','open','high','low','close'], inplace=True)
    df = df.sort_values('time').reset_index(drop=True)
    print(f"    -> Loaded {len(df)} valid OHLC rows")
    return df


# --- MAIN PROCESSING ---
def create_silver_data(bronze_file, raw_file, silver_file, htf_file=None):
    print(f"\n=== Processing {os.path.basename(bronze_file)} ===")

    # load bronze fully (we need entry_time and other trade columns)
    print("Loading bronze data...")
    bronze_df = pd.read_csv(bronze_file)
    if 'entry_time' not in bronze_df.columns:
        raise ValueError("bronze file missing 'entry_time' column")

    bronze_df['entry_time'] = pd.to_datetime(bronze_df['entry_time'], errors='coerce')
    bronze_df.dropna(subset=['entry_time'], inplace=True)
    bronze_df = bronze_df.sort_values('entry_time').reset_index(drop=True)
    print(f"  Bronze rows: {len(bronze_df)}")

    # load raw ohlcv robustly
    print("Loading raw OHLCV...")
    df = robust_read_ohlcv(raw_file)
    if len(df) == 0:
        print("  ❌ No valid OHLCV rows found in raw file — skipping.")
        return

    # drop first REMOVE_INITIAL_ROWS rows to avoid indicator warmup NaNs
    if REMOVE_INITIAL_ROWS > 0:
        if len(df) <= REMOVE_INITIAL_ROWS:
            print(f"  ❌ Not enough rows ({len(df)}) to remove initial {REMOVE_INITIAL_ROWS} warmup rows. Skipping.")
            return
        print(f"Removing first {REMOVE_INITIAL_ROWS} rows for indicator warmup...")
        df = df.iloc[REMOVE_INITIAL_ROWS:].reset_index(drop=True)
        print(f"  {len(df)} rows remain after trimming.")

    # HTF features (optional)
    if htf_file:
        df = add_htf_features(df, htf_file)

    # feature calculations
    df = add_indicators(df)
    df = add_candlestick_patterns(df)
    df = add_support_resistance(df)
    df = add_sessions(df)
    df = add_past_features(df)
    df = add_regimes(df)

    # downcast before merge to reduce memory footprint
    print("Downcasting dtypes to save memory...")
    df = downcast_dtypes(df)

    # merge_asof: find latest candle at or before each entry_time
    print("Merging bronze trades with OHLC features using merge_asof (nearest previous candle)...")

    # ensure sorted
    df = df.sort_values('time').reset_index(drop=True)
    bronze_df = bronze_df.sort_values('entry_time').reset_index(drop=True)

    # Prepare output
    os.makedirs(os.path.dirname(silver_file), exist_ok=True)
    if os.path.exists(silver_file):
        os.remove(silver_file)

    CHUNK_ROWS = 200_000  # smaller chunk size for safety
    if len(bronze_df) > CHUNK_ROWS:
        print(f"  Bronze has {len(bronze_df)} rows; merging in chunks of {CHUNK_ROWS} rows to save memory.")
        for start in range(0, len(bronze_df), CHUNK_ROWS):
            end = start + CHUNK_ROWS
            chunk = bronze_df.iloc[start:end].copy()
            merged = pd.merge_asof(chunk, df, left_on='entry_time', right_on='time', direction='backward')
            merged.drop(columns=['time'], inplace=True, errors='ignore')

            # Write directly to disk (append mode)
            write_mode = 'w' if start == 0 else 'a'
            header = (start == 0)
            merged.to_csv(silver_file, mode=write_mode, header=header, index=False)

            print(f"    ✅ Merged chunk {start}:{min(end, len(bronze_df))} -> written to disk ({len(merged)} rows)")
            del chunk, merged
            gc.collect()
    else:
        print("  Bronze is small enough, merging in one go.")
        silver_df = pd.merge_asof(bronze_df, df, left_on='entry_time', right_on='time', direction='backward')
        silver_df.drop(columns=['time'], inplace=True, errors='ignore')
        silver_df.to_csv(silver_file, index=False)

    print(f"\n✅ Silver dataset written successfully to {silver_file}")
    print("   (No dropna; ML pipeline should handle NaNs.)")

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
        # Optionally look for HTF file with same symbol but '60' or '240' in filename
        # Example: "EURUSD1.csv" -> check for "EURUSD60.csv" and "EURUSD240.csv"
        basename = os.path.splitext(fname)[0]
        candidate_htf = None
        for suffix in ['60','240','120']:  # try common higher TFs
            htf_candidate = f"{basename}{suffix}.csv"
            htf_path = os.path.join(raw_dir, htf_candidate)
            if os.path.exists(htf_path):
                candidate_htf = htf_path
                print(f"Will use HTF file for {fname}: {htf_candidate}")
                break
        create_silver_data(bronze_path, raw_path, silver_path, htf_file=candidate_htf)

    print("✅ Silver data generation complete!")

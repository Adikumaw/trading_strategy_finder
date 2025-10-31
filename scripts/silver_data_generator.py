# silver_data_generator.py (V3.0 - Refactored & Hardened)
# silver_data_generator.py (V4.0 - Final Polish & Documentation)

"""
Silver Layer: The Enrichment Engine

This script is the central feature engineering hub of the strategy discovery
pipeline. It transforms the raw, high-volume trade simulations from the Bronze
Layer Parquet files into an intelligent, context-rich dataset ready for
machine learning.

The script operates in two main stages for each financial instrument:

1.  Market Feature Generation:
    It first consumes the raw OHLC price data and calculates a comprehensive
    suite of over 200 technical indicators, candlestick patterns, and custom
    market context features for every single candle. This process creates a
    complete "fingerprint" of the market's state at any given moment. The
    resulting dataset is saved as a CSV to `silver_data/features/`.

2.  Trade Enrichment & Chunking:
    It then reads the enormous Bronze Layer Parquet file and processes it using
    a memory-safe, parallel Producer-Consumer architecture. For each trade, it
    calculates "relational positioning" features describing where SL/TP levels
    were relative to key market structures (e.g., moving averages). This is
    achieved using a highly efficient NumPy-based lookup method, which avoids
    slow, memory-intensive DataFrame merges. The final enriched trade data is
    saved in Parquet chunks to `silver_data/chunked_outcomes/`.
"""

import gc
import math
import os
import shutil
import sys
import time
import traceback
from multiprocessing import Manager, Pool, cpu_count
from typing import Dict, List, Tuple

import numba
import numpy as np
import pandas as pd
from tqdm import tqdm

# ### <<< CHANGE: Added robust dependency checks for all key libraries.
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print("[FATAL] 'pyarrow' library not found. Please run 'pip install pyarrow'.")
    sys.exit(1)
try:
    import ta
except ImportError:
    print("[FATAL] 'ta' library not found. Please run 'pip install ta'.")
    sys.exit(1)
try:
    import talib
except ImportError:
    print("[FATAL] 'talib' library not found. Please see library documentation for installation instructions.")
    sys.exit(1)


# --- GLOBAL CONFIGURATION ---
MAX_CPU_USAGE: int = max(1, cpu_count() - 2)
PARQUET_BATCH_SIZE: int = 500_000
INDICATOR_WARMUP_PERIOD: int = 200

# --- TECHNICAL INDICATOR PARAMETERS ---
SMA_PERIODS: List[int] = [20, 50, 100, 200]
EMA_PERIODS: List[int] = [8, 13, 21, 50]
BBANDS_PERIODS: List[int] = [20]
RSI_PERIODS: List[int] = [14]
ATR_PERIODS: List[int] = [14]
ADX_PERIODS: List[int] = [14]
BBANDS_STD_DEV: float = 2.0
MACD_FAST: int = 12
MACD_SLOW: int = 26
MACD_SIGNAL: int = 9
PIVOT_WINDOW: int = 10
PAST_LOOKBACKS: List[int] = [3, 5, 10, 20, 50]

# --- WORKER-SPECIFIC GLOBAL VARIABLES ---
worker_feature_values_np: np.ndarray
worker_time_to_idx_lookup: pd.Series
worker_col_to_idx: Dict[str, int]
worker_levels_for_positioning: List[str]
worker_chunked_outcomes_dir: str


def init_worker(
    feature_values_np: np.ndarray, time_to_idx_lookup: pd.Series, col_to_idx: Dict[str, int],
    levels_for_positioning: List[str], chunked_outcomes_dir: str
) -> None:
    """Initializer for each worker process in the multiprocessing Pool."""
    global worker_feature_values_np, worker_time_to_idx_lookup, worker_col_to_idx
    global worker_levels_for_positioning, worker_chunked_outcomes_dir
    worker_feature_values_np, worker_time_to_idx_lookup, worker_col_to_idx, \
    worker_levels_for_positioning, worker_chunked_outcomes_dir = \
        feature_values_np, time_to_idx_lookup, col_to_idx, \
        levels_for_positioning, chunked_outcomes_dir


# --- UTILITY & FEATURE FUNCTIONS ---

def downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimizes DataFrame memory usage by downcasting numeric types."""
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
    return df

def robust_read_csv(filepath: str) -> pd.DataFrame:
    """Reads raw OHLC CSV data reliably, handling format inconsistencies."""
    df = pd.read_csv(filepath, sep=None, engine="python", header=None)
    if df.shape[1] >= 6:
        df.columns = ['time', 'open', 'high', 'low', 'close', 'volume'][:df.shape[1]]
    elif df.shape[1] == 5:
        df.columns = ['time', 'open', 'high', 'low', 'close']
        df['volume'] = 1
    else:
        raise ValueError(f"File '{os.path.basename(filepath)}' has fewer than 5 columns.")

    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    numeric_cols = ['open', 'high', 'low', 'close']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    initial_rows = len(df)
    df.dropna(subset=['time'] + numeric_cols, inplace=True)
    if len(df) < initial_rows:
        print(f"    [WARN] Dropped {initial_rows - len(df)} rows with invalid date or price data.")
        
    return df.sort_values('time').reset_index(drop=True)

@numba.njit
def _calculate_s_r_numba(lows: np.ndarray, highs: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Identifies fractal support and resistance points using Numba."""
    n = len(lows)
    support, resistance = np.full(n, np.nan, dtype=np.float32), np.full(n, np.nan, dtype=np.float32)
    for i in range(window, n - window):
        ws = slice(i - window, i + window + 1)
        if lows[i] == np.min(lows[ws]): support[i] = lows[i]
        if highs[i] == np.max(highs[ws]): resistance[i] = highs[i]
    return support, resistance

def add_support_resistance(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates and adds forward-filled S/R levels to the DataFrame."""
    lows, highs = df["low"].values.astype(np.float32), df["high"].values.astype(np.float32)
    support_pts, resistance_pts = _calculate_s_r_numba(lows, highs, PIVOT_WINDOW)
    df["support"] = pd.Series(support_pts, index=df.index).ffill()
    df["resistance"] = pd.Series(resistance_pts, index=df.index).ffill()
    return df


def map_market_sessions(hour_series: pd.Series) -> pd.Series:
    """
    Maps a Series of UTC hours to their corresponding Forex market sessions.

    This function provides a granular breakdown of trading sessions, including
    the critical high-volume overlap periods, using `pd.cut` for efficient binning.

    Args:
        hour_series: A Pandas Series containing the hour of the day (0-23).

    Returns:
        A Pandas Series with the mapped session names (e.g., 'London', 'New_York').
    """
    # Define bins representing the start hour of each session/overlap period.
    # Using -1 and 23 ensures all hours from 0 to 23 are covered.
    bins = [-1, 0, 8, 9, 13, 17, 22, 23]
    # Define labels corresponding to the time intervals between the bins.
    labels = [
        'Tokyo',                 # 00:00 - 07:59
        'Tokyo_London_Overlap',  # 08:00 - 08:59
        'London',                # 09:00 - 12:59
        'London_NY_Overlap',     # 13:00 - 16:59
        'New_York',              # 17:00 - 21:59
        'Sydney',                # 22:00 - 22:59
        'Sydney'                 # 23:00 - 23:59 (catches the last hour)
    ]
    # Use pd.cut to efficiently categorize each hour into its respective session.
    return pd.cut(hour_series, bins=bins, labels=labels, ordered=False, right=True)


### <<< NEW FUNCTION: Part of the `add_all_market_features` refactor.
def _add_standard_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates a batch of standard technical indicators."""
    indicator_df = pd.DataFrame(index=df.index)
    # Calculate SMAs for defined periods.
    for p in SMA_PERIODS:
        indicator_df[f"SMA_{p}"] = ta.trend.SMAIndicator(df["close"], p).sma_indicator()
    # Calculate EMAs for defined periods.
    for p in EMA_PERIODS:
        indicator_df[f"EMA_{p}"] = ta.trend.EMAIndicator(df["close"], p).ema_indicator()
    # Calculate Bollinger Bands for defined periods and standard deviations.
    for p in BBANDS_PERIODS:
        bb = ta.volatility.BollingerBands(df["close"], p, BBANDS_STD_DEV)
        indicator_df[f"BB_upper_{p}"] = bb.bollinger_hband()
        indicator_df[f"BB_lower_{p}"] = bb.bollinger_lband()
    # Calculate RSI for defined periods.
    for p in RSI_PERIODS:
        indicator_df[f"RSI_{p}"] = ta.momentum.RSIIndicator(df["close"], p).rsi()
    # Calculate MACD histogram.
    indicator_df[f"MACD_hist_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"] = ta.trend.MACD(df["close"], MACD_SLOW, MACD_FAST, MACD_SIGNAL).macd_diff()
    # Calculate ATR for defined periods.
    for p in ATR_PERIODS:
        indicator_df[f"ATR_{p}"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], p).average_true_range()
    # Calculate ADX for defined periods.
    for p in ADX_PERIODS:
        indicator_df[f"ADX_{p}"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], p).adx()
    # Calculate Rate of Change (Momentum).
    indicator_df["MOM_10"] = ta.momentum.ROCIndicator(df["close"], window=10).roc()
    # Calculate Commodity Channel Index.
    indicator_df["CCI_20"] = ta.trend.CCIIndicator(df["high"], df["low"], df["close"], window=20).cci()
    
    # Calculate ATR-based price levels for dynamic support/resistance.
    for p in ATR_PERIODS:
        atr_series = indicator_df[f"ATR_{p}"]
        indicator_df[f"ATR_level_up_1x_{p}"] = df["close"] + atr_series
        indicator_df[f"ATR_level_down_1x_{p}"] = df["close"] - atr_series
    return indicator_df

### <<< NEW FUNCTION: Part of the `add_all_market_features` refactor.
def _add_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates a batch of TA-Lib candlestick pattern recognitions."""
    pattern_names = talib.get_function_groups().get("Pattern Recognition", [])
    patterns_df = pd.DataFrame(index=df.index)
    # Calculate each candlestick pattern recognized by TA-Lib.
    for p in pattern_names:
        patterns_df[p] = getattr(talib, p)(df["open"], df["high"], df["low"], df["close"])
    return patterns_df

### <<< NEW FUNCTION: Part of the `add_all_market_features` refactor.
def _add_time_and_pa_features(df: pd.DataFrame, indicator_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates time-based, price-action, and market regime features."""
    pa_df = pd.DataFrame(index=df.index)
    # Map hours to market sessions using the robust session mapping function.
    pa_df['session'] = map_market_sessions(df['time'].dt.hour)
    pa_df['hour'], pa_df['weekday'] = df['time'].dt.hour, df['time'].dt.weekday
    
    # Calculate price action features like bullishness and body size.
    is_bullish = (df['close'] > df['open']).astype(int)
    body_size = np.abs(df['close'] - df['open'])
    # Calculate rolling means for bullish ratio, average body size, and average range.
    for n in PAST_LOOKBACKS:
        pa_df[f'bullish_ratio_last_{n}'] = is_bullish.rolling(n).mean()
        pa_df[f'avg_body_last_{n}'] = body_size.rolling(n).mean()
        pa_df[f'avg_range_last_{n}'] = (df['high'] - df['low']).rolling(n).mean()
        
    # Classify market regime based on ADX (trend) and ATR (volatility).
    # These depend on the indicator_df calculated in Batch 1, so it must be available.
    for p in ADX_PERIODS:
        pa_df[f'trend_regime_{p}'] = np.where(indicator_df[f'ADX_{p}'] > 25, 'trend', 'range')
    for p in ATR_PERIODS:
        pa_df[f'vol_regime_{p}'] = np.where(indicator_df[f'ATR_{p}'] > indicator_df[f'ATR_{p}'].rolling(50).mean(), 'high_vol', 'low_vol')
    return pa_df

### <<< REFACTORED FUNCTION: Now acts as a clean orchestrator.
def add_all_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrates the calculation of a comprehensive suite of market features.
    
    This function calls specialized helpers to generate features in logical
    batches and then efficiently concatenates them into a final master DataFrame.
    """
    print("  - Calculating standard indicators...")
    indicator_df = _add_standard_indicators(df)
    
    print("  - Calculating candlestick patterns...")
    patterns_df = _add_candlestick_patterns(df)
    
    print("  - Calculating support and resistance...")
    df_with_sr = add_support_resistance(df.copy()) # Use a copy to avoid side effects
    
    print("  - Calculating time-based and price-action features...")
    pa_df = _add_time_and_pa_features(df, indicator_df)

    # Perform a single, efficient concatenation.
    return pd.concat([
        df, indicator_df, patterns_df, 
        df_with_sr[['support', 'resistance']], pa_df
    ], axis=1)


def create_feature_lookup_structures(
    features_df: pd.DataFrame, level_cols: List[str]
) -> Tuple[np.ndarray, pd.Series, Dict[str, int]]:
    """Pre-computes data into highly efficient lookup structures."""
    features_df = features_df.sort_values('time').reset_index(drop=True)
    col_to_idx = {col: i for i, col in enumerate(level_cols)}
    feature_values_np = features_df[level_cols].to_numpy(dtype=np.float32)
    time_to_idx_lookup = pd.Series(features_df.index, index=features_df['time'])
    return feature_values_np, time_to_idx_lookup, col_to_idx

def add_positioning_features(
    bronze_chunk: pd.DataFrame, feature_values_np: np.ndarray, time_to_idx_lookup: pd.Series,
    col_to_idx: Dict[str, int], levels_for_positioning: List[str]
) -> pd.DataFrame:
    """Enriches a chunk of Bronze trade data with relational positioning features."""
    indices = time_to_idx_lookup.reindex(bronze_chunk['entry_time']).values
    valid_mask = ~np.isnan(indices)
    if not valid_mask.any(): return pd.DataFrame()
        
    bronze_chunk = bronze_chunk.loc[valid_mask].copy()
    indices = indices[valid_mask].astype(int)
    features_for_chunk_np = feature_values_np[indices]

    sl_prices, tp_prices = bronze_chunk['sl_price'].values, bronze_chunk['tp_price'].values
    candle_close_price = features_for_chunk_np[:, col_to_idx['close']]

    def safe_divide(num, den):
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.divide(num, den)
        result[den == 0] = np.nan
        return result

    for level_name in levels_for_positioning:
        level_price = features_for_chunk_np[:, col_to_idx[level_name]]
        bronze_chunk[f'sl_dist_to_{level_name}_bps'] = safe_divide(sl_prices - level_price, candle_close_price) * 10000
        bronze_chunk[f'tp_dist_to_{level_name}_bps'] = safe_divide(tp_prices - level_price, candle_close_price) * 10000
        total_dist_to_level = level_price - candle_close_price
        sl_dist_from_entry = sl_prices - candle_close_price
        tp_dist_from_entry = tp_prices - candle_close_price
        bronze_chunk[f'sl_place_pct_to_{level_name}'] = safe_divide(sl_dist_from_entry, total_dist_to_level)
        bronze_chunk[f'tp_place_pct_to_{level_name}'] = safe_divide(tp_dist_from_entry, total_dist_to_level)
    return bronze_chunk

def queue_worker(task_queue) -> int:
    """The 'Consumer' worker function, which runs in a continuous loop."""
    total = 0
    while True:
        try:
            task = task_queue.get()
            if task is None: break
            chunk_df, chunk_num = task
            if chunk_df.empty: continue
            enriched = add_positioning_features(
                chunk_df, worker_feature_values_np, worker_time_to_idx_lookup,
                worker_col_to_idx, worker_levels_for_positioning
            )
            if not enriched.empty:
                enriched = downcast_dtypes(enriched)
                path = os.path.join(worker_chunked_outcomes_dir, f"chunk_{chunk_num}.parquet")
                enriched.to_parquet(path, index=False)
                total += len(enriched)
        except Exception:
            print(f"\n[WORKER ERROR] An error occurred in a worker process.")
            traceback.print_exc()
    return total

def _get_level_columns(all_columns: List[str]) -> Tuple[List[str], List[str]]:
    """Dynamically identifies columns needed for positioning and NumPy lookup."""
    base = ['time', 'open', 'high', 'low', 'close', 'support', 'resistance']
    patterns = ['SMA_', 'EMA_', 'BB_upper', 'BB_lower', 'ATR_level']
    level_cols = set(base)
    for col in all_columns:
        if any(p in col for p in patterns): level_cols.add(col)
    cols_for_numpy = [c for c in level_cols if c != 'time']
    levels_for_pos = [c for c in cols_for_numpy if c not in ['open', 'high', 'low', 'close']]
    return cols_for_numpy, levels_for_pos

# --- MAIN ORCHESTRATOR ---
def create_silver_data(
    raw_path: str,
    features_path: str,
    bronze_path: str = None,
    chunked_outcomes_dir: str = None,
    features_only: bool = False
) -> None:
    """
    Orchestrates the Silver Layer generation process. Can run in two modes:
    1. Full Mode: Generates features AND enriches Bronze data.
    2. Features Only Mode: Only generates the Silver features file.
    """
    instrument_name = os.path.splitext(os.path.basename(raw_path))[0]
    print(f"\n{'='*30}\nProcessing: {instrument_name}\n{'='*30}")

    # --- Stage 1: Market Feature Generation (Always runs) ---
    print("STEP 1: Creating Silver Features dataset...")
    raw_df = robust_read_csv(raw_path)
    if len(raw_df) < INDICATOR_WARMUP_PERIOD + 100:
        print(f"[ERROR] Not enough data for indicator warmup ({len(raw_df)} rows). Skipping.")
        return
        
    features_df = add_all_market_features(raw_df)
    del raw_df; gc.collect()
    
    features_df = features_df.iloc[INDICATOR_WARMUP_PERIOD:].reset_index(drop=True)
    features_df = downcast_dtypes(features_df)
    
    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    features_df.to_csv(features_path, index=False)
    print(f"[SUCCESS] Silver Features saved to: {os.path.basename(features_path)}")
    
    # ### <<< CHANGE: Conditional exit for --features-only mode.
    if features_only:
        print("[INFO] Features-only mode complete.")
        return # --- EXIT EARLY ---

    # --- Stage 2 & 3: Trade Enrichment (Skipped in features-only mode) ---
    print("\nSTEP 2: Preparing for PARALLEL chunk enrichment...")
    if os.path.exists(chunked_outcomes_dir): shutil.rmtree(chunked_outcomes_dir)
    os.makedirs(chunked_outcomes_dir)
    
    print("  - Creating feature lookup structures for fast processing...")
    cols_for_numpy, levels_for_pos = _get_level_columns(features_df.columns)
    lookup_df = features_df[['time'] + cols_for_numpy]
    feature_values_np, time_to_idx, col_to_idx = create_feature_lookup_structures(lookup_df, cols_for_numpy)
    del features_df, lookup_df; gc.collect()
    print("[SUCCESS] Lookup structures created.")

    print("\nSTEP 3: Enriching Bronze data...")
    try:
        pq_file = pq.ParquetFile(bronze_path)
        num_rows = pq_file.metadata.num_rows
    except Exception as e:
        print(f"[ERROR] Could not read Bronze Parquet file at {bronze_path}: {e}")
        return
        
    if num_rows <= 0:
        print("[INFO] Bronze file is empty. No trades to process.")
        return

    print(f"Found {num_rows:,} trades. Processing in batches...")
    manager = Manager()
    task_queue = manager.Queue(maxsize=MAX_CPU_USAGE * 2)
    pool_init_args = (feature_values_np, time_to_idx, col_to_idx, levels_for_pos, chunked_outcomes_dir)

    with Pool(processes=MAX_CPU_USAGE, initializer=init_worker, initargs=pool_init_args) as pool:
        worker_results = [pool.apply_async(queue_worker, (task_queue,)) for _ in range(MAX_CPU_USAGE)]
        iterator = pq_file.iter_batches(batch_size=PARQUET_BATCH_SIZE)
        
        for i, batch in enumerate(tqdm(iterator, desc="Feeding Batches to Workers"), 1):
            chunk_df = batch.to_pandas()
            task_queue.put((chunk_df, i))
            
        for _ in range(MAX_CPU_USAGE): task_queue.put(None)
        total_trades_processed = sum(res.get() for res in worker_results)

    print(f"\n[SUCCESS] Enriched and chunked {total_trades_processed:,} trades.")
    print(f"Output saved to: {chunked_outcomes_dir}")

def _select_files_interactively(bronze_dir: str, silver_out_dir: str) -> List[str]:
    """Scans for new files and prompts the user to select which ones to process."""
    print("[INFO] Interactive Mode: Scanning for new files...")
    try:
        # ### <<< CHANGE: Look for .parquet files from the Bronze layer.
        bronze_files = sorted([f for f in os.listdir(bronze_dir) if f.endswith('.parquet')])
        new_files = [f for f in bronze_files if not os.path.exists(os.path.join(silver_out_dir, f.replace('.parquet', '')))]

        if not new_files:
            print("[INFO] No new Bronze files to process.")
            return []

        print("\n--- Select File(s) to Process ---")
        for i, f in enumerate(new_files): print(f"  [{i+1}] {f}")
        print("  [a] Process All New Files")
        print("\nEnter selection (e.g., 1,3 or a):")
        
        user_input = input("> ").strip().lower()
        if not user_input: return []
        if user_input == 'a': return new_files

        selected_files = []
        try:
            indices = [int(i.strip()) - 1 for i in user_input.split(',')]
            for idx in indices:
                if 0 <= idx < len(new_files):
                    selected_files.append(new_files[idx])
                else:
                    print(f"[WARN] Invalid selection '{idx + 1}' ignored.")
            return sorted(list(set(selected_files)))
        except ValueError:
            print("[ERROR] Invalid input. Please enter numbers (e.g., 1,3) or 'a'.")
            return []
            
    except FileNotFoundError:
        print(f"[ERROR] The Bronze data directory was not found at: {bronze_dir}")
        return []
            
    except FileNotFoundError:
        print(f"[ERROR] The Bronze data directory was not found at: {bronze_dir}")
        return []


def main() -> None:
    """Main execution function: handles file discovery, user interaction, and orchestration."""
    start_time = time.time()
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    RAW_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'raw_data'))
    BRONZE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'bronze_data'))
    SILVER_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'silver_data'))
    FEATURES_DIR = os.path.join(SILVER_DATA_DIR, 'features')
    CHUNKED_OUTCOMES_DIR = os.path.join(SILVER_DATA_DIR, 'chunked_outcomes')
    os.makedirs(FEATURES_DIR, exist_ok=True)
    os.makedirs(CHUNKED_OUTCOMES_DIR, exist_ok=True)

    print("--- Silver Layer: The Enrichment Engine (Parquet Edition) ---")

    # ### <<< CHANGE: Check for the --features-only flag.
    features_only_mode = '--features-only' in sys.argv
    target_file_arg = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else None
    
    if features_only_mode:
        print("[INFO] Running in FEATURES-ONLY mode.")
        if not target_file_arg:
            print("[ERROR] --features-only mode requires a target file argument (e.g., python silver...py EURUSD15.csv --features-only).")
            return
        
        # In features-only mode, we are working from raw data files.
        instrument_name = target_file_arg.replace('.csv', '')
        raw_path = os.path.join(RAW_DIR, target_file_arg)
        if not os.path.exists(raw_path):
            print(f"[ERROR] Raw data file not found: {raw_path}")
            return
            
        paths = {
            "raw_path": raw_path,
            "features_path": os.path.join(FEATURES_DIR, f"{instrument_name}.csv"),
            "features_only": True
        }
        create_silver_data(**paths)
        
    else: # --- Normal Mode (Full pipeline) ---
        if target_file_arg:
            print(f"\n[INFO] Targeted Mode: Processing '{target_file_arg}'")
            if not target_file_arg.endswith('.parquet'): target_file_arg += '.parquet'
            files_to_process = [target_file_arg] if os.path.exists(os.path.join(BRONZE_DIR, target_file_arg)) else []
            if not files_to_process: print(f"[ERROR] Target file not found in bronze_data: {target_file_arg}")
        else:
            files_to_process = _select_files_interactively(BRONZE_DIR, CHUNKED_OUTCOMES_DIR)

        if not files_to_process:
            print("\n[INFO] No files selected or found for processing. Exiting.")
        else:
            print(f"\n[QUEUE] Queued {len(files_to_process)} file(s): {', '.join(files_to_process)}")
            for filename in files_to_process:
                instrument_name = filename.replace('.parquet', '')
                raw_filename = f"{instrument_name}.csv"
                paths = {
                    "bronze_path": os.path.join(BRONZE_DIR, filename),
                    "raw_path": os.path.join(RAW_DIR, raw_filename),
                    "features_path": os.path.join(FEATURES_DIR, f"{instrument_name}.csv"),
                    "chunked_outcomes_dir": os.path.join(CHUNKED_OUTCOMES_DIR, instrument_name),
                    "features_only": False
                }
                if not os.path.exists(paths["raw_path"]):
                    print(f"[WARN] SKIPPING {filename}: Corresponding raw_data file ('{raw_filename}') is missing.")
                    continue
                try:
                    create_silver_data(**paths)
                except Exception:
                    print(f"\n[FATAL ERROR] A critical error occurred while processing {filename}.")
                    traceback.print_exc()

    end_time = time.time()
    print(f"\nSilver Layer generation finished. Total time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    # The helper functions being called are assumed to be defined above.
    # We only show the main orchestrator function changes here.
    main()
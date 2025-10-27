# silver_data_generator.py (V2.3 - Final Version)

"""
Silver Layer: The Enrichment Engine

This script is the central feature engineering hub of the strategy discovery
pipeline. It transforms the raw, high-volume trade simulations from the Bronze
Layer into an intelligent, context-rich dataset ready for machine learning.

The script operates in two main stages for each financial instrument:

1.  Market Feature Generation:
    It first consumes the raw OHLC price data and calculates a comprehensive
    suite of over 200 technical indicators, candlestick patterns, and custom
    market context features for every single candle. This process creates a
    complete "fingerprint" of the market's state at any given moment. The
    resulting dataset is saved to `silver_data/features/`.

2.  Trade Enrichment & Chunking:
    It then processes the enormous Bronze Dataset using a memory-safe, parallel
    Producer-Consumer architecture. For each winning trade, it calculates a
    powerful set of "relational positioning" features that describe where a
    trade's SL/TP levels were relative to key market structures (e.g., moving
    averages, Bollinger Bands). This is achieved using a highly efficient
    NumPy-based lookup method, which avoids memory-intensive DataFrame merges.
    The final enriched trade data is saved in chunks to
    `silver_data/chunked_outcomes/`.
"""

import gc
import math
import os
import shutil
import sys
import traceback
from multiprocessing import Manager, Pool, cpu_count
from typing import Dict, List, Tuple

import numba
import numpy as np
import pandas as pd
import ta
import talib
from tqdm import tqdm

# --- CONFIGURATION ---
MAX_CPU_USAGE: int = max(1, cpu_count() - 2)
CHUNK_SIZE: int = 500_000

# Defines the periods for various technical indicators to be calculated.
# Using lists allows for easy experimentation with multiple periods.
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
INDICATOR_WARMUP_PERIOD: int = 200

# --- WORKER-SPECIFIC GLOBAL VARIABLES ---
worker_feature_values_np: np.ndarray
worker_time_to_idx_lookup: pd.Series
worker_col_to_idx: Dict[str, int]
worker_levels_for_positioning: List[str]
worker_chunked_outcomes_dir: str


def init_worker(
    feature_values_np: np.ndarray,
    time_to_idx_lookup: pd.Series,
    col_to_idx: Dict[str, int],
    levels_for_positioning: List[str],
    chunked_outcomes_dir: str
) -> None:
    """Initializer for each worker process in the multiprocessing Pool."""
    global worker_feature_values_np, worker_time_to_idx_lookup
    global worker_col_to_idx, worker_levels_for_positioning
    global worker_chunked_outcomes_dir
    worker_feature_values_np = feature_values_np
    worker_time_to_idx_lookup = time_to_idx_lookup
    worker_col_to_idx = col_to_idx
    worker_levels_for_positioning = levels_for_positioning
    worker_chunked_outcomes_dir = chunked_outcomes_dir


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
        raise ValueError(f"File '{filepath}' has fewer than 5 columns.")
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=['time'] + numeric_cols, inplace=True)
    return df.sort_values('time').reset_index(drop=True)


@numba.njit
def _calculate_s_r_numba(
    lows: np.ndarray, highs: np.ndarray, window: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Identifies fractal support and resistance points using a Numba kernel."""
    n = len(lows)
    support = np.full(n, np.nan, dtype=np.float32)
    resistance = np.full(n, np.nan, dtype=np.float32)
    for i in range(window, n - window):
        window_slice = slice(i - window, i + window + 1)
        if lows[i] == np.min(lows[window_slice]):
            support[i] = lows[i]
        if highs[i] == np.max(highs[window_slice]):
            resistance[i] = highs[i]
    return support, resistance


def add_support_resistance(df: pd.DataFrame, window: int = PIVOT_WINDOW) -> pd.DataFrame:
    """Calculates and adds forward-filled S/R levels to the DataFrame."""
    lows = df["low"].values.astype(np.float32)
    highs = df["high"].values.astype(np.float32)
    support_pts, resistance_pts = _calculate_s_r_numba(lows, highs, window)
    sr_df = pd.DataFrame({
        'support_points': support_pts,
        'resistance_points': resistance_pts
    }, index=df.index)
    df["support"] = sr_df["support_points"].ffill()
    df["resistance"] = sr_df["resistance_points"].ffill()
    return df


def map_market_sessions(hour_series: pd.Series) -> pd.Series:
    """
    Maps a Series of UTC hours to their corresponding Forex market sessions.

    This function provides a granular breakdown of trading sessions, including
    the critical high-volume overlap periods.

    Args:
        hour_series: A Pandas Series containing the hour of the day (0-23).

    Returns:
        A Pandas Series with the mapped session names.
    """
    # Bins represent the start hour of each session.
    # The hour is an integer from 0 to 23.
    bins = [-1, 0, 8, 9, 13, 17, 22, 23]
    # Labels correspond to the time period *between* the bins.
    labels = [
        'Tokyo',          # 00:00 - 07:59
        'Tokyo_London_Overlap', # 08:00 - 08:59
        'London',         # 09:00 - 12:59
        'London_NY_Overlap',# 13:00 - 16:59
        'New_York',       # 17:00 - 21:59
        'Sydney',         # 22:00 - 22:59
        'Sydney'          # 23:00 - 23:59 (catches the last hour)
    ]
    return pd.cut(hour_series, bins=bins, labels=labels, ordered=False, right=True)


def add_all_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a comprehensive suite of market features for the input OHLC data.
    """
    all_feature_dfs = []

    # --- Batch 1: Standard Technical Indicators ---
    print("Calculating standard indicators...")
    indicator_df = pd.DataFrame(index=df.index)
    for p in SMA_PERIODS:
        indicator_df[f"SMA_{p}"] = ta.trend.SMAIndicator(df["close"], p).sma_indicator()
    for p in EMA_PERIODS:
        indicator_df[f"EMA_{p}"] = ta.trend.EMAIndicator(df["close"], p).ema_indicator()
    for p in BBANDS_PERIODS:
        bb = ta.volatility.BollingerBands(df["close"], p, BBANDS_STD_DEV)
        indicator_df[f"BB_upper_{p}"] = bb.bollinger_hband()
        indicator_df[f"BB_lower_{p}"] = bb.bollinger_lband()
    for p in RSI_PERIODS:
        indicator_df[f"RSI_{p}"] = ta.momentum.RSIIndicator(df["close"], p).rsi()
    indicator_df[f"MACD_hist_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"] = ta.trend.MACD(df["close"], MACD_SLOW, MACD_FAST, MACD_SIGNAL).macd_diff()
    indicator_df["MOM_10"] = ta.momentum.ROCIndicator(df["close"], window=10).roc()
    indicator_df["CCI_20"] = ta.trend.CCIIndicator(df["high"], df["low"], df["close"], window=20).cci()
    for p in ATR_PERIODS:
        indicator_df[f"ATR_{p}"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], p).average_true_range()
    for p in ADX_PERIODS:
        indicator_df[f"ADX_{p}"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], p).adx()
    for p in ATR_PERIODS:
        atr_series = indicator_df[f"ATR_{p}"]
        indicator_df[f"ATR_level_up_1x_{p}"] = df["close"] + atr_series
        indicator_df[f"ATR_level_down_1x_{p}"] = df["close"] - atr_series
    all_feature_dfs.append(indicator_df)

    # --- Batch 2: Candlestick Patterns ---
    print("Calculating candlestick patterns...")
    pattern_names = talib.get_function_groups().get("Pattern Recognition", [])
    patterns_df = pd.DataFrame(index=df.index)
    for p in pattern_names:
        patterns_df[p] = getattr(talib, p)(df["open"], df["high"], df["low"], df["close"])
    all_feature_dfs.append(patterns_df)
    
    # --- Batch 3: Support & Resistance ---
    print("Calculating support and resistance...")
    df_with_sr = add_support_resistance(df.copy())
    all_feature_dfs.append(df_with_sr[['support', 'resistance']])

    # --- Batch 4: Time-based and Price-Action Features ---
    print("Calculating time-based and price-action features...")
    pa_df = pd.DataFrame(index=df.index)
    pa_df['session'] = map_market_sessions(df['time'].dt.hour)
    pa_df['hour'], pa_df['weekday'] = df['time'].dt.hour, df['time'].dt.weekday
    is_bullish = (df['close'] > df['open']).astype(int)
    body_size = np.abs(df['close'] - df['open'])
    for n in PAST_LOOKBACKS:
        pa_df[f'bullish_ratio_last_{n}'] = is_bullish.rolling(n).mean()
        pa_df[f'avg_body_last_{n}'] = body_size.rolling(n).mean()
        pa_df[f'avg_range_last_{n}'] = (df['high'] - df['low']).rolling(n).mean()
    for p in ADX_PERIODS:
        pa_df[f'trend_regime_{p}'] = np.where(indicator_df[f'ADX_{p}'] > 25, 'trend', 'range')
    for p in ATR_PERIODS:
        pa_df[f'vol_regime_{p}'] = np.where(indicator_df[f'ATR_{p}'] > indicator_df[f'ATR_{p}'].rolling(50).mean(), 'high_vol', 'low_vol')
    all_feature_dfs.append(pa_df)

    return pd.concat([df] + all_feature_dfs, axis=1)


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
    bronze_chunk: pd.DataFrame,
    feature_values_np: np.ndarray,
    time_to_idx_lookup: pd.Series,
    col_to_idx: Dict[str, int],
    levels_for_positioning: List[str]
) -> pd.DataFrame:
    """Enriches a chunk of Bronze data using fast NumPy lookups."""
    indices = time_to_idx_lookup.reindex(bronze_chunk['entry_time']).values
    valid_mask = ~np.isnan(indices)
    if not valid_mask.any():
        return pd.DataFrame()

    bronze_chunk = bronze_chunk.loc[valid_mask].copy()
    indices = indices[valid_mask].astype(int)
    features_for_chunk_np = feature_values_np[indices]

    sl_prices = bronze_chunk['sl_price'].values
    tp_prices = bronze_chunk['tp_price'].values
    candle_close_price = features_for_chunk_np[:, col_to_idx['close']]

    def safe_divide(num, den):
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.divide(num, den)
        result[den == 0] = np.nan
        return result

    for level_name in levels_for_positioning:
        level_price = features_for_chunk_np[:, col_to_idx[level_name]]
        bronze_chunk[f'sl_dist_{level_name}_bps'] = safe_divide(sl_prices - level_price, candle_close_price) * 10000
        bronze_chunk[f'tp_dist_{level_name}_bps'] = safe_divide(tp_prices - level_price, candle_close_price) * 10000
        total_dist_to_level = level_price - candle_close_price
        sl_dist_from_entry = sl_prices - candle_close_price
        tp_dist_from_entry = tp_prices - candle_close_price
        bronze_chunk[f'sl_place_{level_name}_pct'] = safe_divide(sl_dist_from_entry, total_dist_to_level)
        bronze_chunk[f'tp_place_{level_name}_pct'] = safe_divide(tp_dist_from_entry, total_dist_to_level)

    return bronze_chunk


def queue_worker(task_queue: Manager.Queue) -> int:
    """The 'Consumer' worker function, which runs in a continuous loop."""
    total_processed_in_worker = 0
    while True:
        try:
            task = task_queue.get()
            if task is None:
                break
            chunk_df, chunk_number = task
            if chunk_df.empty:
                continue
            enriched_chunk = add_positioning_features(
                chunk_df, worker_feature_values_np, worker_time_to_idx_lookup,
                worker_col_to_idx, worker_levels_for_positioning
            )
            if not enriched_chunk.empty:
                enriched_chunk = downcast_dtypes(enriched_chunk)
                output_path = os.path.join(worker_chunked_outcomes_dir, f"chunk_{chunk_number}.csv")
                enriched_chunk.to_csv(output_path, index=False)
                total_processed_in_worker += len(enriched_chunk)
        except Exception:
            print(f"WORKER ERROR processing chunk. See traceback below.")
            traceback.print_exc()
    return total_processed_in_worker


def create_silver_data(
    bronze_path: str, raw_path: str, features_path: str, chunked_outcomes_dir: str
) -> None:
    """Orchestrates the entire Silver Layer generation for a single instrument."""
    instrument_name = os.path.basename(raw_path).replace('.csv', '')
    print(f"\n{'='*25}\nProcessing: {instrument_name}\n{'='*25}")

    # --- STEP 1: Create Silver Features ---
    print("STEP 1: Creating Silver Features dataset...")
    raw_df = robust_read_csv(raw_path)
    if len(raw_df) < INDICATOR_WARMUP_PERIOD + 1:
        print(f"[ERROR] SKIPPING: Not enough data for indicator warmup.")
        return

    features_df = add_all_market_features(raw_df)
    del raw_df; gc.collect()

    features_df = features_df.iloc[INDICATOR_WARMUP_PERIOD:].reset_index(drop=True)
    features_df = downcast_dtypes(features_df)
    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    features_df.to_csv(features_path, index=False)
    print(f"[SUCCESS] Silver Features saved to: {features_path}")

    # --- STEP 2: Prepare for Parallel Enrichment ---
    print("\nSTEP 2: Preparing for PARALLEL chunk enrichment...")
    if os.path.exists(chunked_outcomes_dir):
        shutil.rmtree(chunked_outcomes_dir)
    os.makedirs(chunked_outcomes_dir)
    print("Creating feature lookup structures for fast processing...")
    
    feature_cols = features_df.columns
    base_levels = ['time', 'open', 'high', 'low', 'close', 'support', 'resistance']
    indicator_patterns = ['SMA_', 'EMA_', 'BB_upper', 'BB_lower', 'ATR_level']
    level_cols = base_levels + [c for c in feature_cols if any(p in c for p in indicator_patterns)]
    
    lookup_df = features_df[list(set(level_cols))]
    cols_for_numpy = [c for c in lookup_df.columns if c != 'time']
    feature_values_np, time_to_idx, col_to_idx = create_feature_lookup_structures(lookup_df, cols_for_numpy)
    levels_for_positioning = [c for c in cols_for_numpy if c not in ['open', 'high', 'low', 'close']]
    del features_df, lookup_df; gc.collect()
    print("[SUCCESS] Lookup structures created.")

    # --- STEP 3: Execute Parallel Enrichment with Producer-Consumer Queue ---
    try:
        num_rows = sum(1 for _ in open(bronze_path, 'r')) - 1
    except FileNotFoundError:
        print(f"[ERROR] Bronze file not found at {bronze_path}. Cannot proceed.")
        return
    if num_rows <= 0:
        print("[INFO] Bronze file is empty. No trades to process.")
        return

    num_chunks = math.ceil(num_rows / CHUNK_SIZE)
    print(f"Found {num_rows:,} trades, processing in {num_chunks} chunks.")

    manager = Manager()
    task_queue = manager.Queue(maxsize=MAX_CPU_USAGE * 2)
    pool_init_args = (feature_values_np, time_to_idx, col_to_idx, levels_for_positioning, chunked_outcomes_dir)

    with Pool(processes=MAX_CPU_USAGE, initializer=init_worker, initargs=pool_init_args) as pool:
        worker_results = [pool.apply_async(queue_worker, (task_queue,)) for _ in range(MAX_CPU_USAGE)]
        iterator = pd.read_csv(bronze_path, chunksize=CHUNK_SIZE, parse_dates=['entry_time'])
        for i, chunk_df in enumerate(tqdm(iterator, total=num_chunks, desc="Feeding Chunks to Workers"), 1):
            task_queue.put((chunk_df, i))
        for _ in range(MAX_CPU_USAGE):
            task_queue.put(None)
        total_trades_processed = sum(res.get() for res in worker_results)

    print(f"\n[SUCCESS] Enriched and chunked {total_trades_processed:,} trades.")
    print(f"Output saved to: {chunked_outcomes_dir}")


def main() -> None:
    """Main execution function."""
    core_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.abspath(os.path.join(core_dir, '..', 'raw_data'))
    bronze_dir = os.path.abspath(os.path.join(core_dir, '..', 'bronze_data'))
    features_dir = os.path.abspath(os.path.join(core_dir, '..', 'silver_data', 'features'))
    chunked_outcomes_base_dir = os.path.abspath(os.path.join(core_dir, '..', 'silver_data', 'chunked_outcomes'))
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(chunked_outcomes_base_dir, exist_ok=True)

    files_to_process = []
    target_file_arg = sys.argv[1] if len(sys.argv) > 1 else None
    if target_file_arg:
        print(f"[TARGET] Targeted Mode: Processing '{target_file_arg}'")
        if not os.path.exists(os.path.join(bronze_dir, target_file_arg)):
            print(f"[ERROR] Target file not found in bronze_data: {target_file_arg}")
        else:
            files_to_process = [target_file_arg]
    else:
        print("[SCAN] Interactive Mode: Scanning for new files...")
        try:
            bronze_files = sorted([f for f in os.listdir(bronze_dir) if f.endswith('.csv')])
            new_files = [f for f in bronze_files if not os.path.exists(os.path.join(chunked_outcomes_base_dir, f.replace('.csv', '')))]
            if not new_files:
                print("[INFO] No new Bronze files to process.")
            else:
                print("\n--- Select File(s) to Process ---")
                for i, f in enumerate(new_files): print(f"  [{i+1}] {f}")
                print("\nSelect multiple files with comma-separated numbers (e.g., 1,3,5)")
                user_input = input("Enter number(s) to process: ").strip()
                if user_input:
                    try:
                        indices = [int(i.strip()) - 1 for i in user_input.split(',')]
                        files_to_process = [new_files[idx] for idx in sorted(set(indices)) if 0 <= idx < len(new_files)]
                    except ValueError:
                        print("[ERROR] Invalid input. Please enter numbers only.")
        except FileNotFoundError:
            print(f"[ERROR] The directory '{bronze_dir}' was not found.")

    if not files_to_process:
        print("[INFO] No files selected or found for processing.")
        return

    print(f"\n[QUEUE] Queued {len(files_to_process)} file(s): {files_to_process}")
    for filename in files_to_process:
        instrument_name = filename.replace('.csv', '')
        paths = {
            "bronze_path": os.path.join(bronze_dir, filename),
            "raw_path": os.path.join(raw_dir, filename),
            "features_path": os.path.join(features_dir, filename),
            "chunked_outcomes_dir": os.path.join(chunked_outcomes_base_dir, instrument_name)
        }
        if not all(os.path.exists(paths[k]) for k in ["bronze_path", "raw_path"]):
            print(f"[WARNING] SKIPPING {filename}: Missing raw or bronze file.")
            continue
        try:
            create_silver_data(**paths)
        except Exception:
            print(f"\n[FATAL ERROR] A critical error occurred while processing {filename}.")
            traceback.print_exc()

    print("\n" + "="*50 + "\n[COMPLETE] All Silver Layer data generation tasks are finished.")


if __name__ == "__main__":
    main()
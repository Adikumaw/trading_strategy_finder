# silver_data_generator.py (V2.4 - Final Re-Documentation and Formatting)

"""
Silver Layer: The Enrichment Engine

This script is the central feature engineering hub of the strategy discovery
pipeline. It transforms the raw, high-volume trade simulations from the Bronze
Layer into an intelligent, context-rich dataset ready for machine learning and
pattern discovery.

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
import re
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

# --- GLOBAL CONFIGURATION ---
# Maximum number of CPU cores to use for multiprocessing, leaving 2 free for responsiveness.
MAX_CPU_USAGE: int = max(1, cpu_count() - 2)
# Size of chunks for reading the large Bronze CSV file.
CHUNK_SIZE: int = 500_000
# Number of initial candles to discard to allow indicators to generate stable values.
INDICATOR_WARMUP_PERIOD: int = 200

# --- TECHNICAL INDICATOR PERIODS ---
# Using lists allows for easy experimentation with multiple periods for indicators.
SMA_PERIODS: List[int] = [20, 50, 100, 200]
EMA_PERIODS: List[int] = [8, 13, 21, 50]
BBANDS_PERIODS: List[int] = [20]
RSI_PERIODS: List[int] = [14]
ATR_PERIODS: List[int] = [14]
ADX_PERIODS: List[int] = [14]

# --- TECHNICAL INDICATOR PARAMETERS ---
BBANDS_STD_DEV: float = 2.0
MACD_FAST: int = 12
MACD_SLOW: int = 26
MACD_SIGNAL: int = 9
PIVOT_WINDOW: int = 10  # Window for fractal-based Support/Resistance identification.
PAST_LOOKBACKS: List[int] = [3, 5, 10, 20, 50] # Lookback periods for price-action features.

# --- WORKER-SPECIFIC GLOBAL VARIABLES ---
# These variables are populated in each worker process upon initialization to avoid
# repeatedly passing large data objects, which is crucial for performance and stability.
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
    """
    Initializer for each worker process in the multiprocessing Pool.

    This function is executed once per worker when the pool is spawned. It receives
    large, read-only data objects and stores them in global variables within that
    worker's memory space. This pattern significantly boosts performance by
    preventing the serialization and transfer of large data with every task.

    Args:
        feature_values_np: NumPy array containing all numeric feature values.
        time_to_idx_lookup: Pandas Series mapping timestamps to row indices.
        col_to_idx: Dictionary mapping column names to their index in feature_values_np.
        levels_for_positioning: List of feature column names relevant for positioning calculations.
        chunked_outcomes_dir: The output directory for the worker's processed chunks.
    """
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
    """
    Optimizes DataFrame memory usage by downcasting numeric types to smaller dtypes.

    Iterates through float64 and int64 columns, converting them to float32
    and int32 respectively. This can significantly reduce memory footprint
    with negligible loss of precision for most financial data.

    Args:
        df: The input DataFrame to optimize.

    Returns:
        The DataFrame with downcasted numeric data types.
    """
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
    return df


def robust_read_csv(filepath: str) -> pd.DataFrame:
    """
    Reads raw OHLC CSV data reliably, handling common format inconsistencies.

    This function is resilient to different delimiters by using `sep=None` and
    the 'python' engine. It assigns standard column names, adds a dummy volume
    column if missing, converts columns to proper types, and drops rows with
    missing essential data. It also ensures the DataFrame is sorted by time.

    Args:
        filepath: The full path to the raw input CSV file.

    Returns:
        A clean, sorted DataFrame with standardized columns ('time', 'open',
        'high', 'low', 'close', 'volume').
    """
    df = pd.read_csv(filepath, sep=None, engine="python", header=None)
    # Assign column names based on the number of columns found.
    if df.shape[1] >= 6:
        df.columns = ['time', 'open', 'high', 'low', 'close', 'volume'][:df.shape[1]]
    elif df.shape[1] == 5:
        df.columns = ['time', 'open', 'high', 'low', 'close']
        df['volume'] = 1  # Add a dummy volume column if not present.
    else:
        raise ValueError(f"File '{filepath}' has fewer than 5 columns.")

    # Convert columns to appropriate data types, coercing errors to NaN.
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Remove rows with missing critical data.
    df.dropna(subset=['time'] + numeric_cols, inplace=True)
    # Ensure data is sorted by time, which is critical for all subsequent calculations.
    return df.sort_values('time').reset_index(drop=True)


@numba.njit
def _calculate_s_r_numba(
    lows: np.ndarray, highs: np.ndarray, window: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identifies fractal support and resistance points using a high-speed Numba kernel.

    This function iterates through price data and identifies local minima and maxima
    within a specified rolling window. A low is marked as a support point if it is
    the lowest value in the surrounding window. A high is marked as a resistance
    point if it is the highest value. This "fractal" approach is a common way to
    identify significant price levels.

    Args:
        lows (np.array): A NumPy array of low prices.
        highs (np.array): A NumPy array of high prices.
        window (int): The number of candles to look forward and backward to define the local window.

    Returns:
        tuple: A tuple containing two NumPy arrays:
               - support_points: An array with prices at support fractals, and NaN elsewhere.
               - resistance_points: An array with prices at resistance fractals, and NaN elsewhere.
    """
    n = len(lows)
    support = np.full(n, np.nan, dtype=np.float32)
    resistance = np.full(n, np.nan, dtype=np.float32)
    # Iterate through the data, leaving a buffer for the window size at the start and end.
    for i in range(window, n - window):
        window_slice = slice(i - window, i + window + 1)
        # Check if the current low is the minimum within the defined window.
        if lows[i] == np.min(lows[window_slice]):
            support[i] = lows[i]
        # Check if the current high is the maximum within the defined window.
        if highs[i] == np.max(highs[window_slice]):
            resistance[i] = highs[i]
    return support, resistance


def add_support_resistance(df: pd.DataFrame, window: int = PIVOT_WINDOW) -> pd.DataFrame:
    """
    Calculates and adds forward-filled support and resistance levels to the DataFrame.

    This function wraps the high-speed Numba S/R calculation and then
    forward-fills the resulting sparse support and resistance points. Forward-filling
    ensures that every candle has a value for the "last known" support and
    resistance level, which is crucial for subsequent analysis.

    Args:
        df: The input market data DataFrame.
        window: The window size for the fractal calculation.

    Returns:
        The DataFrame with new 'support' and 'resistance' columns added.
    """
    lows = df["low"].values.astype(np.float32)
    highs = df["high"].values.astype(np.float32)
    # Calculate raw support and resistance points using the Numba function.
    support_pts, resistance_pts = _calculate_s_r_numba(lows, highs, window)
    
    # Create a temporary DataFrame to hold the calculated points.
    sr_df = pd.DataFrame({
        'support_points': support_pts,
        'resistance_points': resistance_pts
    }, index=df.index)
    
    # Forward-fill the support and resistance values to carry them forward in time.
    # Assigning directly to the original df is efficient as it avoids creating unnecessary copies.
    df["support"] = sr_df["support_points"].ffill()
    df["resistance"] = sr_df["resistance_points"].ffill()
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


def add_all_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a comprehensive suite of market features for the input OHLC data.

    This function calculates and aggregates various types of features in batches:
    1. Standard Technical Indicators (SMAs, EMAs, BBands, RSI, MACD, ATR, ADX, etc.).
    2. Candlestick Patterns using the TA-Lib library.
    3. Fractal-based Support and Resistance levels.
    4. Time-based features (session, hour, weekday) and Price-Action characteristics.

    It uses an efficient pattern of creating separate DataFrames for each batch
    and then performing a single, optimized `pd.concat` operation at the end.

    Args:
        df: The raw OHLC DataFrame with a 'time' column.

    Returns:
        The original DataFrame enriched with all calculated market features.
    """
    all_feature_dfs = [] # List to hold all generated feature DataFrames.

    # --- Batch 1: Standard Technical Indicators ---
    print("Calculating standard indicators...")
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
    all_feature_dfs.append(indicator_df) # Add all calculated indicators to the list.

    # --- Batch 2: Candlestick Patterns ---
    print("Calculating candlestick patterns...")
    pattern_names = talib.get_function_groups().get("Pattern Recognition", [])
    patterns_df = pd.DataFrame(index=df.index)
    # Calculate each candlestick pattern recognized by TA-Lib.
    for p in pattern_names:
        patterns_df[p] = getattr(talib, p)(df["open"], df["high"], df["low"], df["close"])
    all_feature_dfs.append(patterns_df) # Add candlestick patterns to the list.
    
    # --- Batch 3: Support & Resistance ---
    print("Calculating support and resistance...")
    # Calculate S/R levels and add them as new columns to a copy of the DataFrame.
    df_with_sr = add_support_resistance(df.copy())
    # Add only the 'support' and 'resistance' columns to the list of feature DataFrames.
    all_feature_dfs.append(df_with_sr[['support', 'resistance']])

    # --- Batch 4: Time-based and Price-Action Features ---
    print("Calculating time-based and price-action features...")
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
    all_feature_dfs.append(pa_df) # Add all price-action and time-based features to the list.

    # Perform a single, efficient concatenation of the original DataFrame with all feature sets.
    return pd.concat([df] + all_feature_dfs, axis=1)


def create_feature_lookup_structures(
    features_df: pd.DataFrame, level_cols: List[str]
) -> Tuple[np.ndarray, pd.Series, Dict[str, int]]:
    """
    Pre-computes data into highly efficient lookup structures for fast retrieval.

    This function transforms the features DataFrame into structures optimized for
    near-instantaneous data lookup, avoiding slow, memory-intensive DataFrame merges.

    Args:
        features_df: The DataFrame containing all market features, sorted by time.
        level_cols: A list of the specific column names required for lookup calculations.

    Returns:
        A tuple containing:
        - feature_values_np (np.array): A 2D NumPy array of the raw numeric feature values.
        - time_to_idx_lookup (pd.Series): A mapping from timestamp to its integer row index.
        - col_to_idx (dict): A mapping from a feature's column name to its integer column index.
    """
    # Ensure the DataFrame is sorted by time, which is critical for correct lookup.
    features_df = features_df.sort_values('time').reset_index(drop=True)
    # Create the mapping from column name to its integer index in the NumPy array.
    col_to_idx = {col: i for i, col in enumerate(level_cols)}
    # Convert the relevant columns to a NumPy array for maximum computational speed.
    feature_values_np = features_df[level_cols].to_numpy(dtype=np.float32)
    # Create the primary time-to-index lookup using a Pandas Series for efficiency.
    time_to_idx_lookup = pd.Series(features_df.index, index=features_df['time'])
    return feature_values_np, time_to_idx_lookup, col_to_idx


def add_positioning_features(
    bronze_chunk: pd.DataFrame,
    feature_values_np: np.ndarray,
    time_to_idx_lookup: pd.Series,
    col_to_idx: Dict[str, int],
    levels_for_positioning: List[str]
) -> pd.DataFrame:
    """
    Enriches a chunk of Bronze trade data with relational positioning features.

    This function utilizes the pre-computed lookup structures to efficiently retrieve
    market features for each trade's entry time. It then calculates features that
    describe the placement of SL/TP levels relative to these market structures,
    discarding trades from the indicator warmup period to ensure data integrity.

    Args:
        bronze_chunk: A chunk of data from the Bronze trades file (trades only).
        feature_values_np: The NumPy array of market feature values.
        time_to_idx_lookup: The timestamp-to-row-index lookup Series.
        col_to_idx: The column-name-to-column-index lookup dictionary.
        levels_for_positioning: List of feature column names to calculate positioning against.

    Returns:
        The enriched DataFrame chunk with new positioning features added.
    """
    # Use .reindex() for a direct lookup. This naturally produces NaN for any entry_time
    # not found in the lookup (i.e., trades from the warmup period).
    indices = time_to_idx_lookup.reindex(bronze_chunk['entry_time']).values
    
    # Create a boolean mask to identify and filter out trades occurring during the warmup period.
    valid_mask = ~np.isnan(indices)
    if not valid_mask.any():
        return pd.DataFrame() # Return an empty DataFrame if no valid trades exist in the chunk.

    # Filter the chunk to keep only trades with valid feature data.
    bronze_chunk = bronze_chunk.loc[valid_mask].copy()
    # Convert valid indices to integers for NumPy array indexing.
    indices = indices[valid_mask].astype(int)
    
    # Efficiently retrieve all required feature rows for the valid trades in one slice.
    features_for_chunk_np = feature_values_np[indices]

    # Extract relevant trade price data into NumPy arrays for vectorized operations.
    sl_prices = bronze_chunk['sl_price'].values
    tp_prices = bronze_chunk['tp_price'].values
    # Get the closing price from the specific candle that triggered the trade.
    candle_close_price = features_for_chunk_np[:, col_to_idx['close']]

    # Helper function for safe division, handling division by zero.
    def safe_divide(num, den):
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.divide(num, den)
        # Set results to NaN where the denominator was zero.
        result[den == 0] = np.nan
        return result

    # Calculate positioning features relative to each specified market level.
    for level_name in levels_for_positioning:
        level_price = features_for_chunk_np[:, col_to_idx[level_name]]
        
        # Feature 1: Distance in Basis Points (bps).
        # Calculates how far SL/TP is from the level, normalized by entry price.
        bronze_chunk[f'sl_dist_{level_name}_bps'] = safe_divide(sl_prices - level_price, candle_close_price) * 10000
        bronze_chunk[f'tp_dist_{level_name}_bps'] = safe_divide(tp_prices - level_price, candle_close_price) * 10000

        # Feature 2: Placement as a Percentage.
        # Calculates where SL/TP is placed on a scale from entry (0%) to the level (100%).
        total_dist_to_level = level_price - candle_close_price
        sl_dist_from_entry = sl_prices - candle_close_price
        tp_dist_from_entry = tp_prices - candle_close_price
        bronze_chunk[f'sl_place_{level_name}_pct'] = safe_divide(sl_dist_from_entry, total_dist_to_level)
        bronze_chunk[f'tp_place_{level_name}_pct'] = safe_divide(tp_dist_from_entry, total_dist_to_level)

    return bronze_chunk


def queue_worker(task_queue: Manager.Queue) -> int:
    """
    The 'Consumer' worker function, which runs in a continuous loop.

    It retrieves tasks (DataFrame chunks) from the shared queue, enriches them
    using the globally available lookup structures, saves the results, and
    then waits for the next task. This ensures continuous processing and high CPU utilization.

    Args:
        task_queue: The shared multiprocessing Manager Queue from which to pull tasks.

    Returns:
        The total number of trades processed by this worker instance.
    """
    total_processed_in_worker = 0
    # Loop indefinitely until a shutdown signal (None) is received.
    while True:
        try:
            task = task_queue.get()
            # Check for the shutdown signal.
            if task is None:
                break # Exit the loop and terminate the worker.

            chunk_df, chunk_number = task
            # Skip if the chunk is empty.
            if chunk_df.empty:
                continue

            # Enrich the trade data using the pre-computed lookups.
            enriched_chunk = add_positioning_features(
                chunk_df, worker_feature_values_np, worker_time_to_idx_lookup,
                worker_col_to_idx, worker_levels_for_positioning
            )

            # If enrichment produced results, downcast types and save to a uniquely named file.
            if not enriched_chunk.empty:
                enriched_chunk = downcast_dtypes(enriched_chunk)
                output_path = os.path.join(worker_chunked_outcomes_dir, f"chunk_{chunk_number}.csv")
                enriched_chunk.to_csv(output_path, index=False)
                total_processed_in_worker += len(enriched_chunk)

        except Exception:
            # Catch any exceptions within the worker and print a traceback for debugging.
            print(f"WORKER ERROR processing chunk. See traceback below.")
            traceback.print_exc()
    return total_processed_in_worker


def create_silver_data(
    bronze_path: str, raw_path: str, features_path: str, chunked_outcomes_dir: str
) -> None:
    """
    Orchestrates the entire Silver Layer generation process for a single instrument.

    This function manages the two main stages: generating market features and then
    enriching the bronze trade data using those features in a parallel, memory-safe manner.

    Args:
        bronze_path: Path to the Bronze Layer CSV file for the instrument.
        raw_path: Path to the original raw OHLC data CSV file for the instrument.
        features_path: Path where the generated Silver Features CSV should be saved.
        chunked_outcomes_dir: Directory where the enriched trade chunks will be saved.
    """
    # Extract instrument name for logging and directory creation.
    instrument_name = os.path.basename(raw_path).replace('.csv', '')
    print(f"\n{'='*25}\nProcessing: {instrument_name}\n{'='*25}")

    # --- STEP 1: Create Silver Features Dataset ---
    print("STEP 1: Creating Silver Features dataset...")
    raw_df = robust_read_csv(raw_path)
    # Ensure there's enough data to calculate stable indicators after the warmup period.
    if len(raw_df) < INDICATOR_WARMUP_PERIOD + 1:
        print(f"[ERROR] SKIPPING: Not enough data for indicator warmup ({len(raw_df)} rows found, need >{INDICATOR_WARMUP_PERIOD}).")
        return

    # Generate all market features for the raw data.
    features_df = add_all_market_features(raw_df)
    del raw_df; gc.collect() # Free up memory by deleting the raw DataFrame.

    # Apply the warmup period by slicing the DataFrame.
    features_df = features_df.iloc[INDICATOR_WARMUP_PERIOD:].reset_index(drop=True)
    # Optimize memory usage by downcasting data types.
    features_df = downcast_dtypes(features_df)
    
    # Save the generated features to disk.
    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    features_df.to_csv(features_path, index=False)
    print(f"[SUCCESS] Silver Features saved to: {features_path}")

    # --- STEP 2: Prepare for Parallel Trade Enrichment ---
    print("\nSTEP 2: Preparing for PARALLEL chunk enrichment...")
    # Clean up the output directory if it already exists.
    if os.path.exists(chunked_outcomes_dir):
        shutil.rmtree(chunked_outcomes_dir)
    os.makedirs(chunked_outcomes_dir) # Create the output directory for chunks.
    print("Creating feature lookup structures for fast processing...")
    
    # Dynamically identify all relevant level columns needed for positioning features.
    # This includes base price columns, S/R, and key indicators like SMAs, BBands, ATRs.
    feature_cols = features_df.columns
    base_levels = ['time', 'open', 'high', 'low', 'close', 'support', 'resistance']
    # Define patterns to identify indicator columns that should be included in the lookup.
    indicator_patterns = ['SMA_', 'EMA_', 'BB_upper', 'BB_lower', 'ATR_level']
    # Construct the list of columns required for the lookup.
    level_cols = base_levels + [c for c in feature_cols if any(p in c for p in indicator_patterns)]
    
    # Create the efficient lookup structures from a subset of the features DataFrame.
    # Using a set() ensures uniqueness if a column name matches multiple patterns.
    lookup_df = features_df[list(set(level_cols))]
    # Determine the columns to be converted into the NumPy array for the lookup.
    cols_for_numpy = [c for c in lookup_df.columns if c != 'time']
    feature_values_np, time_to_idx, col_to_idx = create_feature_lookup_structures(lookup_df, cols_for_numpy)
    # Identify columns relevant for relative positioning calculations (excluding base prices).
    levels_for_positioning = [c for c in cols_for_numpy if c not in ['open', 'high', 'low', 'close']]
    # Free up memory by deleting intermediate DataFrames.
    del features_df, lookup_df; gc.collect()
    print("[SUCCESS] Lookup structures created.")

    # --- STEP 3: Execute Parallel Enrichment using a Producer-Consumer Queue ---
    try:
        # Count the number of trade rows in the Bronze CSV file.
        num_rows = sum(1 for _ in open(bronze_path, 'r')) - 1 # Subtract 1 for the header row.
    except FileNotFoundError:
        print(f"[ERROR] Bronze file not found at {bronze_path}. Cannot proceed.")
        return
    if num_rows <= 0:
        print("[INFO] Bronze file is empty or only contains a header. No trades to process.")
        return

    # Calculate the total number of chunks needed for processing.
    num_chunks = math.ceil(num_rows / CHUNK_SIZE)
    print(f"Found {num_rows:,} trades, processing in {num_chunks} chunks.")

    # Initialize a multiprocessing Manager to create a shared queue.
    manager = Manager()
    # Create a queue with a max size to prevent excessive memory usage.
    task_queue = manager.Queue(maxsize=MAX_CPU_USAGE * 2)
    # Define the arguments to be passed to each worker's initializer function.
    pool_init_args = (feature_values_np, time_to_idx, col_to_idx, levels_for_positioning, chunked_outcomes_dir)

    # Create a Pool of worker processes.
    with Pool(processes=MAX_CPU_USAGE, initializer=init_worker, initargs=pool_init_args) as pool:
        # Asynchronously start the worker processes. They will immediately wait for tasks.
        worker_results = [pool.apply_async(queue_worker, (task_queue,)) for _ in range(MAX_CPU_USAGE)]
        
        # The main process acts as the producer, reading the Bronze CSV in chunks.
        iterator = pd.read_csv(bronze_path, chunksize=CHUNK_SIZE, parse_dates=['entry_time'])
        # Feed the chunks into the task queue for workers to process.
        for i, chunk_df in enumerate(tqdm(iterator, total=num_chunks, desc="Feeding Chunks to Workers"), 1):
            task_queue.put((chunk_df, i))
            
        # After all chunks are queued, send a shutdown signal (None) to each worker.
        for _ in range(MAX_CPU_USAGE):
            task_queue.put(None)

        # Wait for all workers to complete and collect their results (total processed counts).
        total_trades_processed = sum(res.get() for res in worker_results)

    print(f"\n[SUCCESS] Enriched and chunked {total_trades_processed:,} trades.")
    print(f"Output saved to: {chunked_outcomes_dir}")


def main() -> None:
    """
    Main execution function: handles file discovery, user interaction,
    and orchestrates the processing of each selected file.
    """
    # Define and create necessary directories.
    core_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.abspath(os.path.join(core_dir, '..', 'raw_data'))
    bronze_dir = os.path.abspath(os.path.join(core_dir, '..', 'bronze_data'))
    features_dir = os.path.abspath(os.path.join(core_dir, '..', 'silver_data', 'features'))
    chunked_outcomes_base_dir = os.path.abspath(os.path.join(core_dir, '..', 'silver_data', 'chunked_outcomes'))
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(chunked_outcomes_base_dir, exist_ok=True)

    files_to_process = [] # List to store the names of files selected for processing.
    # Check if a specific file was provided as a command-line argument.
    target_file_arg = sys.argv[1] if len(sys.argv) > 1 else None

    if target_file_arg:
        # --- Targeted Mode ---
        print(f"[TARGET] Targeted Mode: Processing '{target_file_arg}'")
        # Verify the target file exists in the bronze data directory.
        if not os.path.exists(os.path.join(bronze_dir, target_file_arg)):
            print(f"[ERROR] Target file not found in bronze_data: {target_file_arg}")
        else:
            files_to_process = [target_file_arg] # Add the target file to the processing list.
    else:
        # --- Interactive Mode ---
        print("[SCAN] Interactive Mode: Scanning for new files...")
        try:
            # List all CSV files in the bronze directory.
            bronze_files = sorted([f for f in os.listdir(bronze_dir) if f.endswith('.csv')])
            # Identify files that have not yet been processed (i.e., lack a corresponding chunked output directory).
            new_files = [f for f in bronze_files if not os.path.exists(os.path.join(chunked_outcomes_base_dir, f.replace('.csv', '')))]
            
            if not new_files:
                print("[INFO] No new Bronze files to process.")
            else:
                # Present the user with a list of files to choose from.
                print("\n--- Select File(s) to Process ---")
                for i, f in enumerate(new_files): print(f"  [{i+1}] {f}")
                print("\nSelect multiple files with comma-separated numbers (e.g., 1,3,5)")
                user_input = input("Enter number(s) to process: ").strip()
                if user_input:
                    try:
                        # Parse user input and validate selected indices.
                        indices = [int(i.strip()) - 1 for i in user_input.split(',')]
                        files_to_process = [new_files[idx] for idx in sorted(set(indices)) if 0 <= idx < len(new_files)]
                    except ValueError:
                        print("[ERROR] Invalid input. Please enter numbers only.")
        except FileNotFoundError:
            print(f"[ERROR] The directory '{bronze_dir}' was not found.")

    # If no files are selected or found, exit gracefully.
    if not files_to_process:
        print("[INFO] No files selected or found for processing.")
        return

    # --- Main Execution Loop ---
    # Process each selected file sequentially to manage resources and logging.
    print(f"\n[QUEUE] Queued {len(files_to_process)} file(s): {files_to_process}")
    for filename in files_to_process:
        instrument_name = filename.replace('.csv', '')
        # Define all necessary input and output paths for the current file.
        paths = {
            "bronze_path": os.path.join(bronze_dir, filename),
            "raw_path": os.path.join(raw_dir, filename),
            "features_path": os.path.join(features_dir, filename),
            "chunked_outcomes_dir": os.path.join(chunked_outcomes_base_dir, instrument_name)
        }
        # Basic check to ensure the input files exist before proceeding.
        if not all(os.path.exists(paths[k]) for k in ["bronze_path", "raw_path"]):
            print(f"[WARNING] SKIPPING {filename}: Missing raw or bronze input file.")
            continue
        
        # Execute the main data processing function for the current file.
        try:
            create_silver_data(**paths)
        except Exception:
            # Catch any unexpected errors during processing and print a traceback.
            print(f"\n[FATAL ERROR] A critical error occurred while processing {filename}.")
            traceback.print_exc()

    print("\n" + "="*50 + "\n[COMPLETE] All Silver Layer data generation tasks are finished.")


if __name__ == "__main__":
    main()
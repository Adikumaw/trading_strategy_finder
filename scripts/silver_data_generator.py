# silver_data_generator.py (Optimized with NumPy lookup for memory and speed)

"""
Silver Layer: The Enrichment Engine

This script is the central feature engineering hub of the entire pipeline. It
takes the raw, high-volume trade simulations from the Bronze Layer and transforms
them into an intelligent, context-rich dataset ready for machine learning.

It operates in two distinct stages for each instrument:
1.  Market Feature Generation: It first consumes the raw OHLC price data and
    calculates a massive suite of over 200 technical indicators, candlestick
    patterns, and custom market context features for every single candle. This
    creates a complete, candle-by-candle "fingerprint" of the market's state.
    This comprehensive dataset is saved to `silver_data/features/`.

2.  Trade Enrichment & Chunking: It then reads the enormous Bronze Dataset in
    memory-safe chunks. For each potential winning trade, it calculates a
    powerful set of "relational positioning" features, which describe where a
    trade's SL/TP were relative to market structures. This is achieved using a
    highly efficient NumPy-based lookup method that avoids large memory overhead
    by not merging DataFrames. The final enriched trade data (without the base
    market features) is saved in chunks to `silver_data/chunked_outcomes/`.
"""

import os
import gc
import pandas as pd
import ta
import talib
import numba
import numpy as np
from tqdm import tqdm
import sys # <-- IMPORT SYS MODULE

# --- CONFIGURATION ---
# Defines the periods for various technical indicators to be calculated.
SMA_PERIODS = [20, 50, 100, 200]
EMA_PERIODS = [8, 13, 21, 50]
BBANDS_PERIOD, BBANDS_STD_DEV = 20, 2.0
RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
ATR_PERIOD = 14
ADX_PERIOD = 14
PIVOT_WINDOW = 10  # Window for calculating fractal-based Support/Resistance
PAST_LOOKBACKS = [3, 5, 10, 20, 50] # Lookback periods for price-action features
# The number of initial candles to discard to allow indicators to generate stable values.
INDICATOR_WARMUP_PERIOD = 200

# --- UTILITY & FEATURE FUNCTIONS ---

def downcast_dtypes(df):
    """
    Optimizes a DataFrame's memory usage by converting numeric columns to smaller dtypes.

    It iterates through all float64 and int64 columns and casts them to
    float32 and int32, respectively. This can significantly reduce the memory
    footprint of large DataFrames without a meaningful loss of precision for
    most financial data.

    Args:
        df (pd.DataFrame): The input DataFrame to optimize.

    Returns:
        pd.DataFrame: The DataFrame with downcasted numeric data types.
    """
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
    return df

def robust_read_csv(filepath):
    """
    Reads raw OHLC CSV data reliably, handling common formatting inconsistencies.

    This function is designed to be resilient to different CSV delimiters by using
    the 'python' engine with 'sep=None'. It automatically assigns standard column
    names, adds a dummy volume column if one is not present, converts columns
    to their proper numeric and datetime formats, and drops any rows with
    missing essential data.

    Args:
        filepath (str): The full path to the raw input CSV file.

    Returns:
        pd.DataFrame: A clean, sorted DataFrame with standardized column names
                      ('time', 'open', 'high', 'low', 'close', 'volume').
    """
    df = pd.read_csv(filepath, sep=None, engine="python", header=None)
    if df.shape[1] > 5:
        df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
    elif df.shape[1] == 5:
        df.columns = ['time', 'open', 'high', 'low', 'close']
        df['volume'] = 1 # Add a dummy volume column if not present
    else:
        raise ValueError(f"File '{filepath}' has fewer than 5 columns.")
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=['time'] + numeric_cols, inplace=True)
    return df.sort_values('time').reset_index(drop=True)

@numba.njit
def _calculate_s_r_numba(lows, highs, window):
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
    support_points, resistance_points = np.full(n, np.nan, dtype=np.float32), np.full(n, np.nan, dtype=np.float32)
    for i in range(window, n - window):
        window_lows = lows[i - window : i + window + 1]
        window_highs = highs[i - window : i + window + 1]
        if lows[i] == np.min(window_lows):
            support_points[i] = lows[i]
        if highs[i] == np.max(window_highs):
            resistance_points[i] = highs[i]
    return support_points, resistance_points

def add_support_resistance(df, window=PIVOT_WINDOW):
    """
    Calculates and adds forward-filled support and resistance levels to the DataFrame.

    This function acts as a wrapper for the core Numba S/R calculation. It extracts
    the necessary NumPy arrays from the DataFrame, calls the high-speed
    _calculate_s_r_numba function, and then forward-fills the resulting sparse
    support and resistance points. Forward-filling ensures that every candle has a
    value for the "last known" support and resistance level.

    Args:
        df (pd.DataFrame): The input market data DataFrame.
        window (int): The window size for the fractal calculation.

    Returns:
        pd.DataFrame: The original DataFrame with two new columns: 'support' and 'resistance'.
    """
    lows, highs = df["low"].values.astype(np.float32), df["high"].values.astype(np.float32)
    support_points, resistance_points = _calculate_s_r_numba(lows, highs, window)
    sr_df = pd.DataFrame({'support_points': support_points, 'resistance_points': resistance_points}, index=df.index)
    # Forward-fill to carry the last known S/R level forward in time
    sr_df["support"], sr_df["resistance"] = sr_df["support_points"].ffill(), sr_df["resistance_points"].ffill()
    return pd.concat([df, sr_df[['support', 'resistance']]], axis=1)

def add_all_market_features(df):
    """
    Generates a comprehensive suite of market features for the input OHLC data.

    This function is the primary feature engineering engine for market context. It
    calculates and appends a wide array of features in organized batches:
    1.  Standard technical indicators (SMAs, EMAs, Bollinger Bands, RSI, MACD, etc.).
    2.  All 60+ classic candlestick patterns recognized by the TA-Lib library.
    3.  Fractal-based support and resistance levels.
    4.  Time-based features (session, hour, weekday) and price-action characteristics
        (e.g., bullish ratio, average body size over lookback periods).

    Args:
        df (pd.DataFrame): The raw OHLC DataFrame with a 'time' column.

    Returns:
        pd.DataFrame: The original DataFrame enriched with over 200 new feature columns.
    """
    new_features_list = []
    
    # --- Batch 1: Standard Technical Indicators ---
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
    
    # Add ATR-based dynamic price levels to the indicator batch
    atr_series = indicator_df[f"ATR_{ATR_PERIOD}"]
    indicator_df["ATR_level_up_1x"] = df["close"] + atr_series
    indicator_df["ATR_level_down_1x"] = df["close"] - atr_series
    new_features_list.append(indicator_df)

    # --- Batch 2: Candlestick Patterns ---
    print("Calculating candlestick patterns...")
    # Use TA-Lib to efficiently generate scores for all classic candlestick patterns
    patterns_data = {p: getattr(talib, p)(df["open"], df["high"], df["low"], df["close"]) for p in talib.get_function_groups().get("Pattern Recognition", [])}
    new_features_list.append(pd.DataFrame(patterns_data, index=df.index))
    
    # --- Batch 3: Support & Resistance ---
    print("Calculating support and resistance...")
    df_with_sr = add_support_resistance(df.copy())
    new_features_list.append(df_with_sr[['support', 'resistance']])

    # --- Batch 4: Time-based and Price-Action Features ---
    print("Calculating time-based and price-action features...")
    price_action_df = pd.DataFrame(index=df.index)
    # Market session based on UTC hour
    price_action_df['session'] = df['time'].dt.hour.map(lambda h: 'London' if 7<=h<12 else 'London_NY_Overlap' if 12<=h<16 else 'New_York' if 16<=h<21 else 'Asian')
    price_action_df['hour'], price_action_df['weekday'] = df['time'].dt.hour, df['time'].dt.weekday
    is_bullish, body_size = (df['close'] > df['open']).astype(int), (df['close'] - df['open']).abs()
    for n in PAST_LOOKBACKS:
        price_action_df[f'bullish_ratio_last_{n}'] = is_bullish.rolling(n).mean()
        price_action_df[f'avg_body_last_{n}'] = body_size.rolling(n).mean()
        price_action_df[f'avg_range_last_{n}'] = (df['high'] - df['low']).rolling(n).mean()
        sma_col = f'SMA_{n}' if f'SMA_{n}' in indicator_df else 'SMA_20'
        price_action_df[f'close_{sma_col}_ratio'] = df['close'] / indicator_df[sma_col]
    # Market regime classification
    price_action_df['trend_regime'] = np.where(indicator_df['ADX'] > 25, 'trend', 'range')
    if f'ATR_{ATR_PERIOD}' in indicator_df:
        price_action_df['vol_regime'] = np.where(indicator_df[f'ATR_{ATR_PERIOD}'] > indicator_df[f'ATR_{ATR_PERIOD}'].rolling(50).mean(), 'high_vol', 'low_vol')
    new_features_list.append(price_action_df)

    # Combine the original data with all new feature batches
    return pd.concat([df] + new_features_list, axis=1)

def create_feature_lookup(features_df, level_cols):
    """
    Pre-computes and organizes feature data into highly efficient lookup structures.

    This is a critical optimization step that transforms the features DataFrame
    into a set of structures designed for near-instantaneous data retrieval.
    This avoids the need for slow, memory-intensive DataFrame merges later on.

    Args:
        features_df (pd.DataFrame): The DataFrame containing all market features,
                                    sorted by time.
        level_cols (list): A list of the column names that are needed for the
                           lookup calculations.

    Returns:
        tuple: A tuple containing three essential lookup components:
               - feature_values_np (np.array): A 2D NumPy array of the raw numeric
                 feature values for maximum computational speed.
               - time_to_idx_lookup (pd.Series): A mapping from a timestamp to its
                 integer row index in the NumPy array.
               - col_to_idx (dict): A mapping from a feature's column name to its
                 integer column index in the NumPy array.
    """
    # Ensure features_df is sorted by time, which is required for the lookup logic
    features_df = features_df.sort_values('time').reset_index(drop=True)
    
    # Create the mapping from column name to its integer index for the NumPy array
    col_to_idx = {col: i for i, col in enumerate(level_cols)}
    
    # Select only the necessary columns and convert to a NumPy array for speed
    feature_values_np = features_df[level_cols].values
    
    # Create the primary time-to-index lookup (Pandas Series is very fast for this)
    time_to_idx_lookup = pd.Series(features_df.index, index=features_df['time'])
    
    return feature_values_np, time_to_idx_lookup, col_to_idx

def add_positioning_features_lookup(bronze_chunk, feature_values_np, time_to_idx_lookup, col_to_idx):
    """
    Enriches a chunk of Bronze data with relational features using the fast lookup structures.

    This function takes a set of simulated trades and adds features that describe
    the placement of their SL and TP levels relative to various market structure
    levels (e.g., SMAs, Bollinger Bands, S/R). It performs these calculations using
    highly optimized, vectorized NumPy operations, leveraging the pre-computed
    lookup structures for extreme efficiency.

    Args:
        bronze_chunk (pd.DataFrame): A chunk of data from the Bronze trades file.
        feature_values_np (np.array): The NumPy array of market feature values.
        time_to_idx_lookup (pd.Series): The timestamp-to-row-index lookup Series.
        col_to_idx (dict): The column-name-to-column-index lookup dictionary.

    Returns:
        pd.DataFrame: The enriched DataFrame chunk with new relational positioning features.
    """
    # Find the row indices in the features array for each trade's entry time.
    # `reindex` with `method='ffill'` is a powerful way to perform a fast,
    # point-in-time correct (backward-looking) lookup.
    indices = time_to_idx_lookup.reindex(bronze_chunk['entry_time'], method='ffill').values
    
    # Handle trades that might occur before the first feature timestamp, which result in NaN
    valid_mask = ~np.isnan(indices)
    if not valid_mask.all():
        bronze_chunk = bronze_chunk.loc[valid_mask]
        indices = indices[valid_mask].astype(int)
        if bronze_chunk.empty:
            return pd.DataFrame()
    indices = indices.astype(int)

    # Directly pull all required feature rows from the main NumPy array in one go.
    # This is extremely fast and memory-efficient.
    features_for_chunk_np = feature_values_np[indices]

    enriched_chunk = bronze_chunk.copy()
    
    # Extract trade and feature data into NumPy arrays for maximum speed
    entry = enriched_chunk['entry_price'].values
    sl = enriched_chunk['sl_price'].values
    tp = enriched_chunk['tp_price'].values
    close_at_entry = features_for_chunk_np[:, col_to_idx['close']]

    def safe_divide(numerator, denominator):
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.divide(numerator, denominator)
        result[denominator == 0] = np.nan
        return result

    for level_name, level_idx in col_to_idx.items():
        # Get the entire column of level prices for the chunk from our NumPy array
        level_price = features_for_chunk_np[:, level_idx]
        
        # --- Feature Calculation 1: Distance in Basis Points ---
        enriched_chunk[f'sl_dist_to_{level_name}_bps'] = safe_divide((sl - level_price), close_at_entry) * 10000
        enriched_chunk[f'tp_dist_to_{level_name}_bps'] = safe_divide((tp - level_price), close_at_entry) * 10000

        # --- Feature Calculation 2: Placement as a Percentage ---
        total_dist_to_level = level_price - entry
        sl_dist_from_entry = sl - entry
        tp_dist_from_entry = tp - entry
        enriched_chunk[f'sl_placement_pct_to_{level_name}'] = safe_divide(sl_dist_from_entry, total_dist_to_level)
        enriched_chunk[f'tp_placement_pct_to_{level_name}'] = safe_divide(tp_dist_from_entry, total_dist_to_level)
            
    return enriched_chunk

def create_silver_data(bronze_path, raw_path, features_path, chunked_outcomes_dir):
    """
    Orchestrates the entire Silver Layer data generation process for a single instrument.

    This function manages the two main stages of the Silver Layer:
    1.  It first calls the feature engineering functions to generate and save a
        comprehensive market features file based on the raw OHLC data.
    2.  It then prepares the highly efficient lookup structures from this feature
        file.
    3.  Finally, it iterates through the massive Bronze trades file in memory-safe
        chunks, enriching each chunk with relational positioning features and
        saving the result to a dedicated instrument-specific directory.

    Args:
        bronze_path (str): Full path to the input Bronze trades CSV file.
        raw_path (str): Full path to the corresponding raw OHLC data file.
        features_path (str): Full path where the Silver market features file should be saved.
        chunked_outcomes_dir (str): Path to the directory where the enriched trade
                                    chunks should be saved.
    """
    print(f"\n{'='*25}\nProcessing: {os.path.basename(raw_path)}\n{'='*25}")

    # --- STEP 1: Create and Save the Full Silver Market Features Dataset ---
    print("STEP 1: Creating Silver Features dataset...")
    raw_df = robust_read_csv(raw_path)
    if len(raw_df) < INDICATOR_WARMUP_PERIOD + 1:
        print(f"❌ SKIPPING: Not enough data for indicator warmup."); return

    features_df = add_all_market_features(raw_df)
    del raw_df; gc.collect() # Free up memory
    
    # Remove the initial warmup period rows where indicators are unreliable
    features_df = features_df.iloc[INDICATOR_WARMUP_PERIOD:].reset_index(drop=True)
    features_df = downcast_dtypes(features_df)
    
    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    features_df.to_csv(features_path, index=False)
    print(f"✅ Silver Features saved to: {features_path}")
    
    # --- STEP 2: Create Enriched and Chunked Trade Outcomes ---
    print("\nSTEP 2: Creating ENRICHED and CHUNKED Silver Outcomes...")
    os.makedirs(chunked_outcomes_dir, exist_ok=True)
    
    # Read the massive bronze file in chunks to manage memory usage
    bronze_iterator = pd.read_csv(bronze_path, chunksize=500_000, parse_dates=['entry_time'])
    
    # Pre-computation: Build the highly efficient lookup structures ONCE before the main loop.
    print("Creating feature lookup structure for fast processing...")
    # It's better to read the saved, clean `features.csv` to ensure consistency.
    # We only read the columns we absolutely need for the lookup.
    all_level_cols = ['time', 'open', 'high', 'low', 'close', 'support', 'resistance',
                      'BB_upper', 'BB_lower', 'ATR_level_up_1x', 'ATR_level_down_1x']
    # Dynamically find SMA/EMA columns as they might vary
    temp_df_cols = pd.read_csv(features_path, nrows=0).columns
    all_level_cols.extend([col for col in temp_df_cols if 'SMA_' in col or 'EMA_' in col])
    
    features_df_for_lookup = pd.read_csv(features_path, parse_dates=['time'], usecols=all_level_cols)
    
    # Define the columns that will be part of the NumPy array
    level_cols_for_numpy = [c for c in all_level_cols if c != 'time']
    
    feature_values_np, time_to_idx_lookup, col_to_idx = create_feature_lookup(features_df_for_lookup, level_cols_for_numpy)
    del features_df_for_lookup, temp_df_cols; gc.collect()
    print("✅ Lookup structure created.")

    chunk_counter = 1
    for chunk in tqdm(bronze_iterator, desc="  Enriching Bronze Chunks"):
        if chunk.empty: continue
        
        # Pass the bronze chunk and the lookup structures to the enrichment function
        enriched_chunk = add_positioning_features_lookup(chunk, feature_values_np, time_to_idx_lookup, col_to_idx)
        
        if not enriched_chunk.empty:
            enriched_chunk = downcast_dtypes(enriched_chunk)
            chunk_output_path = os.path.join(chunked_outcomes_dir, f"chunk_{chunk_counter}.csv")
            enriched_chunk.to_csv(chunk_output_path, index=False)
            chunk_counter += 1
            
    print(f"✅ Enriched and chunked Silver Outcomes saved to: {chunked_outcomes_dir}")

if __name__ == "__main__":
    """
    Main execution block.
    
    This script can be run in two modes:
    1. Discovery Mode (no arguments): Scans for all new bronze files and processes them.
       Example: `python scripts/silver_data_generator.py`
       
    2. Targeted Mode (one argument): Processes only the single file specified.
       Example: `python scripts/silver_data_generator.py XAUUSD15.csv`
    """
    core_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir, bronze_dir = [os.path.abspath(os.path.join(core_dir, d)) for d in ['../raw_data', '../bronze_data']]
    features_dir = os.path.abspath(os.path.join(core_dir, '../silver_data/features'))
    chunked_outcomes_dir = os.path.abspath(os.path.join(core_dir, '../silver_data/chunked_outcomes'))
    
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(chunked_outcomes_dir, exist_ok=True)

    # --- NEW: DUAL-MODE FILE DISCOVERY LOGIC ---
    target_file_arg = sys.argv[1] if len(sys.argv) > 1 else None

    if target_file_arg:
        # --- Targeted Mode ---
        print(f"🎯 Targeted Mode: Processing single file '{target_file_arg}'")
        bronze_path_check = os.path.join(bronze_dir, target_file_arg)
        if not os.path.exists(bronze_path_check):
            print(f"❌ Error: Target file not found in bronze_data directory: {target_file_arg}")
            files_to_process = []
        else:
            files_to_process = [target_file_arg]
    else:
        # --- Discovery Mode (Default) ---
        print("🔍 Discovery Mode: Scanning for all new files...")
        try:
            # Find all available bronze files
            bronze_files = [f for f in os.listdir(bronze_dir) if f.endswith('.csv')]
            
            # Check which ones already have a corresponding silver 'chunked_outcomes' directory
            files_to_process = []
            for f in bronze_files:
                instrument_name = f.replace('.csv', '')
                instrument_chunked_dir = os.path.join(chunked_outcomes_dir, instrument_name)
                if not os.path.exists(instrument_chunked_dir):
                    files_to_process.append(f)
        except FileNotFoundError:
            print(f"❌ Error: The directory '{bronze_dir}' was not found.")
            files_to_process = []

    if not files_to_process:
        print("ℹ️ No new files to process.")
    else:
        print(f"Found {len(files_to_process)} file(s) to process...")
        # --- Main Loop: Iterate through each instrument ---
        for fname in files_to_process:
            instrument_name = fname.replace('.csv', '')
            bronze_path = os.path.join(bronze_dir, fname)
            raw_path = os.path.join(raw_dir, fname)
            features_path = os.path.join(features_dir, fname)
            # The output directory for chunks is specific to each instrument
            instrument_chunked_outcomes_dir = os.path.join(chunked_outcomes_dir, instrument_name)
            
            # --- Pre-computation Checks ---
            if not os.path.exists(raw_path):
                print(f"⚠️ SKIPPING {fname}: Corresponding raw file not found."); continue
            
            if not os.path.exists(bronze_path):
                print(f"⚠️ SKIPPING {fname}: Corresponding bronze file not found."); continue
            
            # --- Execute Processing ---
            try:
                create_silver_data(bronze_path, raw_path, features_path, instrument_chunked_outcomes_dir)
            except Exception as e:
                print(f"❌ FAILED to process {fname}. Error: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "="*50 + "\n✅ All silver data generation complete.")
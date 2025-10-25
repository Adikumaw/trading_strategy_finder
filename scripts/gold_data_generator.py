# gold_data_generator.py (Parallelized Version)

"""
Gold Layer: The Machine Learning Preprocessor

This script represents the crucial final stage of data preparation in the
pipeline. It acts as a specialized transformer, converting the human-readable,
context-rich Silver `features` dataset into a purely numerical, normalized, and
standardized format that is perfectly optimized for machine learning algorithms.

Its sole purpose is to "translate" market context into the mathematical
language that ML models understand, performing several key transformations:
- Relational Transformation: Converts absolute price levels into a normalized
  distance from the current close price.
- Categorical Encoding: Converts text-based features (e.g., 'session') into
  binary (0/1) columns.
- Pattern Compression: Bins noisy candlestick pattern scores into a simple,
  discrete scale.
- Standardization: Rescales all other numerical features to a common scale
  (mean 0, std 1).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import re
from tqdm import tqdm
import gc
from multiprocessing import Pool, cpu_count
import sys # <-- IMPORT SYS MODULE

# --- CONFIGURATION ---

# Sets the maximum number of CPU cores to use for multiprocessing.
# Leaves 2 cores free to ensure system responsiveness.
MAX_CPU_USAGE = max(1, cpu_count() - 2)

# --- CORE FUNCTIONS ---

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

def create_gold_features(features_df):
    """
    Transforms a Silver-layer DataFrame into a Gold-layer, ML-ready dataset.

    This function orchestrates the entire machine learning preprocessing pipeline.
    It takes the human-readable, feature-rich Silver dataset and applies a series
    of transformations to make it suitable for ML models. These steps include
    converting absolute price levels to a normalized distance from the close,
    one-hot encoding categorical variables, compressing candlestick pattern
    scores into a discrete scale, and standardizing all remaining numeric
    features to have a mean of zero and a standard deviation of one.

    Args:
        features_df (pd.DataFrame): The human-readable features dataset from
                                    the Silver layer, containing a 'time' column
                                    and various market features.

    Returns:
        pd.DataFrame: A fully transformed, purely numerical, and standardized
                      DataFrame ready for machine learning tasks.
    """
    df = features_df.copy()

    # --- 1. Relational Transformation: Convert absolute prices to relative distance ---
    # Define regex patterns to identify all columns that represent an absolute price level.
    abs_price_patterns = [
        r'^(open|high|low|close)$', r'^SMA_\d+$', r'^EMA_\d+$',
        r'^BB_(upper|lower)$', r'^(support|resistance)$', r'^ATR_level_.+$'
    ]
    abs_price_cols = [col for col in df.columns for pattern in abs_price_patterns if re.match(pattern, col)]
    
    # For each price column, calculate its normalized distance from that candle's close price.
    # This makes the feature scale-invariant and timeless.
    if 'close' in df.columns:
        close_series = df['close']
        for col in abs_price_cols:
            if col != 'close':
                df[f'{col}_dist_norm'] = (close_series - df[col]) / close_series.replace(0, np.nan)
    
    # Drop the original absolute price columns and the volume column.
    df.drop(columns=list(set(abs_price_cols) | {'volume'}), inplace=True, errors='ignore')

    # --- 2. Categorical Encoding: Convert text columns to binary features ---
    categorical_cols = ['session', 'trend_regime', 'vol_regime']
    # `get_dummies` performs one-hot encoding. `drop_first=True` avoids multicollinearity.
    df = pd.get_dummies(df, columns=[c for c in categorical_cols if c in df.columns], drop_first=True)

    # --- 3. Pattern Compression: Bin candlestick pattern scores ---
    candle_cols = [col for col in df.columns if col.startswith("CDL")]
    def compress_pattern(v):
        """Converts TA-Lib scores (-100 to 100) to a 5-point scale."""
        if v >= 80: return 1.0     # Strong Bullish
        elif v > 0: return 0.5     # Weak Bullish
        elif v <= -80: return -1.0 # Strong Bearish
        elif v < 0: return -0.5    # Weak Bearish
        else: return 0.0           # Neutral / No Pattern
    for col in candle_cols:
        df[col] = df[col].fillna(0).apply(compress_pattern)

    # --- 4. Standardization: Scale all remaining numerical features ---
    # Identify all columns that should NOT be scaled (time, patterns, one-hot encoded).
    one_hot_cols = [col for col in df.columns if any(s in col for s in ['session_', 'trend_regime_', 'vol_regime_'])]
    non_scalable_cols = set(candle_cols + one_hot_cols + ['time'])
    
    # Identify the columns that need to be scaled to have a mean of 0 and std dev of 1.
    numeric_columns_to_scale = [
        col for col in df.columns if col not in non_scalable_cols and
        df[col].dtype in ['float64', 'float32', 'int64', 'int32']
    ]
    
    # Apply StandardScaler.
    scaler = StandardScaler()
    df[numeric_columns_to_scale] = scaler.fit_transform(df[numeric_columns_to_scale].fillna(0))
    
    # --- 5. Final Cleanup ---
    return downcast_dtypes(df)

def process_file_in_parallel(file_path_tuple):
    """
    A worker function to process a single file for parallel execution.

    This function acts as a self-contained wrapper for the entire workflow of
    processing one file. It reads a Silver features CSV, passes it to the
    `create_gold_features` function for transformation, and saves the resulting
    ML-ready DataFrame to the specified output path. It is designed to be
    called by a multiprocessing `Pool`.

    Args:
        file_path_tuple (tuple): A tuple containing two string elements:
                                 (input_silver_path, output_gold_path).

    Returns:
        str: A status message indicating the success or failure of the operation,
             suitable for logging.
    """
    silver_path, gold_path = file_path_tuple
    fname = os.path.basename(silver_path)
    try:
        features_df = pd.read_csv(silver_path, parse_dates=['time'])
        gold_dataset = create_gold_features(features_df)
        gold_dataset.to_csv(gold_path, index=False)
        return f"[SUCCESS] Success! Gold data generated for {fname}."
    except Exception as e:
        return f"[ERROR] FAILED to process {fname}. Error: {e}"

if __name__ == "__main__":
    """
    Main execution block.
    
    This script can be run in two modes:
    1. Discovery Mode (no arguments): Scans for all new silver files and processes them.
       Example: `python scripts/gold_data_generator.py`
       
    2. Targeted Mode (one argument): Processes only the single file specified.
       Example: `python scripts/gold_data_generator.py XAUUSD15.csv`
    """
    # --- Define Project Directory Structure ---
    core_dir = os.path.dirname(os.path.abspath(__file__))
    silver_features_dir = os.path.abspath(os.path.join(core_dir, '..', 'silver_data', 'features'))
    gold_features_dir = os.path.abspath(os.path.join(core_dir, '..', 'gold_data', 'features'))
    os.makedirs(gold_features_dir, exist_ok=True)

    # --- NEW: DUAL-MODE FILE DISCOVERY LOGIC ---
    target_file_arg = sys.argv[1] if len(sys.argv) > 1 else None

    if target_file_arg:
        # --- Targeted Mode ---
        print(f"[TARGET] Targeted Mode: Processing single file '{target_file_arg}'")
        silver_path_check = os.path.join(silver_features_dir, target_file_arg)
        if not os.path.exists(silver_path_check):
            print(f"[ERROR] Error: Target file not found in silver_data/features directory: {target_file_arg}")
            files_to_process = []
        else:
            files_to_process = [target_file_arg]
    else:
        # --- Discovery Mode (Default) ---
        print("[SCAN] Discovery Mode: Scanning for all new files...")
        try:
            # Find all available silver files
            all_files = [f for f in os.listdir(silver_features_dir) if f.endswith('.csv')]
            # Check which files already have a corresponding Gold output to make the script resumable.
            files_to_process = [f for f in all_files if not os.path.exists(os.path.join(gold_features_dir, f))]
        except FileNotFoundError:
            print(f"[ERROR] Error: Directory not found: '{silver_features_dir}'.")
            files_to_process = []

    if not files_to_process:
        print("[INFO] No new files to process.")
    else:
        print(f"Found {len(files_to_process)} new silver feature file(s) to process.")
        
        # --- Configure Multiprocessing ---
        # If in targeted mode or only one file, don't ask, just use multiprocessing logic.
        if target_file_arg or len(files_to_process) == 1:
            use_multiprocessing = True
        else:
            use_multiprocessing = input("Use multiprocessing? (y/n): ").strip().lower() == 'y'
        
        if use_multiprocessing:
            num_processes = MAX_CPU_USAGE
        else:
            num_processes = 1
        
        # --- Prepare Processing Tasks ---
        tasks = [(os.path.join(silver_features_dir, fname), os.path.join(gold_features_dir, fname)) for fname in files_to_process]
        
        # --- Execute Processing ---
        effective_num_processes = min(num_processes, len(tasks))
        print(f"\n[INFO] Starting processing with {effective_num_processes} worker(s)...")
        
        if effective_num_processes > 1:
            with Pool(processes=effective_num_processes) as pool:
                results = list(tqdm(pool.imap_unordered(process_file_in_parallel, tasks), total=len(tasks), desc="Processing Files"))
        else:
            results = [process_file_in_parallel(task) for task in tqdm(tasks, desc="Processing Files")]

        # --- Display Summary ---
        print("\n--- Processing Summary ---")
        for res in results:
            print(res)

    print("\n" + "="*50 + "\n[SUCCESS] All gold data generation complete.")
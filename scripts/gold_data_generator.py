# gold_data_generator.py (V3.0 - Final Polish & Documentation)

"""
Gold Layer: The Machine Learning Preprocessor

This script represents the crucial final stage of data preparation in the
pipeline. It acts as a specialized transformer, converting the human-readable,
context-rich Silver `features` dataset into a purely numerical, normalized, and
standardized Parquet file that is perfectly optimized for machine learning.

Its sole purpose is to "translate" market context into the mathematical
language that ML models understand, performing several key transformations:
- Relational Transformation: Converts absolute price levels into a normalized
  distance from the current close price, making features scale-invariant.
- Categorical Encoding: Converts text-based features (e.g., 'session', 'regime')
  into binary (0/1) columns via one-hot encoding.
- Pattern Compression: Bins noisy candlestick pattern scores into a simple,
  discrete 5-point scale to reduce noise.
- Standardization: Rescales all other numerical features to a common scale
  (mean 0, std 1) using StandardScaler.
"""

import os
import re
import sys
import traceback
from typing import List, Tuple, Set
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

# ### <<< CHANGE: Added robust dependency checks for all key libraries.
try:
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("[FATAL] 'scikit-learn' library not found. Please run 'pip install scikit-learn'.")
    sys.exit(1)
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print("[FATAL] 'pyarrow' library not found. Please run 'pip install pyarrow'.")
    sys.exit(1)


# --- CONFIGURATION ---
# This script processes files serially as the ML preprocessing steps for each
# file are generally fast and do not benefit significantly from parallelization overhead.

# --- HELPER FUNCTIONS ---

def downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimizes a DataFrame's memory usage by downcasting numeric types."""
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
    return df

# ### <<< NEW FUNCTION: Part of the `create_gold_features` refactor.
def _transform_relational_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts absolute price columns to be relative to the current close price.
    This normalization is crucial for making the features scale-invariant, allowing
    a model to learn patterns that are independent of the absolute price level.
    """
    print("  - Applying relational transformation...")
    if 'close' not in df.columns:
        raise KeyError("The required 'close' column is not present in the input DataFrame.")

    # Regex to find all columns that represent an absolute price level.
    abs_price_patterns = re.compile(
        r'^(open|high|low|close)$|^(SMA|EMA)_\d+$|^BB_(upper|lower)_\d+$|'
        r'^(support|resistance)$|^ATR_level_.+_\d+$'
    )
    abs_price_cols = [col for col in df.columns if abs_price_patterns.match(col)]
    
    close_series = df['close']
    for col in abs_price_cols:
        if col != 'close':
            # Create new feature: (level_price - close_price) / close_price
            df[f'{col}_dist_norm'] = (df[col] - close_series) / close_series.replace(0, np.nan)
    
    # Drop the original absolute price columns and unused volume.
    df.drop(columns=list(set(abs_price_cols) | {'volume'}), inplace=True, errors='ignore')
    return df

# ### <<< NEW FUNCTION: Part of the `create_gold_features` refactor.
def _encode_categorical_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Converts text-based categorical columns into numerical binary (0/1) columns
    using one-hot encoding. Returns the transformed DataFrame and the original
    categorical column names for later reference.
    """
    print("  - Encoding categorical features...")
    categorical_cols = [
        col for col in df.columns
        if col.startswith(('session', 'trend_regime', 'vol_regime'))
    ]
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=False, dtype=float)
    return df, categorical_cols

# ### <<< NEW FUNCTION: Part of the `create_gold_features` refactor.
def _compress_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bins noisy TA-Lib candlestick scores (-200 to 200) into a simple, discrete
    5-point scale (-1.0, -0.5, 0, 0.5, 1.0) to reduce noise and capture the
    essential signal (strong/weak pattern).
    """
    print("  - Compressing candlestick patterns...")
    candle_cols = [col for col in df.columns if col.startswith("CDL")]
    
    def compress(v):
        if v >= 100: return 1.0   # Strong Bullish
        if v > 0: return 0.5     # Weak Bullish
        if v <= -100: return -1.0  # Strong Bearish
        if v < 0: return -0.5    # Weak Bearish
        return 0.0               # Neutral
        
    for col in candle_cols:
        df[col] = df[col].fillna(0).apply(compress)
    return df

# ### <<< NEW FUNCTION: Part of the `create_gold_features` refactor.
def _scale_numeric_features(
    df: pd.DataFrame, original_cat_cols: List[str]
) -> pd.DataFrame:
    """
    Standardizes all numerical features (except for candlestick and one-hot encoded
    columns) to have a mean of 0 and a standard deviation of 1.
    """
    print("  - Scaling numeric features...")
    candle_cols = {col for col in df.columns if col.startswith("CDL")}
    
    # Dynamically find all columns created by pd.get_dummies to exclude them from scaling.
    one_hot_cols = {
        col for col in df.columns
        if any(cat_col_base in col for cat_col_base in original_cat_cols)
    }
    
    non_scalable_cols = candle_cols | one_hot_cols | {'time'}
    
    cols_to_scale = [
        col for col in df.columns
        if col not in non_scalable_cols and pd.api.types.is_numeric_dtype(df[col])
    ]
    
    if cols_to_scale:
        scaler = StandardScaler()
        # Fill NaNs with 0 before scaling; this assumes missing values have a neutral impact.
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale].fillna(0))
    
    return df

# ### <<< REFACTORED FUNCTION: Now acts as a clean orchestrator.
def create_gold_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Orchestrates the full preprocessing pipeline to transform a Silver DataFrame
    into a Gold-layer, ML-ready dataset.
    """
    df = features_df.copy()
    
    df = _transform_relational_features(df)
    df, original_cat_cols = _encode_categorical_features(df)
    df = _compress_candlestick_patterns(df)
    df = _scale_numeric_features(df, original_cat_cols)
    
    print("  - Finalizing and downcasting data types...")
    return downcast_dtypes(df)


def _process_single_file(paths_tuple: Tuple[str, str]) -> str:
    """
    Reads, processes, and saves a single Silver feature file into the Gold format.
    """
    silver_path, gold_path = paths_tuple
    fname = os.path.basename(silver_path)
    try:
        features_df = pd.read_csv(silver_path, parse_dates=['time'])
        gold_dataset = create_gold_features(features_df)
        
        # ### <<< CHANGE: Save output to Parquet for consistency and performance.
        gold_dataset.to_parquet(gold_path, index=False)
        return f"[SUCCESS] Gold data generated for {fname}."
    except Exception as e:
        print(f"\n[FATAL ERROR] An unexpected error occurred while processing {fname}.")
        traceback.print_exc()
        return f"[ERROR] FAILED to process {fname}."

# ### <<< NEW FUNCTION: Standardized interactive menu.
def _select_files_interactively(silver_dir: str, gold_dir: str) -> List[str]:
    """Scans for new Silver files and prompts the user to select which to process."""
    print("[INFO] Interactive Mode: Scanning for new files...")
    try:
        silver_files = sorted([f for f in os.listdir(silver_dir) if f.endswith('.csv')])
        gold_bases = {os.path.splitext(f)[0] for f in os.listdir(gold_dir) if f.endswith('.parquet')}
        new_files = [f for f in silver_files if os.path.splitext(f)[0] not in gold_bases]

        if not new_files:
            print("[INFO] No new Silver feature files to process.")
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
            indices = {int(i.strip()) - 1 for i in user_input.split(',')}
            for idx in sorted(indices):
                if 0 <= idx < len(new_files): selected_files.append(new_files[idx])
                else: print(f"[WARN] Invalid selection '{idx + 1}' ignored.")
            return selected_files
        except ValueError:
            print("[ERROR] Invalid input. Please enter numbers (e.g., 1,3) or 'a'.")
            return []
    except FileNotFoundError:
        print(f"[ERROR] The Silver features directory was not found at: {silver_dir}")
        return []

def main() -> None:
    """Main execution function: handles file discovery and orchestrates processing."""
    start_time = time.time()
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    SILVER_FEATURES_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'silver_data', 'features'))
    GOLD_FEATURES_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'gold_data', 'features'))
    os.makedirs(GOLD_FEATURES_DIR, exist_ok=True)

    print("--- Gold Layer: The ML Preprocessor (Parquet Edition) ---")

    target_file_arg = sys.argv[1] if len(sys.argv) > 1 else None
    if target_file_arg:
        print(f"\n[INFO] Targeted Mode: Processing '{target_file_arg}'")
        silver_path_check = os.path.join(SILVER_FEATURES_DIR, target_file_arg)
        files_to_process = [target_file_arg] if os.path.exists(silver_path_check) else []
        if not files_to_process: print(f"[ERROR] Target file not found: {silver_path_check}")
    else:
        files_to_process = _select_files_interactively(SILVER_FEATURES_DIR, GOLD_FEATURES_DIR)

    if not files_to_process:
        print("\n[INFO] No files selected or found for processing. Exiting.")
    else:
        print(f"\n[QUEUE] Queued {len(files_to_process)} file(s): {', '.join(files_to_process)}")
        for filename in files_to_process:
            print("\n" + "="*50 + f"\nProcessing {filename}...")
            silver_path = os.path.join(SILVER_FEATURES_DIR, filename)
            # ### <<< CHANGE: Output file is now .parquet
            gold_path = os.path.join(GOLD_FEATURES_DIR, filename.replace('.csv', '.parquet'))
            result = _process_single_file((silver_path, gold_path))
            print(result)

    end_time = time.time()
    print(f"\nGold Layer generation finished. Total time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
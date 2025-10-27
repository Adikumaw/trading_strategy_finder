# gold_data_generator.py (V2.1 - Serial Processing)

"""
Gold Layer: The Machine Learning Preprocessor

This script represents the crucial final stage of data preparation in the
pipeline. It acts as a specialized transformer, converting the human-readable,
context-rich Silver `features` dataset into a purely numerical, normalized, and
standardized format that is perfectly optimized for machine learning algorithms.

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

import gc
import os
import re
import sys
import traceback
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
# This script does not use multiprocessing for file processing to maintain
# consistency with the Bronze and Silver layers. It processes files serially.


# --- CORE FUNCTIONS ---

def downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimizes a DataFrame's memory usage by downcasting numeric types.

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


def create_gold_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms a Silver-layer DataFrame into a Gold-layer, ML-ready dataset.

    This function orchestrates the entire ML preprocessing pipeline. It applies
    a series of transformations to make the Silver dataset suitable for ML
    models, ensuring all data is numerical, normalized, and standardized.

    Args:
        features_df: The human-readable features dataset from the Silver layer.

    Returns:
        A fully transformed, purely numerical, and standardized DataFrame
        ready for machine learning tasks.
    """
    df = features_df.copy()

    # --- 1. Relational Transformation: Convert absolute prices to relative distance ---
    # This crucial step makes features scale-invariant. Instead of using an
    # absolute price like SMA_50 = 1.1234, we store how far it is from the
    # current close (e.g., 0.005 or 0.5%).
    abs_price_patterns = re.compile(
        r'^(open|high|low|close)$|'
        r'^(SMA|EMA)_\d+$|'
        r'^BB_(upper|lower)_\d+$|'
        r'^(support|resistance)$|'
        r'^ATR_level_.+_\d+$'
    )
    abs_price_cols = [col for col in df.columns if abs_price_patterns.match(col)]
    
    if 'close' in df.columns:
        close_series = df['close']
        for col in abs_price_cols:
            if col != 'close':
                df[f'{col}_dist_norm'] = (df[col] - close_series) / close_series.replace(0, np.nan)
    
    df.drop(columns=list(set(abs_price_cols) | {'volume'}), inplace=True, errors='ignore')

    # --- 2. Categorical Encoding: Convert text columns to binary features ---
    # One-hot encoding transforms categorical text data (e.g., 'session_London')
    # into a binary (0/1) format that models can interpret.
    categorical_cols = [
        col for col in df.columns
        if col.startswith(('session', 'trend_regime', 'vol_regime'))
    ]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # --- 3. Pattern Compression: Bin candlestick pattern scores ---
    # TA-Lib's scores (-100 to 100) can be noisy. Compressing them into a
    # 5-point scale helps the model focus on the signal (strong/weak pattern)
    # rather than the noise.
    candle_cols = [col for col in df.columns if col.startswith("CDL")]
    def compress_pattern(v: float) -> float:
        if v >= 80: return 1.0     # Strong Bullish
        if v > 0: return 0.5     # Weak Bullish
        if v <= -80: return -1.0    # Strong Bearish
        if v < 0: return -0.5    # Weak Bearish
        return 0.0               # Neutral
    
    for col in candle_cols:
        df[col] = df[col].fillna(0).apply(compress_pattern)

    # --- 4. Standardization: Scale all remaining numerical features ---
    # StandardScaler transforms features to have a mean of 0 and a standard
    # deviation of 1, preventing features with larger scales from unfairly
    # dominating the learning process.
    # Dynamically find one-hot encoded columns to exclude them from scaling.
    one_hot_cols = [
        col for col in df.columns
        if any(cat_col in col for cat_col in [c + '_' for c in categorical_cols])
    ]
    non_scalable_cols = set(candle_cols + one_hot_cols + ['time'])
    
    numeric_columns_to_scale = [
        col for col in df.columns
        if col not in non_scalable_cols and pd.api.types.is_numeric_dtype(df[col])
    ]
    
    if numeric_columns_to_scale:
        scaler = StandardScaler()
        df[numeric_columns_to_scale] = scaler.fit_transform(
            df[numeric_columns_to_scale].fillna(0)
        )
    
    # --- 5. Final Cleanup ---
    return downcast_dtypes(df)


def process_and_save_file(paths_tuple: Tuple[str, str]) -> str:
    """
    A self-contained function to process a single file.

    It reads a Silver features CSV, transforms it into an ML-ready Gold
    dataset using `create_gold_features`, and saves the result.

    Args:
        paths_tuple: A tuple containing (input_silver_path, output_gold_path).

    Returns:
        A status message indicating the success or failure of the operation.
    """
    silver_path, gold_path = paths_tuple
    fname = os.path.basename(silver_path)
    try:
        features_df = pd.read_csv(silver_path, parse_dates=['time'])
        gold_dataset = create_gold_features(features_df)
        gold_dataset.to_csv(gold_path, index=False)
        return f"[SUCCESS] Gold data generated for {fname}."
    except Exception as e:
        print(f"An error occurred while processing {fname}: {e}")
        traceback.print_exc()
        return f"[ERROR] FAILED to process {fname}."


def main() -> None:
    """
    Main execution function: handles file discovery, user interaction,
    and orchestrates the serial processing of each file.
    """
    # --- Define Project Directory Structure ---
    core_dir = os.path.dirname(os.path.abspath(__file__))
    silver_features_dir = os.path.abspath(os.path.join(core_dir, '..', 'silver_data', 'features'))
    gold_features_dir = os.path.abspath(os.path.join(core_dir, '..', 'gold_data', 'features'))
    os.makedirs(gold_features_dir, exist_ok=True)

    files_to_process = []
    target_file_arg = sys.argv[1] if len(sys.argv) > 1 else None

    if target_file_arg:
        # --- Targeted Mode (for automation) ---
        print(f"[TARGET] Targeted Mode: Processing '{target_file_arg}'")
        silver_path_check = os.path.join(silver_features_dir, target_file_arg)
        if not os.path.exists(silver_path_check):
            print(f"[ERROR] Target file not found: {silver_path_check}")
        else:
            files_to_process = [target_file_arg]
    else:
        # --- Interactive Mode (for manual runs) ---
        print("[SCAN] Interactive Mode: Scanning for new files...")
        try:
            silver_files = sorted([f for f in os.listdir(silver_features_dir) if f.endswith('.csv')])
            gold_files = os.listdir(gold_features_dir)
            new_files = [f for f in silver_files if f not in gold_files]
            
            if not new_files:
                print("[INFO] No new Silver feature files to process.")
            else:
                print("\n--- Select File(s) to Process ---")
                for i, f in enumerate(new_files):
                    print(f"  [{i+1}] {f}")
                print("\nSelect multiple files with comma-separated numbers (e.g., 1,3,5)")
                user_input = input("Enter number(s) to process: ").strip()
                if user_input:
                    try:
                        indices = [int(i.strip()) - 1 for i in user_input.split(',')]
                        files_to_process = [new_files[idx] for idx in sorted(set(indices)) if 0 <= idx < len(new_files)]
                    except ValueError:
                        print("[ERROR] Invalid input. Please enter numbers only.")
        except FileNotFoundError:
            print(f"[ERROR] The Silver features directory was not found at: {silver_features_dir}")

    if not files_to_process:
        print("[INFO] No files selected or found for processing.")
        return

    print(f"\n[QUEUE] Queued {len(files_to_process)} file(s) for serial processing.")
    
    # --- Serial Execution Loop ---
    # The script processes files one by one to maintain architectural consistency
    # with the Bronze and Silver layers.
    for filename in files_to_process:
        print("\n" + "="*50 + f"\nProcessing {filename}...")
        silver_path = os.path.join(silver_features_dir, filename)
        gold_path = os.path.join(gold_features_dir, filename)
        
        # Call the processing function directly for each file.
        result = process_and_save_file((silver_path, gold_path))
        print(result)

    print("\n" + "="*50 + "\n[COMPLETE] All Gold Layer data generation tasks are finished.")


if __name__ == "__main__":
    main()
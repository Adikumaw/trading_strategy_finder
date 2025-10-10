# gold_data_generator_nolimits.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import re
from tqdm import tqdm
import gc
from multiprocessing import Pool, cpu_count

# --- CONFIGURATION ---
MAX_CPU_USAGE = max(1, cpu_count() - 2)

# --- CORE FUNCTIONS (Unchanged from original script) ---
def downcast_dtypes(df):
    """Downcasts numeric columns for memory efficiency."""
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
    return df

def create_gold_features(features_df):
    """
    Transforms the silver features dataset into a normalized, ML-ready gold feature set.
    """
    # Using .copy() to avoid SettingWithCopyWarning in multiprocessing environments
    df = features_df.copy()

    # --- 1. Transform Absolute Price Features into Relational Features ---
    abs_price_patterns = [
        r'^(open|high|low|close)$', r'^SMA_\d+$', r'^EMA_\d+$',
        r'^BB_(upper|lower)$', r'^(support|resistance)$', r'^ATR_level_.+$'
    ]
    abs_price_cols = [col for col in df.columns for pattern in abs_price_patterns if re.match(pattern, col)]
    
    if 'close' in df.columns:
        close_series = df['close']
        for col in abs_price_cols:
            if col != 'close':
                df[f'{col}_dist_norm'] = (close_series - df[col]) / close_series.replace(0, np.nan)

    # --- 2. Drop Original Price Columns ---
    df.drop(columns=list(set(abs_price_cols) | {'volume'}), inplace=True, errors='ignore')

    # --- 3. Process Categorical, Candle, and Numeric Features ---
    categorical_cols = ['session', 'trend_regime', 'vol_regime']
    df = pd.get_dummies(df, columns=[c for c in categorical_cols if c in df.columns], drop_first=True)
    
    candle_cols = [col for col in df.columns if col.startswith("CDL")]
    def compress_pattern(v):
        if v >= 80: return 1.0
        elif v > 0: return 0.5
        elif v <= -80: return -1.0
        elif v < 0: return -0.5
        else: return 0.0
    for col in candle_cols:
        df[col] = df[col].fillna(0).apply(compress_pattern)
        
    one_hot_cols = [col for col in df.columns if any(s in col for s in ['session_', 'trend_regime_', 'vol_regime_'])]
    non_scalable_cols = set(candle_cols + one_hot_cols + ['time'])
    numeric_columns_to_scale = [col for col in df.columns if col not in non_scalable_cols and df[col].dtype in ['float64', 'float32', 'int64', 'int32']]
    
    scaler = StandardScaler()
    df[numeric_columns_to_scale] = scaler.fit_transform(df[numeric_columns_to_scale].fillna(0))
    
    return downcast_dtypes(df)

def process_file_in_parallel(file_path_tuple):
    """
    Wrapper function to read, process, and save a single file.
    Designed to be used with a multiprocessing Pool.
    """
    silver_path, gold_path = file_path_tuple
    fname = os.path.basename(silver_path)
    
    try:
        features_df = pd.read_csv(silver_path, parse_dates=['time'])
        gold_dataset = create_gold_features(features_df)
        gold_dataset.to_csv(gold_path, index=False)
        return f"✅ Success! Gold data generated for {fname}."
    except Exception as e:
        return f"❌ FAILED to process {fname}. Error: {e}"

if __name__ == "__main__":
    core_dir = os.path.dirname(os.path.abspath(__file__))
    silver_features_dir = os.path.abspath(os.path.join(core_dir, '..', 'silver_data', 'features'))
    gold_features_dir = os.path.abspath(os.path.join(core_dir, '..', 'gold_data', 'features'))
    os.makedirs(gold_features_dir, exist_ok=True)

    try:
        all_files = [f for f in os.listdir(silver_features_dir) if f.endswith('.csv')]
    except FileNotFoundError:
        print(f"❌ Error: The directory '{silver_features_dir}' was not found.")
        all_files = []

    if not all_files:
        print("❌ No silver feature files found to process.")
    else:
        # Filter out files that have already been processed
        files_to_process = [f for f in all_files if not os.path.exists(os.path.join(gold_features_dir, f))]
        
        if not files_to_process:
            print("✅ All gold feature files already exist. Nothing to do.")
        else:
            print(f"Found {len(files_to_process)} new silver feature file(s) to process.")
            
            # Prepare a list of (input, output) path tuples for the multiprocessing pool
            tasks = [(os.path.join(silver_features_dir, fname), os.path.join(gold_features_dir, fname)) for fname in files_to_process]
            
            # --- SPEED OPTIMIZATION: Use a multiprocessing Pool ---
            num_processes = min(MAX_CPU_USAGE, len(tasks))
            print(f"\n🚀 Starting parallel processing with {num_processes} workers...")
            
            with Pool(processes=num_processes) as pool:
                # Use tqdm to create a progress bar for the parallel tasks
                results = list(tqdm(pool.imap(process_file_in_parallel, tasks), total=len(tasks)))

            print("\n--- Processing Summary ---")
            for res in results:
                print(res)

    print("\n" + "="*50 + "\n✅ All gold data generation complete.")
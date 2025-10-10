# gold_data_generator.py (Parallelized Version)

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

# --- CORE FUNCTIONS ---
def downcast_dtypes(df):
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
    return df

def create_gold_features(features_df):
    df = features_df.copy()
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
    df.drop(columns=list(set(abs_price_cols) | {'volume'}), inplace=True, errors='ignore')
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
        print(f"❌ Error: Directory not found: '{silver_features_dir}'."); all_files = []

    if not all_files:
        print("❌ No silver feature files found to process.")
    else:
        files_to_process = [f for f in all_files if not os.path.exists(os.path.join(gold_features_dir, f))]
        
        if not files_to_process:
            print("✅ All gold feature files already exist. Nothing to do.")
        else:
            print(f"Found {len(files_to_process)} new silver feature file(s) to process.")
            
            # --- User-configurable CPU count ---
            use_multiprocessing = input("Use multiprocessing? (y/n): ").strip().lower() == 'y'
            if use_multiprocessing:
                num_processes = MAX_CPU_USAGE
            else:
                try:
                    num_processes = int(input(f"Enter number of processes to use (1-{cpu_count()}): ").strip())
                    if num_processes < 1 or num_processes > cpu_count(): raise ValueError
                except ValueError:
                    print("Invalid input. Defaulting to 1 process."); num_processes = 1
            
            tasks = [(os.path.join(silver_features_dir, fname), os.path.join(gold_features_dir, fname)) for fname in files_to_process]
            
            effective_num_processes = min(num_processes, len(tasks))
            print(f"\n🚀 Starting parallel processing with {effective_num_processes} workers...")
            
            if effective_num_processes > 1:
                with Pool(processes=effective_num_processes) as pool:
                    results = list(tqdm(pool.imap_unordered(process_file_in_parallel, tasks), total=len(tasks)))
            else:
                results = [process_file_in_parallel(task) for task in tqdm(tasks)]

            print("\n--- Processing Summary ---")
            for res in results:
                print(res)

    print("\n" + "="*50 + "\n✅ All gold data generation complete.")
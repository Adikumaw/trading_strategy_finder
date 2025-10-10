# platinum_target_extractor.py (Corrected for User Experience)

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import gc
import hashlib
from multiprocessing import Pool, cpu_count
from functools import partial

# --- CONFIGURATION ---
MAX_CPU_USAGE = max(1, cpu_count() - 2)

# --- Functions (Unchanged) ---
def filter_chunk_for_definition(chunk, definition):
    temp_df = chunk
    if isinstance(definition['sl_def'], str):
        level, bin_val = definition['sl_def'], definition['sl_bin']; pct_col = f"sl_placement_pct_to_{level}"
        if pct_col in temp_df.columns:
            lb, ub = bin_val / 10.0, (bin_val + 1) / 10.0; temp_df = temp_df[(temp_df[pct_col] >= lb) & (temp_df[pct_col] < ub)]
    else: temp_df = temp_df[np.isclose(temp_df['sl_ratio'], definition['sl_ratio'])]
    if isinstance(definition['tp_def'], str):
        level, bin_val = definition['tp_def'], definition['tp_bin']; pct_col = f"tp_placement_pct_to_{level}"
        if pct_col in temp_df.columns:
            lb, ub = bin_val / 10.0, (bin_val + 1) / 10.0; temp_df = temp_df[(temp_df[pct_col] >= lb) & (temp_df[pct_col] < ub)]
    else: temp_df = temp_df[np.isclose(temp_df['tp_ratio'], definition['tp_ratio'])]
    return temp_df

def process_chunk_for_all_definitions(chunk_path, all_definitions, target_dir, dtype_map):
    chunk_df = pd.read_csv(chunk_path, parse_dates=['entry_time'], dtype=dtype_map)
    chunk_df['outcome'] = 1
    appends = {}
    for _, definition in all_definitions.iterrows():
        slice_df = filter_chunk_for_definition(chunk_df, definition)
        if not slice_df.empty:
            agg_df = slice_df.groupby('entry_time').agg(trade_count=('outcome', 'size')).reset_index()
            key = definition['key']
            if key not in appends: appends[key] = []
            appends[key].append(agg_df)
    for key, dataframes in appends.items():
        target_file = os.path.join(target_dir, f"{key}.csv")
        combined_df = pd.concat(dataframes)
        file_exists = os.path.exists(target_file)
        combined_df.to_csv(target_file, mode='a', header=not file_exists, index=False)
    return chunk_path

if __name__ == "__main__":
    core_dir = os.path.dirname(os.path.abspath(__file__))
    combinations_dir = os.path.abspath(os.path.join(core_dir, '..', 'platinum_data', 'combinations'))
    chunked_outcomes_dir = os.path.abspath(os.path.join(core_dir, '..', 'silver_data', 'chunked_outcomes'))
    targets_dir = os.path.abspath(os.path.join(core_dir, '..', 'platinum_data', 'targets'))
    os.makedirs(targets_dir, exist_ok=True)

    try:
        combination_files = [f for f in os.listdir(combinations_dir) if f.endswith('.csv')]
    except FileNotFoundError:
        print(f"❌ Directory not found: {combinations_dir}"); combination_files = []

    if not combination_files:
        print("❌ No combination files found.")
    else:
        # --- UX FIX: Ask for multiprocessing settings ONCE at the start ---
        print(f"Found {len(combination_files)} instrument(s) to process.")
        use_multiprocessing = input("Use multiprocessing for each instrument? (y/n): ").strip().lower() == 'y'
        if use_multiprocessing:
            num_processes = MAX_CPU_USAGE
        else:
            try:
                num_processes = int(input(f"Enter number of processes to use per instrument (1-{cpu_count()}): ").strip())
                if num_processes < 1 or num_processes > cpu_count(): raise ValueError
            except ValueError:
                print("Invalid input. Defaulting to 1 process."); num_processes = 1

        # --- Main loop now applies the user's choice to every file ---
        for fname in combination_files:
            print(f"\n{'='*25}\nExtracting targets for: {fname}\n{'='*25}")
            instrument_name = fname.replace('.csv', '')
            
            combinations_path = os.path.join(combinations_dir, fname)
            instrument_chunk_dir = os.path.join(chunked_outcomes_dir, instrument_name)
            instrument_target_dir = os.path.join(targets_dir, instrument_name)

            if not os.path.exists(instrument_chunk_dir):
                print(f"❌ Chunks not found for {instrument_name}. Run chunk_maker first."); continue
            
            if os.path.exists(instrument_target_dir) and len(os.listdir(instrument_target_dir)) > 0:
                print(f"✅ Targets already seem to be extracted for {instrument_name}. Skipping."); continue

            os.makedirs(instrument_target_dir, exist_ok=True)
            
            all_definitions = pd.read_csv(combinations_path)
            all_definitions['sl_def'] = all_definitions['sl_def'].astype(object); all_definitions['tp_def'] = all_definitions['tp_def'].astype(object)
            if 'sl_bin' in all_definitions.columns: all_definitions['sl_bin'] = all_definitions['sl_bin'].astype('Int64')
            if 'tp_bin' in all_definitions.columns: all_definitions['tp_bin'] = all_definitions['tp_bin'].astype('Int64')
            key_cols = ['type', 'sl_def', 'sl_bin', 'tp_def', 'tp_bin']
            all_definitions['key'] = all_definitions[key_cols].astype(str).sum(axis=1).apply(lambda x: hashlib.sha256(x.encode()).hexdigest())

            chunk_files = [os.path.join(instrument_chunk_dir, f) for f in os.listdir(instrument_chunk_dir) if f.endswith('.csv')]
            
            print("Analyzing column types for memory-efficient loading...")
            temp_df = pd.read_csv(chunk_files[0], nrows=1)
            dtype_map = {col: 'float32' for col in temp_df.columns if col not in ['entry_time', 'exit_time', 'trade_type', 'outcome']}
            print("✅ Column types analyzed.")
            
            # Use the number of processes chosen by the user at the start
            effective_num_processes = min(num_processes, len(chunk_files))
            print(f"\n🚀 Starting extraction with {effective_num_processes} worker(s)...")
            
            func = partial(process_chunk_for_all_definitions, all_definitions=all_definitions, target_dir=instrument_target_dir, dtype_map=dtype_map)

            if effective_num_processes > 1:
                with Pool(processes=effective_num_processes) as pool:
                    list(tqdm(pool.imap_unordered(func, chunk_files), total=len(chunk_files), desc="Processing Chunks"))
            else:
                for chunk_file in tqdm(chunk_files, desc="Processing Chunks"):
                    func(chunk_file)

            print(f"\n✅ Target extraction complete for {instrument_name}.")

    print("\n" + "="*50 + "\n✅ All target extraction complete.")
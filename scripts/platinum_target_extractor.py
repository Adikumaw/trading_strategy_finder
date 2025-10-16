# platinum_target_extractor.py (Corrected for User Experience)

"""
Platinum Layer - Stage 2: The Target Extractor (The Pre-Processor)

This script is the heavy-lifting, "map-reduce" style pre-processing engine of
the Platinum layer. Its primary purpose is to solve a major I/O bottleneck.

Instead of requiring the final ML script to read a multi-gigabyte data file
thousands of times (once for each strategy blueprint), this script pre-computes
all the necessary data. It iterates through each strategy blueprint and extracts
its corresponding "target variable" — the count of successful trades per candle
(`trade_count`).

The output is thousands of tiny, fast-loading CSV files, one for each unique
strategy blueprint. This massive pre-computation shifts the workload to an
offline, parallelizable task, making the final discovery stage incredibly fast.
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import gc
import hashlib
from multiprocessing import Pool, cpu_count
from functools import partial

# --- CONFIGURATION ---

# Sets the maximum number of CPU cores to use for multiprocessing.
# Leaves 2 cores free to ensure system responsiveness.
MAX_CPU_USAGE = max(1, cpu_count() - 2)

# --- CORE FUNCTIONS ---

def filter_chunk_for_definition(chunk, definition):
    """
    Filters a DataFrame chunk to find all trades that match a specific
    strategy blueprint (definition).

    Args:
        chunk (pd.DataFrame): A chunk of enriched trade data.
        definition (pd.Series): A single row representing one strategy blueprint.

    Returns:
        pd.DataFrame: A filtered DataFrame containing only matching trades.
    """
    temp_df = chunk
    # --- Filter based on Stop-Loss definition ---
    if isinstance(definition['sl_def'], str):
        # Case 1: SL is dynamically defined by a binned distance to a level.
        level, bin_val = definition['sl_def'], definition['sl_bin']
        pct_col = f"sl_placement_pct_to_{level}"
        if pct_col in temp_df.columns:
            # Calculate the bin's lower and upper percentage bounds.
            lb, ub = bin_val / 10.0, (bin_val + 1) / 10.0
            temp_df = temp_df[(temp_df[pct_col] >= lb) & (temp_df[pct_col] < ub)]
    else:
        # Case 2: SL is a fixed percentage ratio. Use np.isclose for float safety.
        temp_df = temp_df[np.isclose(temp_df['sl_ratio'], definition['sl_ratio'])]

    # --- Filter based on Take-Profit definition ---
    if isinstance(definition['tp_def'], str):
        # Case 3: TP is dynamically defined by a binned distance to a level.
        level, bin_val = definition['tp_def'], definition['tp_bin']
        pct_col = f"tp_placement_pct_to_{level}"
        if pct_col in temp_df.columns:
            lb, ub = bin_val / 10.0, (bin_val + 1) / 10.0
            temp_df = temp_df[(temp_df[pct_col] >= lb) & (temp_df[pct_col] < ub)]
    else:
        # Case 4: TP is a fixed percentage ratio.
        temp_df = temp_df[np.isclose(temp_df['tp_ratio'], definition['tp_ratio'])]
        
    return temp_df

def process_chunk_for_all_definitions(chunk_path, all_definitions, target_dir, dtype_map):
    """
    A worker function for multiprocessing. It processes a single data chunk,
    finds matching trades for ALL strategy definitions, aggregates them, and
    appends the results to the appropriate target files.

    Args:
        chunk_path (str): Path to the single chunk CSV file to process.
        all_definitions (pd.DataFrame): The master list of all strategy blueprints.
        target_dir (str): The output directory for the target files.
        dtype_map (dict): A mapping of column names to dtypes for efficient loading.

    Returns:
        str: The path of the chunk that was processed.
    """
    # Load the data chunk with optimized data types to save memory.
    chunk_df = pd.read_csv(chunk_path, parse_dates=['entry_time'], dtype=dtype_map)
    chunk_df['outcome'] = 1 # A dummy column for counting occurrences.
    
    # Use a dictionary to accumulate results in memory before writing to disk.
    appends = {}
    
    # For this one chunk, iterate through every single strategy blueprint.
    for _, definition in all_definitions.iterrows():
        slice_df = filter_chunk_for_definition(chunk_df, definition)
        if not slice_df.empty:
            # Aggregate the results to find the trade count for each entry timestamp.
            # This becomes the 'y' variable for the ML model.
            agg_df = slice_df.groupby('entry_time').agg(trade_count=('outcome', 'size')).reset_index()
            key = definition['key']
            if key not in appends:
                appends[key] = []
            appends[key].append(agg_df)
            
    # After checking all definitions, write the accumulated results to their respective files.
    for key, dataframes in appends.items():
        target_file = os.path.join(target_dir, f"{key}.csv")
        combined_df = pd.concat(dataframes)
        file_exists = os.path.exists(target_file)
        # Append to the target file. The mode='a' is the core of the map-reduce logic.
        combined_df.to_csv(target_file, mode='a', header=not file_exists, index=False)
        
    return chunk_path

if __name__ == "__main__":
    # --- Define Project Directory Structure ---
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

        # --- Main loop: Process each instrument ---
        for fname in combination_files:
            print(f"\n{'='*25}\nExtracting targets for: {fname}\n{'='*25}")
            instrument_name = fname.replace('.csv', '')
            
            combinations_path = os.path.join(combinations_dir, fname)
            instrument_chunk_dir = os.path.join(chunked_outcomes_dir, instrument_name)
            instrument_target_dir = os.path.join(targets_dir, instrument_name)

            if not os.path.exists(instrument_chunk_dir):
                print(f"❌ Chunks not found for {instrument_name}. Run silver_data_generator first."); continue
            
            # --- Resumability Check ---
            if os.path.exists(instrument_target_dir) and len(os.listdir(instrument_target_dir)) > 0:
                print(f"✅ Targets already seem to be extracted for {instrument_name}. Skipping."); continue

            os.makedirs(instrument_target_dir, exist_ok=True)
            
            # --- Prepare Definitions ---
            all_definitions = pd.read_csv(combinations_path)
            # Ensure correct dtypes for filtering logic
            all_definitions['sl_def'] = all_definitions['sl_def'].astype(object)
            all_definitions['tp_def'] = all_definitions['tp_def'].astype(object)
            if 'sl_bin' in all_definitions.columns: all_definitions['sl_bin'] = all_definitions['sl_bin'].astype('Int64')
            if 'tp_bin' in all_definitions.columns: all_definitions['tp_bin'] = all_definitions['tp_bin'].astype('Int64')
            
            # Create a unique hash key for each definition. This key becomes the filename.
            key_cols = ['type', 'sl_def', 'sl_bin', 'tp_def', 'tp_bin']
            all_definitions['key'] = all_definitions[key_cols].astype(str).sum(axis=1).apply(lambda x: hashlib.sha256(x.encode()).hexdigest())

            chunk_files = [os.path.join(instrument_chunk_dir, f) for f in os.listdir(instrument_chunk_dir) if f.endswith('.csv')]
            
            # --- Memory Optimization: Analyze dtypes before loading all chunks ---
            print("Analyzing column types for memory-efficient loading...")
            temp_df = pd.read_csv(chunk_files[0], nrows=1)
            dtype_map = {col: 'float32' for col in temp_df.columns if col not in ['entry_time', 'exit_time', 'trade_type', 'outcome']}
            print("✅ Column types analyzed.")
            
            # --- Execute Processing in Parallel ---
            effective_num_processes = min(num_processes, len(chunk_files))
            print(f"\n🚀 Starting extraction with {effective_num_processes} worker(s)...")
            
            # `partial` pre-fills the worker function with arguments that are the
            # same for every chunk, which is a clean way to use multiprocessing.
            func = partial(process_chunk_for_all_definitions, all_definitions=all_definitions, target_dir=instrument_target_dir, dtype_map=dtype_map)

            if effective_num_processes > 1:
                with Pool(processes=effective_num_processes) as pool:
                    # `imap_unordered` is memory-efficient and `tqdm` provides a progress bar.
                    list(tqdm(pool.imap_unordered(func, chunk_files), total=len(chunk_files), desc="Processing Chunks"))
            else:
                # Run sequentially if only one process is requested.
                for chunk_file in tqdm(chunk_files, desc="Processing Chunks"):
                    func(chunk_file)

            print(f"\n✅ Target extraction complete for {instrument_name}.")

    print("\n" + "="*50 + "\n✅ All target extraction complete.")
# platinum_target_extractor.py (Upgraded with Key Persistence)

"""
Platinum Layer - Stage 2: The Target Extractor (The Pre-Processor)

This script is the heavy-lifting, "map-reduce" style pre-processing engine of
the Platinum layer. Its primary purpose is to solve a major I/O bottleneck by
pre-computing all necessary data for the final discovery stage.

The script operates in two main phases for each instrument:
1.  Key Generation: It first loads the master list of strategy blueprints from
    the `combinations` file. For each blueprint, it generates a unique hash key
    (e.g., 'a1b2c3d4e5f6g7h8'). This key will serve as the filename for that
    blueprint's target data.

2.  Target Extraction: Using multiple CPU cores, it reads the Silver layer's
    data chunks ONCE. For each chunk, it finds all trades that match every single
    blueprint and calculates the `trade_count` per candle. These results are
    appended to the corresponding key's file.

After processing, it saves the blueprint-to-key mapping back to the original
`combinations` file. This creates a "single source of truth," making it simple
for the next script to link a specific strategy to its pre-computed target data,
enabling incredibly fast final analysis.
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import gc
import hashlib
from multiprocessing import Pool, cpu_count
from functools import partial
import sys # <-- IMPORT SYS MODULE

# --- CONFIGURATION ---

# Sets the maximum number of CPU cores to use for multiprocessing.
# Leaves 2 cores free to ensure system responsiveness.
MAX_CPU_USAGE = max(1, cpu_count() - 2)

# --- CORE FUNCTIONS ---

def filter_chunk_for_definition(chunk, definition):
    """
    Filters a data chunk to isolate trades matching a single strategy blueprint.

    This function acts as a dynamic filter, applying a series of conditions
    to a DataFrame based on the provided strategy definition. It correctly
    handles both "dynamic" rules (where SL/TP are defined by a binned distance
    to a market level) and "static" rules (where SL/TP are fixed percentage ratios).
    The use of `np.isclose` ensures safe comparison of floating-point numbers.

    Args:
        chunk (pd.DataFrame): A chunk of enriched trade data from the Silver layer.
        definition (pd.Series): A single row from the combinations DataFrame,
                                representing one unique strategy blueprint.

    Returns:
        pd.DataFrame: A new DataFrame containing only the rows from the original
                      chunk that perfectly match the strategy definition.
    """
    # Start with the full chunk and progressively filter it down.
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
    Processes one data chunk against all strategy definitions for parallel execution.

    This is the core worker function for the multiprocessing pool. It loads a
    single data chunk, then iterates through every strategy blueprint defined in
    the `all_definitions` DataFrame. For each blueprint, it filters the chunk
    to find matching trades, aggregates them to count trades per timestamp, and
    appends these counts to the correct target file on disk. This "map-reduce"
    style approach avoids reading the source chunks multiple times.

    Args:
        chunk_path (str): The full path to the single chunk CSV file to be processed.
        all_definitions (pd.DataFrame): The master DataFrame of all strategy
                                        blueprints, which MUST include the 'key' column.
        target_dir (str): The path to the output directory where target files
                          (named by key) will be saved.
        dtype_map (dict): A mapping of column names to data types, used for
                          loading the chunk CSV efficiently.

    Returns:
        str: The path of the chunk that was successfully processed, used for
             progress tracking.
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
            # This becomes the 'y' variable (target) for the ML model.
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
    """
    Main execution block.
    
    This script can be run in two modes:
    1. Discovery Mode (no arguments): Scans for all new combination files and processes them.
       Example: `python scripts/platinum_target_extractor.py`
       
    2. Targeted Mode (one argument): Processes only the single file specified.
       Example: `python scripts/platinum_target_extractor.py XAUUSD15.csv`
    """
    # --- Define Project Directory Structure ---
    core_dir = os.path.dirname(os.path.abspath(__file__))
    combinations_dir = os.path.abspath(os.path.join(core_dir, '..', 'platinum_data', 'combinations'))
    chunked_outcomes_dir = os.path.abspath(os.path.join(core_dir, '..', 'silver_data', 'chunked_outcomes'))
    targets_dir = os.path.abspath(os.path.join(core_dir, '..', 'platinum_data', 'targets'))
    os.makedirs(targets_dir, exist_ok=True)

    # --- NEW: DUAL-MODE FILE DISCOVERY LOGIC ---
    target_file_arg = sys.argv[1] if len(sys.argv) > 1 else None

    if target_file_arg:
        # --- Targeted Mode ---
        print(f"🎯 Targeted Mode: Processing single file '{target_file_arg}'")
        combinations_path_check = os.path.join(combinations_dir, target_file_arg)
        if not os.path.exists(combinations_path_check):
            print(f"❌ Error: Target file not found in platinum_data/combinations: {target_file_arg}")
            files_to_process = []
        else:
            files_to_process = [target_file_arg]
    else:
        # --- Discovery Mode (Default) ---
        print("🔍 Discovery Mode: Scanning for all new files...")
        try:
            combination_files = [f for f in os.listdir(combinations_dir) if f.endswith('.csv')]
            # Check which files already have the 'key' column added, indicating they are processed.
            files_to_process = []
            for f in combination_files:
                try:
                    df = pd.read_csv(os.path.join(combinations_dir, f), nrows=0)
                    if 'key' not in df.columns:
                        files_to_process.append(f)
                except Exception:
                    files_to_process.append(f) # Process if file is empty or unreadable
        except FileNotFoundError:
            print(f"❌ Directory not found: {combinations_dir}"); files_to_process = []

    if not files_to_process:
        print("ℹ️ No new combination files found to process.")
    else:
        print(f"Found {len(files_to_process)} instrument(s) to process.")
        
        # --- Configure Multiprocessing ---
        if target_file_arg or len(files_to_process) == 1:
            use_multiprocessing = True
        else:
            use_multiprocessing = input("Use multiprocessing for each instrument? (y/n): ").strip().lower() == 'y'
        
        if use_multiprocessing:
            num_processes = MAX_CPU_USAGE
        else:
            num_processes = 1

        # --- Main loop: Process each instrument ---
        for fname in files_to_process:
            print(f"\n{'='*25}\nExtracting targets for: {fname}\n{'='*25}")
            instrument_name = fname.replace('.csv', '')
            
            combinations_path = os.path.join(combinations_dir, fname)
            instrument_chunk_dir = os.path.join(chunked_outcomes_dir, instrument_name)
            instrument_target_dir = os.path.join(targets_dir, instrument_name)

            if not os.path.exists(instrument_chunk_dir):
                print(f"❌ Chunks not found for {instrument_name}. Run silver_data_generator first."); continue
            
            # --- Prepare Definitions and Keys ---
            all_definitions = pd.read_csv(combinations_path)
            
            # This check is now inside the loop, specific to each file.
            if 'key' in all_definitions.columns:
                print(f"✅ Keys already exist in {fname}. Targets presumed extracted. Skipping.")
                continue

            os.makedirs(instrument_target_dir, exist_ok=True)
            
            # Ensure correct dtypes for filtering logic and hashing
            all_definitions['sl_def'] = all_definitions['sl_def'].astype(object)
            all_definitions['tp_def'] = all_definitions['tp_def'].astype(object)
            if 'sl_bin' in all_definitions.columns: all_definitions['sl_bin'] = all_definitions['sl_bin'].astype('Int64')
            if 'tp_bin' in all_definitions.columns: all_definitions['tp_bin'] = all_definitions['tp_bin'].astype('Int64')
            
            # --- Generate the Unique Key for Each Blueprint ---
            # This key will be used as the filename for the target CSV.
            # Using a separator and a consistent NaN representation makes the hash robust.
            key_cols = ['type', 'sl_def', 'sl_bin', 'tp_def', 'tp_bin']
            def create_hashable_string(row):
                return '-'.join([str(row[c]) for c in key_cols])
                
            all_definitions['key'] = all_definitions.apply(create_hashable_string, axis=1)\
                                                    .apply(lambda x: hashlib.sha256(x.encode()).hexdigest()[:16])

            chunk_files = [os.path.join(instrument_chunk_dir, f) for f in os.listdir(instrument_chunk_dir) if f.endswith('.csv')]
            if not chunk_files:
                print(f"⚠️ No chunk files found in {instrument_chunk_dir} to process. Skipping target extraction.")
                continue
            
            # --- Memory Optimization: Analyze dtypes ---
            print("Analyzing column types for memory-efficient loading...")
            temp_df = pd.read_csv(chunk_files[0], nrows=1)
            dtype_map = {col: 'float32' for col in temp_df.columns if col not in ['entry_time', 'exit_time', 'trade_type', 'outcome']}
            print("✅ Column types analyzed.")
            
            # --- Execute Processing in Parallel ---
            effective_num_processes = min(num_processes, len(chunk_files))
            print(f"\n🚀 Starting extraction with {effective_num_processes} worker(s)...")
            
            func = partial(process_chunk_for_all_definitions, all_definitions=all_definitions, target_dir=instrument_target_dir, dtype_map=dtype_map)

            if effective_num_processes > 1:
                with Pool(processes=effective_num_processes) as pool:
                    list(tqdm(pool.imap_unordered(func, chunk_files), total=len(chunk_files), desc="Processing Chunks"))
            else:
                for chunk_file in tqdm(chunk_files, desc="Processing Chunks"):
                    func(chunk_file)
            
            # --- Save the Keys Back to the Combinations File ---
            # This is the crucial step that creates the "single source of truth".
            # The next script will read this file and know exactly which key
            # belongs to which strategy blueprint.
            try:
                all_definitions.to_csv(combinations_path, index=False)
                print(f"\n✅ Target extraction complete and keys saved back to {fname}.")
            except Exception as e:
                print(f"\n❌ FAILED to save keys back to {fname}. Error: {e}")

    print("\n" + "="*50 + "\n✅ All target extraction complete.")
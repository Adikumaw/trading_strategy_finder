# platinum_target_extractor.py (V3.1 - DType Bug Fix)

"""
Platinum Layer - Stage 2: The Target Extractor (The Pre-Processor)

This script is the heavy-lifting, "map-reduce" style pre-processing engine of
the Platinum layer. Its primary purpose is to solve a major I/O bottleneck by
pre-computing all necessary data for the final discovery stage.

This version uses a more sophisticated parallel model. It processes the large
data chunks serially, one at a time. For each chunk, it creates a pool of workers
that process the 50,000+ strategy definitions in parallel. This architecture is
highly memory-efficient, as the large data chunk is only held in each worker's
memory once, while the tasks they receive (batches of definitions) are very small.
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import gc
import hashlib
from multiprocessing import Pool, cpu_count
import sys
import traceback
from typing import Dict, List, Tuple

# --- CONFIGURATION ---
MAX_CPU_USAGE = max(1, cpu_count() - 2)
BPS_BIN_SIZE: float = 5.0

# --- WORKER-SPECIFIC GLOBAL VARIABLES ---
worker_chunk_df: pd.DataFrame

def init_worker(chunk_df_for_worker: pd.DataFrame):
    """
    Initializer for each worker process in the multiprocessing Pool.
    """
    global worker_chunk_df
    worker_chunk_df = chunk_df_for_worker

# --- CORE WORKER FUNCTIONS ---

def filter_chunk_for_definition(chunk: pd.DataFrame, definition: pd.Series) -> pd.DataFrame:
    """
    Filters a data chunk to isolate trades matching a single strategy blueprint.
    """
    strat_type = definition['type']
    valid_indices = chunk.index.copy()

    if strat_type == 'SL-Pct':
        level, sl_bin_val = definition['sl_def'], definition['sl_bin']
        pct_col = f"sl_place_pct_to_{level}"
        if pct_col not in chunk.columns: return pd.DataFrame()
        lb, ub = sl_bin_val / 10.0, (sl_bin_val + 1) / 10.0
        pct_mask = (chunk[pct_col] >= lb) & (chunk[pct_col] < ub)
        # --- MODIFIED: Explicitly cast definition to float to prevent DType error ---
        ratio_mask = np.isclose(chunk['tp_ratio'], float(definition['tp_def']))
        valid_indices = valid_indices[pct_mask & ratio_mask]

    elif strat_type == 'TP-Pct':
        level, tp_bin_val = definition['tp_def'], definition['tp_bin']
        pct_col = f"tp_place_pct_to_{level}"
        if pct_col not in chunk.columns: return pd.DataFrame()
        lb, ub = tp_bin_val / 10.0, (tp_bin_val + 1) / 10.0
        pct_mask = (chunk[pct_col] >= lb) & (chunk[pct_col] < ub)
        # --- MODIFIED: Explicitly cast definition to float to prevent DType error ---
        ratio_mask = np.isclose(chunk['sl_ratio'], float(definition['sl_def']))
        valid_indices = valid_indices[pct_mask & ratio_mask]

    elif strat_type == 'SL-BPS':
        level, sl_bin_val = definition['sl_def'], definition['sl_bin']
        bps_col = f"sl_dist_to_{level}_bps"
        if bps_col not in chunk.columns: return pd.DataFrame()
        lb, ub = sl_bin_val, sl_bin_val + BPS_BIN_SIZE
        bps_mask = (chunk[bps_col] >= lb) & (chunk[bps_col] < ub)
        # --- MODIFIED: Explicitly cast definition to float to prevent DType error ---
        ratio_mask = np.isclose(chunk['tp_ratio'], float(definition['tp_def']))
        valid_indices = valid_indices[bps_mask & ratio_mask]

    elif strat_type == 'TP-BPS':
        level, tp_bin_val = definition['tp_def'], definition['tp_bin']
        bps_col = f"tp_dist_to_{level}_bps"
        if bps_col not in chunk.columns: return pd.DataFrame()
        lb, ub = tp_bin_val, tp_bin_val + BPS_BIN_SIZE
        bps_mask = (chunk[bps_col] >= lb) & (chunk[bps_col] < ub)
        # --- MODIFIED: Explicitly cast definition to float to prevent DType error ---
        ratio_mask = np.isclose(chunk['sl_ratio'], float(definition['sl_def']))
        valid_indices = valid_indices[bps_mask & ratio_mask]
    else:
        return pd.DataFrame()

    return chunk.loc[valid_indices]


def process_definition_batch(definitions_batch: pd.DataFrame) -> Dict[str, List[pd.DataFrame]]:
    """
    Processes a small batch of definitions against the globally available data chunk.
    """
    global worker_chunk_df
    
    appends = {}
    for _, definition in definitions_batch.iterrows():
        slice_df = filter_chunk_for_definition(worker_chunk_df, definition)
        if not slice_df.empty:
            agg_df = slice_df.groupby('entry_time').agg(trade_count=('outcome', 'size')).reset_index()
            key = definition['key']
            if key not in appends:
                appends[key] = []
            appends[key].append(agg_df)
            
    return appends


# --- DEDICATED INSTRUMENT PROCESSING FUNCTION ---

def extract_targets_for_instrument(instrument_name, combinations_dir, chunked_outcomes_dir, targets_dir):
    """
    Orchestrates the entire target extraction process for a single instrument
    using definition-level parallelism.
    """
    combinations_path = os.path.join(combinations_dir, f"{instrument_name}.csv")
    instrument_chunk_dir = os.path.join(chunked_outcomes_dir, instrument_name)
    instrument_target_dir = os.path.join(targets_dir, instrument_name)

    if not os.path.isdir(instrument_chunk_dir):
        print(f"[ERROR] Chunks not found for {instrument_name}. Skipping.")
        return
    
    all_definitions = pd.read_csv(combinations_path)
    if all_definitions.empty or 'key' in all_definitions.columns:
        print(f"[INFO] File for {instrument_name} is empty or already processed. Skipping.")
        return

    os.makedirs(instrument_target_dir, exist_ok=True)
    
    print("Generating unique keys for each strategy blueprint...")
    key_cols = ['type', 'sl_def', 'sl_bin', 'tp_def', 'tp_bin']
    def create_hashable_string(row):
        return '-'.join([str(row[c]) if pd.notna(row[c]) else 'nan' for c in key_cols])
    all_definitions['key'] = all_definitions.apply(create_hashable_string, axis=1)\
                                            .apply(lambda x: hashlib.sha256(x.encode()).hexdigest()[:16])
    print(f"Generated {len(all_definitions['key'].unique())} unique keys for {len(all_definitions)} definitions.")

    chunk_files = [os.path.join(instrument_chunk_dir, f) for f in os.listdir(instrument_chunk_dir) if f.endswith('.csv')]
    if not chunk_files:
        print(f"[WARNING] No chunks found in {instrument_chunk_dir}. Skipping.")
        return
    
    for chunk_path in tqdm(chunk_files, desc=f"Processing Chunks for {instrument_name}"):
        try:
            temp_df = pd.read_csv(chunk_path, nrows=1)
            dtype_map = {col: 'float32' for col in temp_df.columns if 'ratio' in col or 'pct' in col or 'bps' in col}
            
            chunk_df = pd.read_csv(chunk_path, parse_dates=['entry_time'], dtype=dtype_map)
            if chunk_df.empty:
                continue
            chunk_df['outcome'] = 1
            
            # --- MODIFIED: Use Pandas-native slicing to create batches ---
            # This avoids the FutureWarning from using np.array_split on a DataFrame.
            num_definitions = len(all_definitions)
            num_batches = min(num_definitions, MAX_CPU_USAGE * 4) # Avoid creating empty batches
            if num_batches == 0: continue
            
            # Generate integer indices to split the DataFrame into roughly equal parts.
            split_indices = np.linspace(0, num_definitions, num_batches + 1, dtype=int)
            
            # Create a list of DataFrame slices (batches) using the indices.
            definition_batches = [
                all_definitions.iloc[split_indices[i]:split_indices[i+1]]
                for i in range(num_batches)
            ]
            
            with Pool(processes=MAX_CPU_USAGE, initializer=init_worker, initargs=(chunk_df,)) as pool:
                results = pool.map(process_definition_batch, definition_batches)

            final_appends = {}
            for single_worker_result in results:
                for key, dataframes in single_worker_result.items():
                    if key not in final_appends:
                        final_appends[key] = []
                    final_appends[key].extend(dataframes)
            
            for key, dataframes in final_appends.items():
                target_file = os.path.join(instrument_target_dir, f"{key}.csv")
                combined_df = pd.concat(dataframes)
                file_exists = os.path.exists(target_file)
                combined_df.to_csv(target_file, mode='a', header=not file_exists, index=False)
            
            del chunk_df, results, final_appends, definition_batches
            gc.collect()

        except Exception as e:
            print(f"\n[ERROR] Failed to process chunk {os.path.basename(chunk_path)}. Error: {e}")
            traceback.print_exc()

    try:
        all_definitions.to_csv(combinations_path, index=False)
        print(f"\n[SUCCESS] Target extraction complete. Keys saved to {instrument_name}.csv.")
    except Exception as e:
        print(f"\n[ERROR] FAILED to save keys back to {instrument_name}.csv. Error: {e}")


# --- STANDARDIZED MAIN ORCHESTRATOR ---

def main():
    """
    Main execution block: handles file discovery and orchestrates the serial
    processing of each instrument by calling the dedicated processing function.
    """
    core_dir = os.path.dirname(os.path.abspath(__file__))
    combinations_dir = os.path.abspath(os.path.join(core_dir, '..', 'platinum_data', 'combinations'))
    chunked_outcomes_dir = os.path.abspath(os.path.join(core_dir, '..', 'silver_data', 'chunked_outcomes'))
    targets_dir = os.path.abspath(os.path.join(core_dir, '..', 'platinum_data', 'targets'))
    os.makedirs(targets_dir, exist_ok=True)

    files_to_process = []
    target_file_arg = sys.argv[1] if len(sys.argv) > 1 else None

    if target_file_arg:
        print(f"[TARGET] Targeted Mode: Processing '{target_file_arg}'")
        path_check = os.path.join(combinations_dir, target_file_arg)
        if not os.path.exists(path_check):
            print(f"[ERROR] Target combination file not found: {path_check}")
        else:
            files_to_process = [target_file_arg]
    else:
        print("[SCAN] Interactive Mode: Scanning for new combination files...")
        try:
            all_combo_files = sorted([f for f in os.listdir(combinations_dir) if f.endswith('.csv')])
            new_files = []
            for f in all_combo_files:
                try:
                    df_header = pd.read_csv(os.path.join(combinations_dir, f), nrows=0)
                    if 'key' not in df_header.columns:
                        new_files.append(f)
                except (pd.errors.EmptyDataError, IndexError):
                    if os.path.getsize(os.path.join(combinations_dir, f)) > 50:
                        new_files.append(f)

            if not new_files:
                print("[INFO] No new combination files to process.")
            else:
                print("\n--- Select File(s) to Process ---")
                for i, f in enumerate(new_files): print(f"  [{i+1}] {f}")
                print("\nSelect multiple files with comma-separated numbers (e.g., 1,3,5)")
                user_input = input("Enter number(s) to process: ").strip()
                if user_input:
                    try:
                        selected_indices = [int(i.strip()) - 1 for i in user_input.split(',')]
                        valid_indices = sorted(set(idx for idx in selected_indices if 0 <= idx < len(new_files)))
                        files_to_process = [new_files[idx] for idx in valid_indices]
                    except ValueError:
                        print("[ERROR] Invalid input.")
        except FileNotFoundError:
            print(f"[ERROR] The combinations directory was not found at: {combinations_dir}")

    if not files_to_process:
        print("[INFO] No files selected or found for processing.")
        return

    print(f"\n[QUEUE] Queued {len(files_to_process)} file(s) for processing: {files_to_process}")
    for fname in files_to_process:
        instrument_name = fname.replace('.csv', '')
        try:
            print(f"\n{'='*25}\nExtracting targets for: {instrument_name}\n{'='*25}")
            extract_targets_for_instrument(
                instrument_name, combinations_dir, chunked_outcomes_dir, targets_dir
            )
        except Exception:
            print(f"\n[FATAL ERROR] A critical error occurred while processing {instrument_name}.")
            traceback.print_exc()

    print("\n" + "="*50 + "\n[COMPLETE] All target extraction tasks are finished.")

if __name__ == "__main__":
    main()
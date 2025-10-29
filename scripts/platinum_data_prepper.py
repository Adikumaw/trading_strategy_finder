# platinum_unifier.py (V1.0 - Unified Discovery & Extraction)

"""
Platinum Layer - Pre Processor: The Architect & Extractor

This single, high-performance script replaces the two previous Platinum stages
(combinations_generator and target_extractor). It embodies a "Live Hashing and
Streaming" architecture to maximize performance and minimize I/O.

The script operates in two distinct, sequential phases:

Phase 1: Parallel Discovery and Streaming (The "Mapper")
- It reads the enriched Silver data chunks in parallel.
- For each chunk, a worker discovers all unique strategy blueprints and
  simultaneously aggregates the number of trades per candle for each blueprint.
- These results (blueprint -> trade counts) are returned to the main process.
- The main process receives these results in a stream. For each new, unseen
  blueprint, it generates a unique hash key and immediately streams the
  aggregated trade counts to a temporary file on disk ({key}.partX.csv).
- This achieves the primary goal: reading the expensive source data ONLY ONCE.

Phase 2: Parallel Consolidation (The "Reducer")
- After all chunks have been processed, the script has a master list of all
  discovered blueprints and thousands of small, temporary target files.
- It starts a second parallel process. Each worker is assigned a single hash key.
- The worker's job is to find all temporary files for that key, load them,
  perform a final aggregation (groupby.sum()), and write the clean, final
  target file to the `platinum_data/targets` directory.

This architecture is extremely memory-efficient and scalable, as it never holds
the full result set in memory, using the filesystem as an intelligent buffer.
"""

import os
import re
import sys
import shutil
import hashlib
import traceback
from multiprocessing import Pool, cpu_count
from collections import defaultdict
from typing import Dict, List, Tuple
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm

# --- CONFIGURATION ---
# Sets the maximum number of CPU cores to use. Leaving 2 cores free ensures
# the system remains responsive for other tasks.
MAX_CPU_USAGE: int = max(1, cpu_count() - 2)

# The size of each bin for basis points. e.g., 5.0 means bins will be
# [-5.0, 0.0, 5.0, 10.0], etc. This controls the granularity of the discovery.
BPS_BIN_SIZE: float = 5.0

# --- PHASE 1: WORKER FUNCTION (MAPPER) ---

def discover_and_aggregate_chunk(task_tuple: Tuple[str, List[str], int]) -> Tuple[Dict, int]:
    """
    Worker for Phase 1. Reads a chunk, discovers all blueprints, aggregates their
    trade counts by timestamp, and returns the results.
    """
    chunk_path, all_levels, chunk_id = task_tuple
    
    # The primary data structure for this chunk's results.
    # Maps blueprint_tuple -> {timestamp -> trade_count}
    chunk_results = defaultdict(lambda: defaultdict(int))

    try:
        chunk = pd.read_csv(chunk_path, parse_dates=['entry_time'])
        if chunk.empty:
            return {}, chunk_id
        
        chunk['sl_ratio'] = chunk['sl_ratio'].round(5)
        chunk['tp_ratio'] = chunk['tp_ratio'].round(5)
        
        # --- Pre-Binning Optimization ---
        # Create all binned columns once in a fast, vectorized step.
        binned_cols = {}
        for level in all_levels:
            for sltp in ['sl', 'tp']:
                pct_col = f"{sltp}_place_pct_to_{level}"
                bps_col = f"{sltp}_dist_to_{level}_bps"
                if pct_col in chunk.columns:
                    binned_cols[f'{pct_col}_bin'] = np.floor(chunk[pct_col] * 10).astype('Int64')
                if bps_col in chunk.columns:
                    binned_cols[f'{bps_col}_bin'] = (np.floor(chunk[bps_col] / BPS_BIN_SIZE) * BPS_BIN_SIZE).astype('Int64')
        
        binned_df = pd.DataFrame(binned_cols)
        chunk = pd.concat([chunk[['entry_time', 'sl_ratio', 'tp_ratio']], binned_df], axis=1)
        
        # --- Iterate through rows to discover and aggregate ---
        for row in chunk.itertuples(index=False):
            entry_time = row.entry_time
            
            for level in all_levels:
                # Check SL-Pct type
                sl_pct_bin_col = f'sl_place_pct_to_{level}_bin'
                if sl_pct_bin_col in row._fields:
                    sl_bin = getattr(row, sl_pct_bin_col)
                    if pd.notna(sl_bin) and -20 <= sl_bin <= 20:
                        blueprint = ('SL-Pct', level, sl_bin, 'ratio', row.tp_ratio)
                        chunk_results[blueprint][entry_time] += 1
                
                # Check TP-Pct type
                tp_pct_bin_col = f'tp_place_pct_to_{level}_bin'
                if tp_pct_bin_col in row._fields:
                    tp_bin = getattr(row, tp_pct_bin_col)
                    if pd.notna(tp_bin) and -20 <= tp_bin <= 20:
                        blueprint = ('TP-Pct', 'ratio', row.sl_ratio, level, tp_bin)
                        chunk_results[blueprint][entry_time] += 1
                
                # Check SL-BPS type
                sl_bps_bin_col = f'sl_dist_to_{level}_bps_bin'
                if sl_bps_bin_col in row._fields:
                    sl_bin = getattr(row, sl_bps_bin_col)
                    if pd.notna(sl_bin) and -50 <= sl_bin <= 50:
                        blueprint = ('SL-BPS', level, sl_bin, 'ratio', row.tp_ratio)
                        chunk_results[blueprint][entry_time] += 1

                # Check TP-BPS type
                tp_bps_bin_col = f'tp_dist_to_{level}_bps_bin'
                if tp_bps_bin_col in row._fields:
                    tp_bin = getattr(row, tp_bps_bin_col)
                    if pd.notna(tp_bin) and -50 <= tp_bin <= 50:
                        blueprint = ('TP-BPS', 'ratio', row.sl_ratio, level, tp_bin)
                        chunk_results[blueprint][entry_time] += 1

    except Exception:
        print(f"[WORKER WARNING] Could not process chunk {os.path.basename(chunk_path)}.")
        traceback.print_exc()

    return chunk_results, chunk_id

# --- PHASE 2: WORKER FUNCTION (REDUCER) ---

def consolidate_target_files(key, temp_dir, final_dir):
    """
    Worker for Phase 2. Finds all temporary part-files for a single key,
    loads them, consolidates them, and writes the final clean target file.
    """
    try:
        part_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.startswith(key)]
        if not part_files:
            return
        
        df = pd.concat([pd.read_csv(f) for f in part_files], ignore_index=True)
        
        # The final aggregation step
        final_df = df.groupby('entry_time')['trade_count'].sum().reset_index()
        
        final_target_path = os.path.join(final_dir, f"{key}.csv")
        final_df.to_csv(final_target_path, index=False)
        
        # Clean up the temporary files for this key
        for f in part_files:
            os.remove(f)
    except Exception:
        print(f"[CONSOLIDATOR WARNING] Failed to consolidate key {key}.")
        traceback.print_exc()

# --- MAIN ORCHESTRATOR ---

def run_unified_processor(instrument_name, base_dirs):
    """
    Orchestrates the entire unified discovery and extraction process.
    """
    # Unpack directories
    chunked_outcomes_dir = os.path.join(base_dirs['silver'], instrument_name)
    combinations_path = os.path.join(base_dirs['platinum_combo'], f"{instrument_name}.csv")
    temp_targets_dir = os.path.join(base_dirs['platinum_temp'], instrument_name)
    final_targets_dir = os.path.join(base_dirs['platinum_final'], instrument_name)
    
    # --- Setup and Cleanup ---
    if os.path.exists(combinations_path):
        print(f"[INFO] {instrument_name} already has a combinations file. Skipping.")
        return
    
    for d in [temp_targets_dir, final_targets_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    chunk_files = [os.path.join(chunked_outcomes_dir, f) for f in os.listdir(chunked_outcomes_dir) if f.endswith('.csv')]
    if not chunk_files:
        print(f"[ERROR] No chunk files found for {instrument_name}.")
        return

    # --- Robust Level Discovery ---
    # Read the header of the first chunk to dynamically discover all possible market levels.
    header_df = pd.read_csv(chunk_files[0], nrows=0)
    # This regex captures the 'level' part from columns like 'sl_place_pct_SMA_50'
    # or 'tp_dist_to_BB_upper_20_bps', making the discovery very robust.
    level_pattern = re.compile(r'(?:sl|tp)_(?:place_pct_to|dist_to)_([a-zA-Z0-9_]+)(?:_bps)?')
    all_levels = sorted(list(set(
        match.group(1) for col in header_df.columns for match in [level_pattern.match(col)] if match
    )))
    print(f"Discovered {len(all_levels)} levels for {instrument_name}.")

    # --- PHASE 1: DISCOVERY & STREAMING ---
    print("\n--- Phase 1: Discovering Blueprints and Streaming Targets ---")
    tasks = [(path, all_levels, i) for i, path in enumerate(chunk_files)]
    master_blueprint_keys = {} # Our live cache: blueprint_tuple -> key

    with Pool(processes=MAX_CPU_USAGE) as pool:
        for chunk_results, chunk_id in tqdm(pool.imap_unordered(discover_and_aggregate_chunk, tasks), total=len(tasks), desc="Processing Chunks"):
            for blueprint, counts in chunk_results.items():
                if blueprint not in master_blueprint_keys:
                    # New discovery: generate and store key
                    key_str = '-'.join([str(x) for x in blueprint])
                    key = hashlib.sha256(key_str.encode()).hexdigest()[:16]
                    master_blueprint_keys[blueprint] = key
                
                key = master_blueprint_keys[blueprint]
                
                # Stream to temporary file
                temp_file_path = os.path.join(temp_targets_dir, f"{key}.part{chunk_id}.csv")
                pd.DataFrame(list(counts.items()), columns=['entry_time', 'trade_count']).to_csv(temp_file_path, index=False)
    
    print(f"\nPhase 1 Complete. Discovered {len(master_blueprint_keys)} unique blueprints.")

    # --- Save Final Combinations File ---
    definitions = []
    for blueprint, key in master_blueprint_keys.items():
        type, sl_def, sl_bin, tp_def, tp_bin = blueprint
        definitions.append({
            'key': key, 'type': type, 
            'sl_def': sl_def, 'sl_bin': sl_bin, 
            'tp_def': tp_def, 'tp_bin': tp_bin
        })
    pd.DataFrame(definitions).to_csv(combinations_path, index=False)
    print(f"Saved final combinations file to {combinations_path}")

    # --- PHASE 2: CONSOLIDATION ---
    print("\n--- Phase 2: Consolidating Temporary Target Files ---")
    all_keys = list(master_blueprint_keys.values())
    
    consolidation_func = partial(consolidate_target_files, temp_dir=temp_targets_dir, final_dir=final_targets_dir)
    
    with Pool(processes=MAX_CPU_USAGE) as pool:
        list(tqdm(pool.imap_unordered(consolidation_func, all_keys), total=len(all_keys), desc="Consolidating Targets"))
        
    print("Phase 2 Complete. Final targets saved.")
    
    # Final cleanup of the temporary directory
    shutil.rmtree(temp_targets_dir)


def main():
    # Standardized main function for orchestration
    core_dir = os.path.dirname(os.path.abspath(__file__))
    base_dirs = {
        'silver': os.path.abspath(os.path.join(core_dir, '..', 'silver_data', 'chunked_outcomes')),
        'platinum_combo': os.path.abspath(os.path.join(core_dir, '..', 'platinum_data', 'combinations')),
        'platinum_temp': os.path.abspath(os.path.join(core_dir, '..', 'platinum_data', 'temp_targets')),
        'platinum_final': os.path.abspath(os.path.join(core_dir, '..', 'platinum_data', 'targets'))
    }
    
    for d in base_dirs.values():
        os.makedirs(d, exist_ok=True)
    
    # --- File Discovery ---
    instrument_folders_to_process = []
    target_arg = sys.argv[1] if len(sys.argv) > 1 else None
    
    if target_arg:
        instrument_name = target_arg.replace('.csv', '')
        print(f"[TARGET] Targeted Mode: Processing instrument '{instrument_name}'")
        if os.path.isdir(os.path.join(base_dirs['silver'], instrument_name)):
            instrument_folders_to_process = [instrument_name]
        else:
            print(f"[ERROR] Silver chunk directory not found for: {instrument_name}")
    else:
        print("[SCAN] Interactive Mode: Scanning for new instrument folders...")
        try:
            all_folders = sorted([d for d in os.listdir(base_dirs['silver']) if os.path.isdir(os.path.join(base_dirs['silver'], d))])
            new_folders = [f for f in all_folders if not os.path.exists(os.path.join(base_dirs['platinum_combo'], f"{f}.csv"))]
            
            if not new_folders:
                print("[INFO] No new instruments to process.")
            else:
                print("\n--- Select Instrument(s) to Process ---")
                for i, f in enumerate(new_folders): print(f"  [{i+1}] {f}")
                print("\nSelect multiple with comma-separated numbers (e.g., 1,3,5)")
                user_input = input("Enter number(s) to process: ").strip()
                if user_input:
                    try:
                        indices = [int(i.strip()) - 1 for i in user_input.split(',')]
                        valid_indices = sorted(set(idx for idx in indices if 0 <= idx < len(new_folders)))
                        instrument_folders_to_process = [new_folders[idx] for idx in valid_indices]
                    except ValueError:
                        print("[ERROR] Invalid input.")
        except FileNotFoundError:
            print(f"[ERROR] Source directory not found: {base_dirs['silver']}")

    if not instrument_folders_to_process:
        print("[INFO] No instruments selected for processing.")
        return

    print(f"\n[QUEUE] Queued {len(instrument_folders_to_process)} instrument(s): {instrument_folders_to_process}")
    for instrument_name in instrument_folders_to_process:
        try:
            print(f"\n{'='*50}\nProcessing Instrument: {instrument_name}\n{'='*50}")
            run_unified_processor(instrument_name, base_dirs)
        except Exception:
            print(f"\n[FATAL ERROR] A critical error occurred while processing {instrument_name}.")
            traceback.print_exc()

    print("\n" + "="*50 + "\n[COMPLETE] All Platinum preprocessing tasks are finished.")

if __name__ == "__main__":
    main()
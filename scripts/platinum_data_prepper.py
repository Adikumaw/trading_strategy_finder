# platinum_data_prepper.py (V2.0 - Buffered Streaming Architecture)

"""
Platinum Layer - Pre-Processor: The Data Prepper

This single, high-performance script unifies the discovery of strategy
blueprints and the pre-computation of their performance data. It embodies a
"Buffered Streaming" architecture for maximum performance and I/O efficiency.

The script operates in two distinct, sequential phases:

Phase 1: Parallel Discovery and Buffered Streaming
- It reads the enriched Silver data chunks in parallel. A worker (mapper)
  discovers all blueprints in a chunk and aggregates their trade counts.
- These results are returned to the main process, which collects them in a large
  in-memory buffer.
- Once the buffer reaches a size threshold, a "flush" operation is triggered.
  The buffer is efficiently grouped by strategy key, and the batched results
  are APPENDED to a single temporary file per key (e.g., temp_targets/{key}.csv).
- This avoids the "millions of tiny files" problem, dramatically reducing I/O
  overhead while still reading the source data only ONCE.

Phase 2: Parallel Consolidation (The "Reducer")
- After all data is streamed to the temporary files, a second parallel process
  is started. Each worker is assigned a temporary file.
- The worker's job is to load its file, perform the final aggregation
  (groupby().sum()), and write the clean, final target file to the
  `platinum_data/targets` directory.
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

# The size of each bin for basis points. This controls the granularity of the discovery.
BPS_BIN_SIZE: float = 5.0
# The number of records to hold in memory before flushing to disk.
# Balances memory usage with I/O efficiency.
BUFFER_FLUSH_THRESHOLD: int = 2_000_000

# --- PICKLE-SAFE HELPER FUNCTION ---
def nested_dd():
    """
    A picklable factory function for creating nested defaultdicts.
    This replaces the lambda function which is not compatible with multiprocessing.
    """
    return defaultdict(int)

# --- PHASE 1: WORKER FUNCTION (MAPPER) ---
def discover_and_aggregate_chunk(
    task_tuple: Tuple[str, List[str]]
) -> Dict[Tuple, Dict]:
    """
    Worker for Phase 1. Reads a chunk, discovers all blueprints using
    vectorized groupby, and returns the aggregated results.
    """
    chunk_path, all_levels = task_tuple
    chunk_results: Dict[Tuple, Dict] = defaultdict(nested_dd)

    try:
        chunk = pd.read_csv(chunk_path, parse_dates=['entry_time'])
        if chunk.empty: return {}
        
        chunk['sl_ratio'], chunk['tp_ratio'] = chunk['sl_ratio'].round(5), chunk['tp_ratio'].round(5)
        
        for level in all_levels:
            for sltp in ['sl', 'tp']:
                pct_col, bps_col = f"{sltp}_place_pct_to_{level}", f"{sltp}_dist_to_{level}_bps"
                if pct_col in chunk.columns: chunk[f'{pct_col}_bin'] = np.floor(chunk[pct_col] * 10)
                if bps_col in chunk.columns: chunk[f'{bps_col}_bin'] = np.floor(chunk[bps_col] / BPS_BIN_SIZE) * BPS_BIN_SIZE

        for level in all_levels:
            sl_pct_bin_col = f'sl_place_pct_to_{level}_bin'
            if sl_pct_bin_col in chunk.columns:
                df_filtered = chunk[chunk[sl_pct_bin_col].between(-20, 20, inclusive='both')]
                df_sl_pct = df_filtered.dropna(subset=[sl_pct_bin_col, 'tp_ratio'])
                if not df_sl_pct.empty:
                    groups = df_sl_pct.groupby([sl_pct_bin_col, 'tp_ratio', 'entry_time']).size()
                    for (sl_bin, tp_ratio, time), count in groups.items():
                        # --- MODIFIED: Using normalized 'ratio' schema ---
                        blueprint = ('SL-Pct', level, int(sl_bin), 'ratio', tp_ratio)
                        chunk_results[blueprint][time] += count

            tp_pct_bin_col = f'tp_place_pct_to_{level}_bin'
            if tp_pct_bin_col in chunk.columns:
                df_filtered = chunk[chunk[tp_pct_bin_col].between(-20, 20, inclusive='both')]
                df_tp_pct = df_filtered.dropna(subset=[tp_pct_bin_col, 'sl_ratio'])
                if not df_tp_pct.empty:
                    groups = df_tp_pct.groupby([tp_pct_bin_col, 'sl_ratio', 'entry_time']).size()
                    for (tp_bin, sl_ratio, time), count in groups.items():
                        # --- MODIFIED: Using normalized 'ratio' schema ---
                        blueprint = ('TP-Pct', 'ratio', sl_ratio, level, int(tp_bin))
                        chunk_results[blueprint][time] += count
            
            sl_bps_bin_col = f'sl_dist_to_{level}_bps_bin'
            if sl_bps_bin_col in chunk.columns:
                df_filtered = chunk[chunk[sl_bps_bin_col].between(-50, 50, inclusive='both')]
                df_sl_bps = df_filtered.dropna(subset=[sl_bps_bin_col, 'tp_ratio'])
                if not df_sl_bps.empty:
                    groups = df_sl_bps.groupby([sl_bps_bin_col, 'tp_ratio', 'entry_time']).size()
                    for (sl_bin, tp_ratio, time), count in groups.items():
                        # --- MODIFIED: Using normalized 'ratio' schema ---
                        blueprint = ('SL-BPS', level, sl_bin, 'ratio', tp_ratio)
                        chunk_results[blueprint][time] += count

            tp_bps_bin_col = f'tp_dist_to_{level}_bps_bin'
            if tp_bps_bin_col in chunk.columns:
                df_filtered = chunk[chunk[tp_bps_bin_col].between(-50, 50, inclusive='both')]
                df_tp_bps = df_filtered.dropna(subset=[tp_bps_bin_col, 'sl_ratio'])
                if not df_tp_bps.empty:
                    groups = df_tp_bps.groupby([tp_bps_bin_col, 'sl_ratio', 'entry_time']).size()
                    for (tp_bin, sl_ratio, time), count in groups.items():
                        # --- MODIFIED: Using normalized 'ratio' schema ---
                        blueprint = ('TP-BPS', 'ratio', sl_ratio, level, tp_bin)
                        chunk_results[blueprint][time] += count
    except Exception:
        print(f"[WORKER WARNING] Could not process chunk {os.path.basename(chunk_path)}.")
        traceback.print_exc()

    return chunk_results

# --- NEW: BUFFER FLUSHING FUNCTION ---
def flush_buffer_to_disk(buffer: List[Tuple], temp_dir: str):
    """
    Converts the in-memory buffer to a DataFrame, groups by key, and appends
    to the appropriate temporary files in efficient batches.
    """
    if not buffer: return
    df = pd.DataFrame(buffer, columns=['key', 'entry_time', 'trade_count'])
    
    for key, group in df.groupby('key'):
        filepath = os.path.join(temp_dir, f"{key}.csv")
        file_exists = os.path.exists(filepath)
        # Append the entire group for this key at once
        group[['entry_time', 'trade_count']].to_csv(filepath, mode='a', header=not file_exists, index=False)

# --- PHASE 2: WORKER FUNCTION (REDUCER) ---
def consolidate_target_file(temp_file_path: str, final_dir: str) -> None:
    """
    Worker for Phase 2. Reads a single temporary file, consolidates it,
    writes the final target file, and cleans up the temp file.
    """
    key = os.path.basename(temp_file_path).replace('.csv', '')
    try:
        df = pd.read_csv(temp_file_path)
        if df.empty:
            os.remove(temp_file_path)
            return
            
        final_df = df.groupby('entry_time')['trade_count'].sum().reset_index()
        final_target_path = os.path.join(final_dir, f"{key}.csv")
        final_df.to_csv(final_target_path, index=False)
        
        os.remove(temp_file_path)
    except Exception:
        print(f"[CONSOLIDATOR WARNING] Failed to consolidate key {key}.")
        traceback.print_exc()

# --- MAIN ORCHESTRATOR ---
def run_data_prepper_for_instrument(instrument_name: str, base_dirs: Dict[str, str]) -> None:
    chunked_outcomes_dir = os.path.join(base_dirs['silver'], instrument_name)
    combinations_path = os.path.join(base_dirs['platinum_combo'], f"{instrument_name}.csv")
    temp_targets_dir = os.path.join(base_dirs['platinum_temp'], instrument_name)
    final_targets_dir = os.path.join(base_dirs['platinum_final'], instrument_name)
    
    if os.path.exists(combinations_path):
        print(f"[INFO] {instrument_name} already has a combinations file. Skipping.")
        return
    
    for d in [temp_targets_dir, final_targets_dir]:
        if os.path.exists(d): shutil.rmtree(d)
        os.makedirs(d)

    try:
        chunk_files = [os.path.join(chunked_outcomes_dir, f) for f in os.listdir(chunked_outcomes_dir) if f.endswith('.csv')]
        if not chunk_files:
            print(f"[ERROR] No chunk files found for {instrument_name}.")
            return
    except FileNotFoundError:
        print(f"[ERROR] Chunks directory not found: {chunked_outcomes_dir}")
        return

    header_df = pd.read_csv(chunk_files[0], nrows=0)
    level_pattern = re.compile(r'(?:sl|tp)_(?:place_pct_to|dist_to)_([a-zA-Z0-9_]+)(?:_bps)?')
    all_levels = sorted(list(set(
        match.group(1) for col in header_df.columns for match in [level_pattern.match(col)] if match
    )))
    print(f"Discovered {len(all_levels)} market levels for {instrument_name}.")

    print("\n--- Phase 1: Discovering Blueprints and Streaming Targets ---")
    tasks = [(path, all_levels) for path in chunk_files]
    master_blueprint_keys: Dict[Tuple, str] = {}
    master_buffer: List[Tuple] = []

    with Pool(processes=MAX_CPU_USAGE) as pool:
        for chunk_results in tqdm(pool.imap_unordered(discover_and_aggregate_chunk, tasks), total=len(tasks), desc="Phase 1: Processing Chunks"):
            for blueprint, counts in chunk_results.items():
                if blueprint not in master_blueprint_keys:
                    key_str = '-'.join([str(x) for x in blueprint])
                    key = hashlib.sha256(key_str.encode()).hexdigest()[:16]
                    master_blueprint_keys[blueprint] = key
                
                key = master_blueprint_keys[blueprint]
                for time, count in counts.items():
                    master_buffer.append((key, time, count))
            
            if len(master_buffer) >= BUFFER_FLUSH_THRESHOLD:
                print(f"\n[INFO] Buffer limit reached. Flushing {len(master_buffer)} records...")
                flush_buffer_to_disk(master_buffer, temp_targets_dir)
                master_buffer.clear()
    
    if master_buffer:
        print(f"\n[INFO] Flushing remaining {len(master_buffer)} records from buffer...")
        flush_buffer_to_disk(master_buffer, temp_targets_dir)
        master_buffer.clear()
    
    print(f"\nPhase 1 Complete. Discovered {len(master_blueprint_keys)} unique blueprints.")

    definitions = []
    for blueprint, key in master_blueprint_keys.items():
        type, sl_def, sl_bin, tp_def, tp_bin = blueprint
        # --- MODIFIED: Handle normalized 'ratio' schema ---
        final_sl_def, final_sl_bin = (sl_def, sl_bin) if sl_def != 'ratio' else ('ratio', sl_bin)
        final_tp_def, final_tp_bin = (tp_def, tp_bin) if tp_def != 'ratio' else ('ratio', tp_bin)
        definitions.append({'key': key, 'type': type, 'sl_def': final_sl_def, 'sl_bin': final_sl_bin, 'tp_def': final_tp_def, 'tp_bin': final_tp_bin})
        
    pd.DataFrame(definitions).to_csv(combinations_path, index=False)
    print(f"Saved final combinations file to {combinations_path}")

    print("\n--- Phase 2: Consolidating Temporary Target Files ---")
    temp_files = [os.path.join(temp_targets_dir, f) for f in os.listdir(temp_targets_dir)]
    if not temp_files:
        print("[WARNING] No temporary files were generated. Skipping consolidation.")
    else:
        consolidation_func = partial(consolidate_target_file, final_dir=final_targets_dir)
        with Pool(processes=MAX_CPU_USAGE) as pool:
            list(tqdm(pool.imap_unordered(consolidation_func, temp_files), total=len(temp_files), desc="Phase 2: Consolidating Targets"))
        print("Phase 2 Complete. Final targets saved.")

    shutil.rmtree(temp_targets_dir)

def main() -> None:
    # This function is correct and remains unchanged.
    core_dir = os.path.dirname(os.path.abspath(__file__))
    base_dirs = { 'silver': os.path.abspath(os.path.join(core_dir, '..', 'silver_data', 'chunked_outcomes')), 'platinum_combo': os.path.abspath(os.path.join(core_dir, '..', 'platinum_data', 'combinations')), 'platinum_temp': os.path.abspath(os.path.join(core_dir, '..', 'platinum_data', 'temp_targets')), 'platinum_final': os.path.abspath(os.path.join(core_dir, '..', 'platinum_data', 'targets')) }
    for d in base_dirs.values(): os.makedirs(d, exist_ok=True)
    instrument_folders_to_process, target_arg = [], sys.argv[1] if len(sys.argv) > 1 else None
    if target_arg:
        instrument_name = target_arg.replace('.csv', '')
        print(f"[TARGET] Targeted Mode: Processing instrument '{instrument_name}'")
        if os.path.isdir(os.path.join(base_dirs['silver'], instrument_name)): instrument_folders_to_process = [instrument_name]
        else: print(f"[ERROR] Silver chunk directory not found for: {instrument_name}")
    else:
        print("[SCAN] Interactive Mode: Scanning for new instrument folders...")
        try:
            all_folders = sorted([d for d in os.listdir(base_dirs['silver']) if os.path.isdir(os.path.join(base_dirs['silver'], d))])
            new_folders = [f for f in all_folders if not os.path.exists(os.path.join(base_dirs['platinum_combo'], f"{f}.csv"))]
            if not new_folders: print("[INFO] No new instruments to process.")
            else:
                print("\n--- Select Instrument(s) to Process ---")
                for i, f in enumerate(new_folders): print(f"  [{i+1}] {f}")
                user_input = input("Enter number(s) to process: ").strip()
                if user_input:
                    try:
                        indices = [int(i.strip()) - 1 for i in user_input.split(',')]
                        instrument_folders_to_process = [new_folders[idx] for idx in sorted(set(indices)) if 0 <= idx < len(new_folders)]
                    except ValueError: print("[ERROR] Invalid input.")
        except FileNotFoundError: print(f"[ERROR] Source directory not found: {base_dirs['silver']}")
    if not instrument_folders_to_process: print("[INFO] No instruments selected for processing.")
    else:

        print(f"\n[QUEUE] Queued {len(instrument_folders_to_process)} instrument(s): {instrument_folders_to_process}")
        for instrument_name in instrument_folders_to_process:
            try:
                print(f"\n{'='*50}\nProcessing Instrument: {instrument_name}\n{'='*50}")
                run_data_prepper_for_instrument(instrument_name, base_dirs)
            except Exception:
                print(f"\n[FATAL ERROR] A critical error occurred while processing {instrument_name}.")
                traceback.print_exc()
    print("\n" + "="*50 + "\n[COMPLETE] All Platinum preprocessing tasks are finished.")

if __name__ == "__main__":
    main()
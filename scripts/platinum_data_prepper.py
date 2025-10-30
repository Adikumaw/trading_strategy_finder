# platinum_preprocessor.py (V3.1 - Robust Error Handling)

"""
Platinum Layer - Pre-Processor: The Data Prepper

This single, high-performance script unifies the discovery of strategy
blueprints and the pre-computation of their performance data. It uses a
"Sharded Streaming" architecture for maximum performance and I/O efficiency.

Phase 1: Parallel Discovery and Sharded Streaming
- It reads Silver data chunks in parallel. A worker (mapper) discovers all
  blueprints in a chunk and aggregates their trade counts.
- The main process collects these results in a large in-memory buffer.
- Once the buffer is full, a "flush" operation is triggered. The buffer is
  "sharded" based on the strategy key, and results are appended in large,
  efficient batches to a small, fixed number of temporary shard files.
- This solves the I/O bottleneck of writing to thousands of tiny files.

Phase 2: Parallel Consolidation of Shards
- After streaming is complete, a second parallel process is started. Each worker
  is assigned one shard file.
- The worker loads its shard, performs a final aggregation (groupby('key')),
  and writes the final, clean target files. This is memory-safe as each
  shard is a manageable size.
"""

import os
import re
import sys
import shutil
import hashlib
import traceback
from multiprocessing import Pool, cpu_count, current_process
from collections import defaultdict
from typing import Dict, List, Tuple
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm

# --- CONFIGURATION ---
MAX_CPU_USAGE: int = max(1, cpu_count() - 2)
BPS_BIN_SIZE: float = 5.0
BUFFER_FLUSH_THRESHOLD: int = 2_000_000
NUM_SHARDS: int = 128

# --- PICKLE-SAFE HELPER FUNCTION ---
def nested_dd():
    return defaultdict(int)

# --- PHASE 1: WORKER FUNCTION (MAPPER) ---
def discover_and_aggregate_chunk(task_tuple: Tuple[str, List[str]]) -> Dict:
    """
    Worker for Phase 1. Now returns a dictionary which includes either the
    result or the error information for robust handling.
    """
    chunk_path, all_levels = task_tuple
    try:
        chunk_results: Dict[Tuple, Dict] = defaultdict(nested_dd)
        chunk = pd.read_csv(chunk_path, parse_dates=['entry_time'])
        if chunk.empty: return {'status': 'success', 'result': {}}
        
        chunk['sl_ratio'], chunk['tp_ratio'] = chunk['sl_ratio'].round(5), chunk['tp_ratio'].round(5)
        
        for level in all_levels:
            for sltp in ['sl', 'tp']:
                pct_col, bps_col = f"{sltp}_place_pct_to_{level}", f"{sltp}_dist_to_{level}_bps"
                if pct_col in chunk.columns: chunk[f'{pct_col}_bin'] = np.floor(chunk[pct_col] * 10)
                if bps_col in chunk.columns: chunk[f'{bps_col}_bin'] = np.floor(chunk[bps_col] / BPS_BIN_SIZE) * BPS_BIN_SIZE
        
        for level in all_levels:
            sl_pct_bin_col = f'sl_place_pct_to_{level}_bin'
            if sl_pct_bin_col in chunk.columns:
                df_filtered = chunk[chunk[sl_pct_bin_col].between(-20, 20, inclusive='both')].dropna(subset=[sl_pct_bin_col, 'tp_ratio'])
                if not df_filtered.empty:
                    groups = df_filtered.groupby([sl_pct_bin_col, 'tp_ratio', 'entry_time']).size()
                    for (sl_bin, tp_ratio, time), count in groups.items():
                        chunk_results[('SL-Pct', level, int(sl_bin), 'ratio', tp_ratio)][time] += count
            tp_pct_bin_col = f'tp_place_pct_to_{level}_bin'
            if tp_pct_bin_col in chunk.columns:
                df_filtered = chunk[chunk[tp_pct_bin_col].between(-20, 20, inclusive='both')].dropna(subset=[tp_pct_bin_col, 'sl_ratio'])
                if not df_filtered.empty:
                    groups = df_filtered.groupby([tp_pct_bin_col, 'sl_ratio', 'entry_time']).size()
                    for (tp_bin, sl_ratio, time), count in groups.items():
                        chunk_results[('TP-Pct', 'ratio', sl_ratio, level, int(tp_bin))][time] += count
            sl_bps_bin_col = f'sl_dist_to_{level}_bps_bin'
            if sl_bps_bin_col in chunk.columns:
                df_filtered = chunk[chunk[sl_bps_bin_col].between(-50, 50, inclusive='both')].dropna(subset=[sl_bps_bin_col, 'tp_ratio'])
                if not df_filtered.empty:
                    groups = df_filtered.groupby([sl_bps_bin_col, 'tp_ratio', 'entry_time']).size()
                    for (sl_bin, tp_ratio, time), count in groups.items():
                        chunk_results[('SL-BPS', level, sl_bin, 'ratio', tp_ratio)][time] += count
            tp_bps_bin_col = f'tp_dist_to_{level}_bps_bin'
            if tp_bps_bin_col in chunk.columns:
                df_filtered = chunk[chunk[tp_bps_bin_col].between(-50, 50, inclusive='both')].dropna(subset=[tp_bps_bin_col, 'sl_ratio'])
                if not df_filtered.empty:
                    groups = df_filtered.groupby([tp_bps_bin_col, 'sl_ratio', 'entry_time']).size()
                    for (tp_bin, sl_ratio, time), count in groups.items():
                        chunk_results[('TP-BPS', 'ratio', sl_ratio, level, tp_bin)][time] += count
        
        return {'status': 'success', 'result': chunk_results}

    except Exception as e:
        # --- MODIFIED: Return the error information instead of just printing ---
        worker_name = current_process().name
        error_info = {
            'status': 'error',
            'worker': worker_name,
            'chunk': os.path.basename(chunk_path),
            'error': str(e),
            'traceback': traceback.format_exc()
        }
        return error_info

# --- SHARDED BUFFER FLUSHING FUNCTION ---
def flush_buffer_to_shards(buffer: List[Tuple], temp_dir: str):
    # This function is correct and remains unchanged.
    if not buffer: return
    df = pd.DataFrame(buffer, columns=['key', 'entry_time', 'trade_count'])
    df['shard_id'] = df['key'].apply(lambda x: hash(x) % NUM_SHARDS)
    for shard_id, group in df.groupby('shard_id'):
        filepath = os.path.join(temp_dir, f"_temp_shard_{shard_id}.csv")
        group[['key', 'entry_time', 'trade_count']].to_csv(filepath, mode='a', header=not os.path.exists(filepath), index=False)

# --- PHASE 2: WORKER FUNCTION (REDUCER) ---
def consolidate_shard_file(shard_path: str, final_dir: str) -> int:
    # This function is correct and remains unchanged.
    try:
        df = pd.read_csv(shard_path)
        if df.empty:
            os.remove(shard_path)
            return 0
        for key, group in df.groupby('key'):
            final_df = group.groupby('entry_time')['trade_count'].sum().reset_index()
            final_target_path = os.path.join(final_dir, f"{key}.csv")
            final_df.to_csv(final_target_path, index=False)
        os.remove(shard_path)
        return len(df['key'].unique())
    except Exception:
        print(f"[CONSOLIDATOR WARNING] Failed to consolidate shard {os.path.basename(shard_path)}.")
        traceback.print_exc()
        return 0

# --- MAIN ORCHESTRATOR ---
def run_preprocessor_for_instrument(instrument_name: str, base_dirs: Dict[str, str]) -> None:
    # Most of this function is correct, but the main processing loop is updated.
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
    all_levels = sorted(list(set(match.group(1) for col in header_df.columns for match in [level_pattern.match(col)] if match)))
    print(f"Discovered {len(all_levels)} market levels for {instrument_name}.")

    print("\n--- Phase 1: Discovering Blueprints and Streaming to Shards ---")
    tasks = [(path, all_levels) for path in chunk_files]
    master_blueprint_keys: Dict[Tuple, str] = {}
    master_buffer: List[Tuple] = []

    # --- MODIFIED: Added robust error handling around the processing loop ---
    try:
        with Pool(processes=MAX_CPU_USAGE) as pool:
            with tqdm(total=len(tasks), desc="Phase 1: Processing Chunks") as pbar:
                for result_dict in pool.imap_unordered(discover_and_aggregate_chunk, tasks):
                    if result_dict['status'] == 'error':
                        # A worker failed! Print the clean error and stop everything.
                        print("\n" + "="*80)
                        print(f"[FATAL WORKER ERROR] Worker '{result_dict['worker']}' failed on chunk '{result_dict['chunk']}'.")
                        print(f"Error: {result_dict['error']}")
                        print("--- Full Traceback from Worker ---")
                        print(result_dict['traceback'])
                        print("="*80)
                        # Terminate the pool immediately to stop other workers
                        pool.terminate()
                        # Exit the script with an error code
                        sys.exit(1)
                    
                    # If we get here, the worker was successful
                    chunk_results = result_dict['result']
                    for blueprint, counts in chunk_results.items():
                        if blueprint not in master_blueprint_keys:
                            key_str = '-'.join([str(x) for x in blueprint])
                            key = hashlib.sha256(key_str.encode()).hexdigest()[:16]
                            master_blueprint_keys[blueprint] = key
                        key = master_blueprint_keys[blueprint]
                        for time, count in counts.items():
                            master_buffer.append((key, time, count))
                    
                    if len(master_buffer) >= BUFFER_FLUSH_THRESHOLD:
                        flush_buffer_to_shards(master_buffer, temp_targets_dir)
                        master_buffer.clear()
                    
                    pbar.update(1)

    except Exception as e:
        print("\n[FATAL ORCHESTRATOR ERROR] The main process encountered an unhandled exception.")
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)

    if master_buffer:
        flush_buffer_to_shards(master_buffer, temp_targets_dir)
        master_buffer.clear()
    
    print(f"\nPhase 1 Complete. Discovered {len(master_blueprint_keys)} unique blueprints.")

    definitions = [{'key': key, 'type': bp[0], 'sl_def': bp[1], 'sl_bin': bp[2], 'tp_def': bp[3], 'tp_bin': bp[4]} for bp, key in master_blueprint_keys.items()]
    pd.DataFrame(definitions).to_csv(combinations_path, index=False)
    print(f"Saved final combinations file to {combinations_path}")

    print("\n--- Phase 2: Consolidating Shard Files ---")
    shard_files = [os.path.join(temp_targets_dir, f) for f in os.listdir(temp_targets_dir) if f.startswith('_temp_shard_')]
    if not shard_files:
        print("[WARNING] No temporary shard files were generated. Skipping consolidation.")
    else:
        consolidation_func = partial(consolidate_shard_file, final_dir=final_targets_dir)
        with Pool(processes=MAX_CPU_USAGE) as pool:
            list(tqdm(pool.imap_unordered(consolidation_func, shard_files), total=len(shard_files), desc="Phase 2: Consolidating Shards"))
        print("Phase 2 Complete. Final targets saved.")
    
    if os.path.exists(temp_targets_dir):
        shutil.rmtree(temp_targets_dir)

# --- The main() function remains unchanged ---
def main() -> None:
    core_dir = os.path.dirname(os.path.abspath(__file__))
    base_dirs = { 'silver': os.path.abspath(os.path.join(core_dir, '..', 'silver_data', 'chunked_outcomes')), 'platinum_combo': os.path.abspath(os.path.join(core_dir, '..', 'platinum_data', 'combinations')), 'platinum_temp': os.path.abspath(os.path.join(core_dir, '..', 'platinum_data', 'temp_targets')), 'platinum_final': os.path.abspath(os.path.join(core_dir, '..', 'platinum_data', 'targets')) }
    for d in base_dirs.values(): os.makedirs(d, exist_ok=True)
    instrument_folders_to_process = []
    target_arg = sys.argv[1] if len(sys.argv) > 1 else None
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
                run_preprocessor_for_instrument(instrument_name, base_dirs)
            except Exception:
                print(f"\n[FATAL ORCHESTRATOR ERROR] An unhandled exception occurred in main loop for {instrument_name}.")
                traceback.print_exc()
    print("\n" + "="*50 + "\n[COMPLETE] All Platinum preprocessing tasks are finished.")

if __name__ == "__main__":
    main()
# platinum_data_prepper.py (V1.0 - Documented & Vectorized)


"""
Platinum Layer - Pre-Processor: The Data Prepper

This single, high-performance script unifies the discovery of strategy
blueprints and the pre-computation of their performance data. It replaces the
two previous Platinum stages (combinations_generator and target_extractor)
and embodies a "Live Hashing and Streaming" architecture to maximize
performance and minimize I/O.

The script operates in two distinct, sequential phases:

Phase 1: Parallel Discovery and Streaming (The "Mapper")

    It reads the enriched Silver data chunks in parallel.

    For each chunk, a worker discovers all unique strategy blueprints and
    simultaneously aggregates the number of trades per candle for each blueprint.

    These results (blueprint -> trade counts) are returned to the main process.

    The main process receives these results in a stream. For each new, unseen
    blueprint, it generates a unique hash key and immediately streams the
    aggregated trade counts to a temporary file on disk ({key}.partX.csv).

    This achieves the primary goal: reading the expensive source data ONLY ONCE.

Phase 2: Parallel Consolidation (The "Reducer")

    After all chunks have been processed, the script has a master list of all
    discovered blueprints and thousands of small, temporary target files.

    It starts a second parallel process. Each worker is assigned a single hash key.

    The worker's job is to find all temporary files for that key, load them,
    perform a final aggregation (groupby().sum()), and write the clean, final
    target file to the platinum_data/targets directory.

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

# The size of each bin for basis points. This controls the granularity of the discovery.
BPS_BIN_SIZE: float = 5.0

# --- NEW: PICKLE-SAFE HELPER FUNCTION ---
def nested_dd():
    """
    A picklable factory function for creating nested defaultdicts.
    This replaces the lambda function which is not compatible with multiprocessing.
    """
    return defaultdict(int)

# --- PHASE 1: WORKER FUNCTION (MAPPER) ---
def discover_and_aggregate_chunk(task_tuple: Tuple[str, List[str], int]) -> Tuple[Dict[Tuple, Dict], int]:
    """
    Worker for Phase 1 (Mapper). Reads a chunk, discovers all blueprints,
    aggregates their trade counts by timestamp, and returns the results.This function is highly optimized. It performs all binning calculations in
    a vectorized manner and then uses a series of `groupby` operations to
    discover and count all blueprint occurrences at once, avoiding slow
    row-by-row iteration.

    Args:
        task_tuple: A tuple containing:
            - chunk_path (str): The path to the data chunk file.
            - all_levels (List[str]): A list of all market level names to check.
            - chunk_id (int): A unique identifier for the chunk.

    Returns:
        A tuple containing:
        - chunk_results (Dict): A dictionary mapping a blueprint tuple to its
        aggregated trade counts for the chunk.
        - chunk_id (int): The original chunk identifier.
    """
    chunk_path, all_levels, chunk_id = task_tuple

    # The primary data structure for this chunk's results.
    # Format: {blueprint_tuple -> {timestamp -> trade_count}}
    # --- MODIFIED: Use the named function instead of a lambda ---
    chunk_results: Dict[Tuple, Dict] = defaultdict(nested_dd)

    try:
        chunk = pd.read_csv(chunk_path, parse_dates=['entry_time'])
        if chunk.empty:
            return {}, chunk_id
        
        chunk['sl_ratio'] = chunk['sl_ratio'].round(5)
        chunk['tp_ratio'] = chunk['tp_ratio'].round(5)
        
        # --- Vectorized Pre-Binning ---
        # Create all binned columns once for maximum speed.
        for level in all_levels:
            for sltp in ['sl', 'tp']:
                pct_col = f"{sltp}_place_pct_to_{level}"
                bps_col = f"{sltp}_dist_to_{level}_bps"
                if pct_col in chunk.columns:
                    chunk[f'{pct_col}_bin'] = np.floor(chunk[pct_col] * 10)
                if bps_col in chunk.columns:
                    chunk[f'{bps_col}_bin'] = np.floor(chunk[bps_col] / BPS_BIN_SIZE) * BPS_BIN_SIZE

        # --- Vectorized Discovery and Aggregation ---
        for level in all_levels:
            # --- SL-Pct ---
            sl_pct_bin_col = f'sl_place_pct_to_{level}_bin'
            if sl_pct_bin_col in chunk.columns:
                # --- ADDED: Filter for valid bin range before grouping ---
                df_filtered = chunk[chunk[sl_pct_bin_col].between(-20, 20, inclusive='both')]
                df_sl_pct = df_filtered.dropna(subset=[sl_pct_bin_col, 'tp_ratio'])
                if not df_sl_pct.empty:
                    groups = df_sl_pct.groupby([sl_pct_bin_col, 'tp_ratio', 'entry_time']).size()
                    for (sl_bin, tp_ratio, time), count in groups.items():
                        blueprint = ('SL-Pct', level, int(sl_bin), 'ratio', tp_ratio)
                        chunk_results[blueprint][time] += count

            # --- TP-Pct ---
            tp_pct_bin_col = f'tp_place_pct_to_{level}_bin'
            if tp_pct_bin_col in chunk.columns:
                # --- ADDED: Filter for valid bin range before grouping ---
                df_filtered = chunk[chunk[tp_pct_bin_col].between(-20, 20, inclusive='both')]
                df_tp_pct = df_filtered.dropna(subset=[tp_pct_bin_col, 'sl_ratio'])
                if not df_tp_pct.empty:
                    groups = df_tp_pct.groupby([tp_pct_bin_col, 'sl_ratio', 'entry_time']).size()
                    for (tp_bin, sl_ratio, time), count in groups.items():
                        blueprint = ('TP-Pct', 'ratio', sl_ratio, level, int(tp_bin))
                        chunk_results[blueprint][time] += count
            
            # --- SL-BPS ---
            sl_bps_bin_col = f'sl_dist_to_{level}_bps_bin'
            if sl_bps_bin_col in chunk.columns:
                # --- ADDED: Filter for valid bin range before grouping ---
                df_filtered = chunk[chunk[sl_bps_bin_col].between(-50, 50, inclusive='both')]
                df_sl_bps = df_filtered.dropna(subset=[sl_bps_bin_col, 'tp_ratio'])
                if not df_sl_bps.empty:
                    groups = df_sl_bps.groupby([sl_bps_bin_col, 'tp_ratio', 'entry_time']).size()
                    for (sl_bin, tp_ratio, time), count in groups.items():
                        blueprint = ('SL-BPS', level, sl_bin, 'ratio', tp_ratio)
                        chunk_results[blueprint][time] += count

            # --- TP-BPS ---
            tp_bps_bin_col = f'tp_dist_to_{level}_bps_bin'
            if tp_bps_bin_col in chunk.columns:
                # --- ADDED: Filter for valid bin range before grouping ---
                df_filtered = chunk[chunk[tp_bps_bin_col].between(-50, 50, inclusive='both')]
                df_tp_bps = df_filtered.dropna(subset=[tp_bps_bin_col, 'sl_ratio'])
                if not df_tp_bps.empty:
                    groups = df_tp_bps.groupby([tp_bps_bin_col, 'sl_ratio', 'entry_time']).size()
                    for (tp_bin, sl_ratio, time), count in groups.items():
                        blueprint = ('TP-BPS', 'ratio', sl_ratio, level, tp_bin)
                        chunk_results[blueprint][time] += count

    except Exception:
        print(f"[WORKER WARNING] Could not process chunk {os.path.basename(chunk_path)}.")
        traceback.print_exc()

    return chunk_results, chunk_id

# --- PHASE 2: WORKER FUNCTION (REDUCER) ---
def consolidate_target_files(key: str, temp_dir: str, final_dir: str) -> None:
    """
    Worker for Phase 2 (Reducer). Finds all temporary part-files for a single
    key, loads them, consolidates them into a single aggregated file, and
    writes the final clean target file.Args:
        key: The unique hash key for a single strategy blueprint.
        temp_dir: The directory containing the temporary part-files.
        final_dir: The directory where the final consolidated target file will be saved.
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

def run_data_prepper_for_instrument(instrument_name: str, base_dirs: Dict[str, str]) -> None:
    """
    Orchestrates the entire unified discovery and extraction process for a single instrument.
    Args:
        instrument_name: The name of the instrument to process (e.g., 'XAUUSD15').
        base_dirs: A dictionary of required base directory paths.
    """
    chunked_outcomes_dir = os.path.join(base_dirs['silver'], instrument_name)
    combinations_path = os.path.join(base_dirs['platinum_combo'], f"{instrument_name}.csv")
    temp_targets_dir = os.path.join(base_dirs['platinum_temp'], instrument_name)
    final_targets_dir = os.path.join(base_dirs['platinum_final'], instrument_name)

    # --- Setup and Cleanup ---
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
    # This regex captures the 'level' part from columns like 'sl_place_pct_SMA_50'
    # or 'tp_dist_to_BB_upper_20_bps', making the discovery very robust.
    level_pattern = re.compile(r'(?:sl|tp)_(?:place_pct_to|dist_to)_([a-zA-Z0-9_]+)(?:_bps)?')
    all_levels = sorted(list(set(
        match.group(1) for col in header_df.columns for match in [level_pattern.match(col)] if match
    )))
    print(f"Discovered {len(all_levels)} market levels for {instrument_name}.")

    print("\n--- Phase 1: Discovering Blueprints and Streaming Targets ---")
    tasks = [(path, all_levels, i) for i, path in enumerate(chunk_files)]
    master_blueprint_keys: Dict[Tuple, str] = {}

    with Pool(processes=MAX_CPU_USAGE) as pool:
        for chunk_results, chunk_id in tqdm(pool.imap_unordered(discover_and_aggregate_chunk, tasks), total=len(tasks), desc="Phase 1: Processing Chunks"):
            for blueprint, counts in chunk_results.items():
                if blueprint not in master_blueprint_keys:
                    # New discovery: generate and store key
                    key_str = '-'.join([str(x) for x in blueprint])
                    key = hashlib.sha256(key_str.encode()).hexdigest()[:16]
                    master_blueprint_keys[blueprint] = key
                
                key = master_blueprint_keys[blueprint]
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

    print("\n--- Phase 2: Consolidating Temporary Target Files ---")
    all_keys = list(master_blueprint_keys.values())

    consolidation_func = partial(consolidate_target_files, temp_dir=temp_targets_dir, final_dir=final_targets_dir)

    with Pool(processes=MAX_CPU_USAGE) as pool:
        list(tqdm(pool.imap_unordered(consolidation_func, all_keys), total=len(all_keys), desc="Phase 2: Consolidating Targets"))
        
    print("Phase 2 Complete. Final targets saved.")
    shutil.rmtree(temp_targets_dir)

def main() -> None:
    """
    Main execution function: handles file discovery, user interaction,
    and orchestrates the processing of each instrument.
    """
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
                user_input = input("Enter number(s) to process: ").strip()
                if user_input:
                    try:
                        indices = [int(i.strip()) - 1 for i in user_input.split(',')]
                        instrument_folders_to_process = [new_folders[idx] for idx in sorted(set(indices)) if 0 <= idx < len(new_folders)]
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
            run_data_prepper_for_instrument(instrument_name, base_dirs)
        except Exception:
            print(f"\n[FATAL ERROR] A critical error occurred while processing {instrument_name}.")
            traceback.print_exc()

    print("\n" + "="*50 + "\n[COMPLETE] All Platinum preprocessing tasks are finished.")

if __name__ == "__main__":
    main()
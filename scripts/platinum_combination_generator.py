# platinum_combinations_generator.py (V6 - Re-documented & Robust)

"""
Platinum Layer - Stage 1: The Combination Generator (The Architect)

This script is the first and foundational stage of the Platinum discovery engine.
Its purpose is to scan the vast, enriched trade data from the Silver Layer and
discover every unique strategy "blueprint" that exists within it.

A "blueprint" defines a strategy's exit logic, for example:
- "Set the Stop-Loss 20% of the way to the daily support and use a fixed 0.5% Take-Profit."
- "Use a fixed 0.3% Stop-Loss and set the Take-Profit with a 10-pip buffer
  behind the upper Bollinger Band."

It transforms an infinite problem (every possible SL/TP placement) into a
finite one by using discretization (binning). Instead of treating every unique
placement as different, it groups them into logical buckets (e.g., all
placements between 10-20 basis points are treated as one).

This version uses a "map-reduce" parallel processing model to scan the data
chunks concurrently, significantly speeding up the discovery process.
"""

import os
import re
import sys
import traceback
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Set, Tuple

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


# --- WORKER FUNCTION ---

def find_combinations_in_chunk(task_tuple: Tuple[str, List[str]]) -> Tuple[Set, Set, Set, Set]:
    """
    The "Worker" or "Mapper" function, executed in parallel by the Pool.
    
    It receives a path to a single data chunk and the complete list of market
    levels to check against. It reads the chunk, discretizes (bins) the
    placement features, discovers all unique strategy blueprints within it,
    and returns them as sets for efficient merging.

    Args:
        task_tuple: A tuple containing:
            - chunk_path (str): The full path to the .csv chunk file.
            - all_levels (List[str]): A list of all market level names to check
              (e.g., 'SMA_50', 'support', 'BB_upper_20').

    Returns:
        A tuple of four sets containing the unique combinations found:
        (sl_pct_combos, tp_pct_combos, sl_bps_combos, tp_bps_combos).
    """
    chunk_path, all_levels = task_tuple
    
    # Initialize local sets for this worker's results. Using sets automatically
    # handles deduplication at the chunk level.
    local_sl_pct, local_tp_pct = set(), set()
    local_sl_bps, local_tp_bps = set(), set()

    def to_bin_pct(series: pd.Series) -> pd.Series:
        """Discretizes a percentage placement into 10% buckets (e.g., 0.1, 0.2)."""
        # Multiplies by 10 and floors, e.g., 0.23 -> 2.0, 0.99 -> 9.0
        return np.floor(series * 10).astype('Int64')
    
    def to_bin_bps(series: pd.Series) -> pd.Series:
        """Discretizes a basis points distance into buckets of BPS_BIN_SIZE."""
        # e.g., 13.5 bps with size 5.0 -> floor(2.7) * 5.0 -> 10.0
        return (np.floor(series / BPS_BIN_SIZE) * BPS_BIN_SIZE).astype('Int64')

    try:
        chunk = pd.read_csv(chunk_path)
        # Round the fixed ratios to a consistent precision to avoid floating point noise.
        if 'sl_ratio' in chunk.columns: chunk['sl_ratio'] = chunk['sl_ratio'].round(5)
        if 'tp_ratio' in chunk.columns: chunk['tp_ratio'] = chunk['tp_ratio'].round(5)
            
        for level in all_levels:
            # Define the expected column names based on the current level.
            sl_pct_col = f"sl_place_pct_to_{level}"
            tp_pct_col = f"tp_place_pct_to_{level}"
            sl_bps_col = f"sl_dist_to_{level}_bps"
            tp_bps_col = f"tp_dist_to_{level}_bps"

            # --- Discover "Semi-Dynamic SL (Percentage)" strategies ---
            if sl_pct_col in chunk.columns and 'tp_ratio' in chunk.columns:
                chunk['sl_bin_pct'] = to_bin_pct(chunk[sl_pct_col])
                # Filter for a sensible range to avoid extreme/outlier bins.
                # Bins are x10, so -20 to 20 represents -200% to +200% placement.
                sl_binned_chunk = chunk[chunk['sl_bin_pct'].between(-20, 20, inclusive='both')]
                # Find all unique combinations of (fixed TP ratio, binned SL placement).
                for combo in sl_binned_chunk[['tp_ratio', 'sl_bin_pct']].drop_duplicates().itertuples(index=False):
                    local_sl_pct.add((level, combo.sl_bin_pct, combo.tp_ratio))

            # --- Discover "Semi-Dynamic TP (Percentage)" strategies ---
            if tp_pct_col in chunk.columns and 'sl_ratio' in chunk.columns:
                chunk['tp_bin_pct'] = to_bin_pct(chunk[tp_pct_col])
                tp_binned_chunk = chunk[chunk['tp_bin_pct'].between(-20, 20, inclusive='both')]
                for combo in tp_binned_chunk[['sl_ratio', 'tp_bin_pct']].drop_duplicates().itertuples(index=False):
                    local_tp_pct.add((level, combo.tp_bin_pct, combo.sl_ratio))
            
            # --- Discover "Semi-Dynamic SL (Basis Points)" strategies ---
            if sl_bps_col in chunk.columns and 'tp_ratio' in chunk.columns:
                chunk['sl_bin_bps'] = to_bin_bps(chunk[sl_bps_col])
                # Filter for a sensible range of basis point buffers.
                sl_binned_chunk = chunk[chunk['sl_bin_bps'].between(-50, 50, inclusive='both')]
                for combo in sl_binned_chunk[['tp_ratio', 'sl_bin_bps']].drop_duplicates().itertuples(index=False):
                    local_sl_bps.add((level, combo.sl_bin_bps, combo.tp_ratio))

            # --- Discover "Semi-Dynamic TP (Basis Points)" strategies ---
            if tp_bps_col in chunk.columns and 'sl_ratio' in chunk.columns:
                chunk['tp_bin_bps'] = to_bin_bps(chunk[tp_bps_col])
                tp_binned_chunk = chunk[chunk['tp_bin_bps'].between(-50, 50, inclusive='both')]
                for combo in tp_binned_chunk[['sl_ratio', 'tp_bin_bps']].drop_duplicates().itertuples(index=False):
                    local_tp_bps.add((level, combo.tp_bin_bps, combo.sl_ratio))
                
    except Exception as e:
        print(f"[WORKER WARNING] Could not process chunk {os.path.basename(chunk_path)}. Error: {e}")
        traceback.print_exc()

    return local_sl_pct, local_tp_pct, local_sl_bps, local_tp_bps


# --- MAIN ORCHESTRATOR FUNCTION ---

def generate_strategy_definitions_parallel(chunked_outcomes_dir: str) -> pd.DataFrame:
    """
    Discovers all unique strategy blueprints by processing enriched data chunks in parallel.

    This function acts as the "Manager" in a map-reduce model.
    1. It discovers all chunk files and unique market levels from the data headers.
    2. It "maps" the `find_combinations_in_chunk` task across a pool of workers.
    3. It "reduces" the results by collecting the sets of unique combinations from
       all workers and merging them into a final, consolidated DataFrame.

    Args:
        chunked_outcomes_dir: The directory containing the Silver Layer's chunked outcome files.

    Returns:
        A DataFrame containing all unique strategy blueprints discovered, or an
        empty DataFrame if none are found.
    """
    print("Scanning for chunks and preparing for parallel processing...")
    try:
        chunk_files = [os.path.join(chunked_outcomes_dir, f) for f in os.listdir(chunked_outcomes_dir) if f.endswith('.csv')]
        if not chunk_files:
            print("[ERROR] No chunk files found in the directory.")
            return pd.DataFrame()
    except FileNotFoundError:
        print(f"[ERROR] Chunk directory not found: {chunked_outcomes_dir}")
        return pd.DataFrame()

    # --- Robust Level Discovery ---
    # Read the header of the first chunk to dynamically discover all possible market levels.
    header_df = pd.read_csv(chunk_files[0], nrows=0)
    # This regex captures the 'level' part from columns like 'sl_place_pct_SMA_50'
    # or 'tp_dist_to_BB_upper_20_bps', making the discovery very robust.
    level_pattern = re.compile(r'(?:sl|tp)_(?:place_pct_to|dist_to)_([a-zA-Z0-9_]+)(?:_bps)?')
    all_levels = sorted(list(set(
        match.group(1) for col in header_df.columns for match in [level_pattern.match(col)] if match
    )))
    
    if not all_levels:
        print("[ERROR] No valid placement feature columns found in chunk headers.")
        return pd.DataFrame()

    print(f"[debug] Discovered levels: {all_levels}")
    print(f"Found {len(all_levels)} potential levels. Dispatching {len(chunk_files)} chunks to workers...")

    # Create a list of tasks for the multiprocessing Pool.
    tasks = [(chunk_path, all_levels) for chunk_path in chunk_files]
    
    # Initialize master sets to collect and merge results from all workers.
    master_sl_pct, master_tp_pct = set(), set()
    master_sl_bps, master_tp_bps = set(), set()

    # --- Parallel Execution (Map-Reduce) ---
    with Pool(processes=MAX_CPU_USAGE) as pool:
        # `imap_unordered` is highly efficient here as the order of chunk completion is irrelevant.
        # It processes results as they become available, maximizing CPU utilization.
        for result_tuple in tqdm(pool.imap_unordered(find_combinations_in_chunk, tasks), total=len(tasks), desc="Scanning Chunks for Combinations"):
            # This is the "reduce" step. The main process consumes results from
            # workers and efficiently merges them into the master sets.
            master_sl_pct.update(result_tuple[0])
            master_tp_pct.update(result_tuple[1])
            master_sl_bps.update(result_tuple[2])
            master_tp_bps.update(result_tuple[3])
    
    print("Parallel processing complete. Consolidating final definitions...")
    
    # --- Consolidation ---
    # Convert the sets of discovered blueprints into a structured DataFrame.
    definitions = []
    for level, sl_bin, tp_ratio in master_sl_pct:
        definitions.append({'type': 'SL-Pct', 'sl_def': level, 'sl_bin': sl_bin, 'tp_def': tp_ratio, 'tp_bin': np.nan})
    for level, tp_bin, sl_ratio in master_tp_pct:
        definitions.append({'type': 'TP-Pct', 'sl_def': sl_ratio, 'sl_bin': np.nan, 'tp_def': level, 'tp_bin': tp_bin})
    for level, sl_bin, tp_ratio in master_sl_bps:
        definitions.append({'type': 'SL-BPS', 'sl_def': level, 'sl_bin': sl_bin, 'tp_def': tp_ratio, 'tp_bin': np.nan})
    for level, tp_bin, sl_ratio in master_tp_bps:
        definitions.append({'type': 'TP-BPS', 'sl_def': sl_ratio, 'sl_bin': np.nan, 'tp_def': level, 'tp_bin': tp_bin})

    if not definitions:
        print("\n[INFO] No valid combinations were found.")
        return pd.DataFrame()

    return pd.DataFrame(definitions).drop_duplicates().reset_index(drop=True)


def main() -> None:
    """
    Main execution function: handles file discovery, user interaction,
    and orchestrates the serial processing of each instrument folder.
    """
    # --- Define Project Directory Structure ---
    core_dir = os.path.dirname(os.path.abspath(__file__))
    chunked_outcomes_parent_dir = os.path.abspath(os.path.join(core_dir, '..', 'silver_data', 'chunked_outcomes'))
    combinations_dir = os.path.abspath(os.path.join(core_dir, '..', 'platinum_data', 'combinations'))
    os.makedirs(combinations_dir, exist_ok=True)

    instrument_folders_to_process = []
    target_file_arg = sys.argv[1] if len(sys.argv) > 1 else None
    
    if target_file_arg:
        # --- Targeted Mode ---
        instrument_name = target_file_arg.replace('.csv', '')
        print(f"[TARGET] Targeted Mode: Processing instrument '{instrument_name}'")
        instrument_chunk_dir_check = os.path.join(chunked_outcomes_parent_dir, instrument_name)
        if not os.path.isdir(instrument_chunk_dir_check):
            print(f"[ERROR] Target instrument folder not found: {instrument_chunk_dir_check}")
        else:
            instrument_folders_to_process = [instrument_name]
    else:
        # --- Interactive Mode ---
        print("[SCAN] Interactive Mode: Scanning for new instrument folders...")
        try:
            all_folders = sorted([d for d in os.listdir(chunked_outcomes_parent_dir) if os.path.isdir(os.path.join(chunked_outcomes_parent_dir, d))])
            new_folders = [f for f in all_folders if not os.path.exists(os.path.join(combinations_dir, f"{f}.csv"))]
            
            if not new_folders:
                print("[INFO] No new instrument folders to process.")
            else:
                print("\n--- Select Folder(s) to Process ---")
                for i, f in enumerate(new_folders): print(f"  [{i+1}] {f}")
                print("\nSelect multiple folders with comma-separated numbers (e.g., 1,3,5)")
                user_input = input("Enter number(s) to process: ").strip()
                if user_input:
                    try:
                        indices = [int(i.strip()) - 1 for i in user_input.split(',')]
                        valid_indices = sorted(set(idx for idx in indices if 0 <= idx < len(new_folders)))
                        instrument_folders_to_process = [new_folders[idx] for idx in valid_indices]
                    except ValueError:
                        print("[ERROR] Invalid input. Please enter numbers separated by commas.")
        except FileNotFoundError:
            print(f"[ERROR] Source directory not found: {chunked_outcomes_parent_dir}")

    if not instrument_folders_to_process:
        print("[INFO] No folders selected or found for processing.")
        return

    # --- Main Execution Loop ---
    # The script processes each instrument folder one by one (serially).
    # Parallelism is used *within* each instrument to process its many chunks.
    print(f"\n[QUEUE] Queued {len(instrument_folders_to_process)} folder(s): {instrument_folders_to_process}")
    for instrument_name in instrument_folders_to_process:
        instrument_chunk_dir = os.path.join(chunked_outcomes_parent_dir, instrument_name)
        definitions_path = os.path.join(combinations_dir, f"{instrument_name}.csv")

        try:
            print(f"\n{'='*25}\nGenerating combinations for: {instrument_name}\n{'='*25}")
            # Call the parallel orchestrator function for the current instrument.
            strategy_definitions = generate_strategy_definitions_parallel(instrument_chunk_dir)

            if not strategy_definitions.empty:
                strategy_definitions.to_csv(definitions_path, index=False)
                print(f"\n[SUCCESS] Saved {len(strategy_definitions)} unique combinations to: {definitions_path}")
            else:
                print(f"\n[INFO] No valid combinations were found for {instrument_name}.")
        except Exception as e:
            print(f"\n[FATAL ERROR] A critical error occurred while processing {instrument_name}.")
            traceback.print_exc()

    print("\n" + "="*50 + "\n[COMPLETE] All combination generation tasks are finished.")


if __name__ == "__main__":
    main()
    
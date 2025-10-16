# platinum_combinations_generator.py (Updated for chunked_outcomes)

"""
Platinum Layer - Stage 1: The Combination Generator (The Architect)

This script is the first and foundational stage of the Platinum discovery engine.
Its purpose is to scan the vast, enriched trade data from the Silver Layer and
discover every unique strategy "blueprint" that exists within it.

It transforms an infinite problem into a finite one by using a technique called
discretization (binning). Instead of treating every possible SL/TP placement
as unique, it groups them into finite buckets (e.g., "TP placed 70-80% of the
way to a resistance level").

The output is a master list of all unique blueprints, which defines the complete
search space for the subsequent Platinum stages to analyze.
"""

import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import itertools
import gc

# --- CONFIGURATION (No specific configuration needed for this script) ---

def generate_strategy_definitions(chunked_outcomes_dir):
    """
    Scans a directory of pre-chunked, enriched outcome files to generate a
    master list of all unique, binned strategy definitions.

    Args:
        chunked_outcomes_dir (str): The path to the directory containing the
                                    CSV chunk files for a single instrument.

    Returns:
        pd.DataFrame: A DataFrame containing all unique strategy blueprints
                      found in the data, or an empty DataFrame if none are found.
    """
    print("Scanning chunked outcomes to generate binned strategy definitions...")
    
    # Use sets to automatically handle duplicates and improve performance
    semi_tp_binned_combos = set()
    semi_sl_binned_combos = set()
    
    try:
        chunk_files = [os.path.join(chunked_outcomes_dir, f) for f in os.listdir(chunked_outcomes_dir) if f.endswith('.csv')]
        if not chunk_files:
            print("❌ No chunk files found in the directory.")
            return pd.DataFrame()
    except FileNotFoundError:
        print(f"❌ Chunk directory not found: {chunked_outcomes_dir}")
        return pd.DataFrame()

    # --- Discover all potential placement levels efficiently ---
    # Read only the header of the first chunk to identify all available relational
    # positioning columns without loading the whole file into memory.
    header_df = pd.read_csv(chunk_files[0], nrows=0)
    placement_cols = sorted([c.replace('sl_placement_pct_to_', '') 
                             for c in header_df.columns 
                             if c.startswith('sl_placement_pct_to_')])

    print(f"Found {len(placement_cols)} potential placement levels to bin.")

    def to_bin(series):
        """
        Discretizes a continuous percentage series into integer bins.
        Example: A value of 0.85 (85%) becomes bin 8.
        """
        return np.floor(series * 10).astype('Int64')

    # --- Iterate through the pre-made chunk files ---
    for chunk_path in tqdm(chunk_files, desc="Scanning for Combinations"):
        chunk = pd.read_csv(chunk_path)
        # Round ratios to handle potential floating point inaccuracies
        chunk['sl_ratio'] = chunk['sl_ratio'].round(5)
        chunk['tp_ratio'] = chunk['tp_ratio'].round(5)
            
        # For each possible indicator level, find the unique binned combinations
        for level in placement_cols:
            sl_pct_col = f"sl_placement_pct_to_{level}"
            tp_pct_col = f"tp_placement_pct_to_{level}"

            # --- Find "Semi-Dynamic-SL-Binned" strategies ---
            # These have a SL defined by a binned distance to a level,
            # and a TP defined by a fixed percentage ratio.
            if sl_pct_col in chunk.columns:
                chunk['sl_bin'] = to_bin(chunk[sl_pct_col])
                # Filter for a reasonable range of bins to reduce noise
                sl_binned_chunk = chunk[(chunk['sl_bin'] >= -10) & (chunk['sl_bin'] < 20)]
                for combo in sl_binned_chunk[['tp_ratio', 'sl_bin']].drop_duplicates().itertuples(index=False):
                    semi_sl_binned_combos.add((level, combo.sl_bin, combo.tp_ratio))

            # --- Find "Semi-Dynamic-TP-Binned" strategies ---
            # These have a TP defined by a binned distance to a level,
            # and a SL defined by a fixed percentage ratio.
            if tp_pct_col in chunk.columns:
                chunk['tp_bin'] = to_bin(chunk[tp_pct_col])
                tp_binned_chunk = chunk[(chunk['tp_bin'] >= -10) & (chunk['tp_bin'] < 20)]
                for combo in tp_binned_chunk[['sl_ratio', 'tp_bin']].drop_duplicates().itertuples(index=False):
                    semi_tp_binned_combos.add((level, combo.tp_bin, combo.sl_ratio))
    
    # --- Consolidate discovered combinations into a final DataFrame ---
    definitions = []
    for sl_level, sl_bin, tp_ratio in semi_sl_binned_combos:
        definitions.append({'type': 'Semi-Dynamic-SL-Binned', 'sl_def': sl_level, 'sl_bin': sl_bin, 'tp_def': tp_ratio, 'tp_bin': np.nan})
    for tp_level, tp_bin, sl_ratio in semi_tp_binned_combos:
        definitions.append({'type': 'Semi-Dynamic-TP-Binned', 'sl_def': sl_ratio, 'sl_bin': np.nan, 'tp_def': tp_level, 'tp_bin': tp_bin})

    if not definitions:
        return pd.DataFrame()

    return pd.DataFrame(definitions).drop_duplicates().reset_index(drop=True)

if __name__ == "__main__":
    # --- Define Project Directory Structure ---
    core_dir = os.path.dirname(os.path.abspath(__file__))
    # Input: The parent directory containing all instrument-specific chunk folders
    chunked_outcomes_parent_dir = os.path.abspath(os.path.join(core_dir, '..', 'silver_data', 'chunked_outcomes'))
    # Output: The directory where the master combination lists will be saved
    combinations_dir = os.path.abspath(os.path.join(core_dir, '..', 'platinum_data', 'combinations'))
    os.makedirs(combinations_dir, exist_ok=True)

    try:
        # Get the list of instrument folders (e.g., 'XAUUSD15', 'EURUSD60')
        instrument_folders = [d for d in os.listdir(chunked_outcomes_parent_dir) if os.path.isdir(os.path.join(chunked_outcomes_parent_dir, d))]
    except FileNotFoundError:
        print(f"❌ Source directory not found: {chunked_outcomes_parent_dir}")
        instrument_folders = []

    if not instrument_folders:
        print("❌ No instrument chunk folders found in 'silver_data/chunked_outcomes'.")
    else:
        # --- Main Loop: Process each instrument folder ---
        for instrument_name in instrument_folders:
            # The input is the specific directory for this instrument's chunks
            instrument_chunk_dir = os.path.join(chunked_outcomes_parent_dir, instrument_name)
            # The output filename is based on the instrument name
            definitions_path = os.path.join(combinations_dir, f"{instrument_name}.csv")

            # --- Resumability Check ---
            # Skip generation if the output file already exists.
            if os.path.exists(definitions_path):
                print(f"ℹ️ Combinations file already exists for {instrument_name}. Skipping generation.")
                continue

            try:
                print(f"\n{'='*25}\nGenerating combinations for: {instrument_name}\n{'='*25}")
                strategy_definitions = generate_strategy_definitions(instrument_chunk_dir)

                if not strategy_definitions.empty:
                    strategy_definitions.to_csv(definitions_path, index=False)
                    print(f"\n✅ Success! Saved {len(strategy_definitions)} combinations to: {definitions_path}")
                else:
                    print("\nℹ️ No valid combinations were found.")

            except Exception as e:
                print(f"\n❌ FAILED to process {instrument_name}. Error: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "="*50 + "\n✅ All combination generation complete.")
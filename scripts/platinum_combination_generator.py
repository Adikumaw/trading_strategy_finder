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

By default, this script focuses on discovering "semi-dynamic" strategies (e.g.,
a dynamic SL combined with a static TP ratio) to manage computational
complexity. Logic to discover "fully dynamic" strategies (dynamic SL + dynamic TP)
is included but commented out, as it can lead to a combinatorial explosion,
generating a massive number of blueprints that may be computationally expensive
to test in subsequent stages.

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
    Discovers all unique strategy blueprints from an instrument's enriched data chunks.

    This function iterates through a directory of pre-chunked, enriched outcome
    files. It discretizes the continuous 'placement_pct' features into integer
    bins and identifies every unique combination of stop-loss and take-profit
    definitions. It primarily focuses on "semi-dynamic" strategies, where one
    parameter (SL or TP) is dynamic (binned relative to a market level) and the
    other is static (a fixed ratio).

    Args:
        chunked_outcomes_dir (str): The path to the directory containing the
                                    CSV chunk files for a single instrument.

    Returns:
        pd.DataFrame: A DataFrame containing all unique strategy blueprints
                      found in the data. Each row represents a distinct strategy
                      to be tested. Returns an empty DataFrame if none are found.
    """
    print("Scanning chunked outcomes to generate binned strategy definitions...")
    
    # Use sets to automatically handle duplicates and improve performance.
    # This is far more efficient than appending to a list and dropping duplicates later.
    semi_tp_binned_combos = set()
    semi_sl_binned_combos = set()
    
    # --- OPTIONAL: Set for Fully Dynamic Strategies ---
    # fully_dynamic_combos = set()

    try:
        chunk_files = [os.path.join(chunked_outcomes_dir, f) for f in os.listdir(chunked_outcomes_dir) if f.endswith('.csv')]
        if not chunk_files:
            print("❌ No chunk files found in the directory.")
            return pd.DataFrame()
    except FileNotFoundError:
        print(f"❌ Chunk directory not found: {chunked_outcomes_dir}")
        return pd.DataFrame()

    # --- Step 1: Discover all potential placement levels efficiently ---
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

        This helper function takes a pandas Series of floating-point values
        (representing placement percentages) and converts them into discrete
        integer bins by multiplying by 10 and taking the floor.
        Example: A value of 0.85 (85%) becomes bin 8.

        Args:
            series (pd.Series): The input Series of continuous data.

        Returns:
            pd.Series: A Series of discrete integer bins.
        """
        return np.floor(series * 10).astype('Int64')

    # --- Step 2: Iterate through all chunks to find unique combinations ---
    for chunk_path in tqdm(chunk_files, desc="Scanning for Combinations"):
        chunk = pd.read_csv(chunk_path)
        # Round ratios to handle potential floating point inaccuracies
        if 'sl_ratio' in chunk.columns:
            chunk['sl_ratio'] = chunk['sl_ratio'].round(5)
        if 'tp_ratio' in chunk.columns:
            chunk['tp_ratio'] = chunk['tp_ratio'].round(5)
            
        # For each possible indicator level, find the unique binned combinations
        for level in placement_cols:
            sl_pct_col = f"sl_placement_pct_to_{level}"
            tp_pct_col = f"tp_placement_pct_to_{level}"

            # --- Find "Semi-Dynamic-SL-Binned" strategies ---
            # These have a SL defined by a binned distance to a level,
            # and a TP defined by a fixed percentage ratio.
            if sl_pct_col in chunk.columns and 'tp_ratio' in chunk.columns:
                chunk['sl_bin'] = to_bin(chunk[sl_pct_col])
                # Filter for a reasonable range of bins to reduce noise
                sl_binned_chunk = chunk[(chunk['sl_bin'] >= -10) & (chunk['sl_bin'] < 20)]
                for combo in sl_binned_chunk[['tp_ratio', 'sl_bin']].drop_duplicates().itertuples(index=False):
                    semi_sl_binned_combos.add((level, combo.sl_bin, combo.tp_ratio))

            # --- Find "Semi-Dynamic-TP-Binned" strategies ---
            # These have a TP defined by a binned distance to a level,
            # and a SL defined by a fixed percentage ratio.
            if tp_pct_col in chunk.columns and 'sl_ratio' in chunk.columns:
                chunk['tp_bin'] = to_bin(chunk[tp_pct_col])
                tp_binned_chunk = chunk[(chunk['tp_bin'] >= -10) & (chunk['tp_bin'] < 20)]
                for combo in tp_binned_chunk[['sl_ratio', 'tp_bin']].drop_duplicates().itertuples(index=False):
                    semi_tp_binned_combos.add((level, combo.tp_bin, combo.sl_ratio))
        
        # --- OPTIONAL: Find "Fully-Dynamic-Binned" strategies ---
        # This section discovers blueprints where both SL and TP are dynamic
        # (i.e., based on market levels).
        # WARNING: This creates a massive number of combinations and can
        # significantly increase processing time in subsequent scripts.
        # Uncomment the code block below to enable this discovery.
        # 
        # # Use itertools.product to test every SL level against every TP level
        # for sl_level, tp_level in itertools.product(placement_cols, placement_cols):
        #     # Optional: Prevent using the same level for both SL and TP
        #     if sl_level == tp_level:
        #         continue

        #     sl_pct_col = f"sl_placement_pct_to_{sl_level}"
        #     tp_pct_col = f"tp_placement_pct_to_{tp_level}"

        #     # Ensure both required placement columns exist in the current chunk
        #     if sl_pct_col in chunk.columns and tp_pct_col in chunk.columns:
        #         chunk['sl_bin'] = to_bin(chunk[sl_pct_col])
        #         chunk['tp_bin'] = to_bin(chunk[tp_pct_col])
                
        #         # Filter for a reasonable range of bins
        #         binned_chunk = chunk[
        #             (chunk['sl_bin'] >= -10) & (chunk['sl_bin'] < 20) &
        #             (chunk['tp_bin'] >= -10) & (chunk['tp_bin'] < 20)
        #         ].dropna(subset=['sl_bin', 'tp_bin'])
                
        #         # Find all unique (sl_bin, tp_bin) pairs for this specific (sl_level, tp_level) pair
        #         for combo in binned_chunk[['sl_bin', 'tp_bin']].drop_duplicates().itertuples(index=False):
        #             fully_dynamic_combos.add((sl_level, int(combo.sl_bin), tp_level, int(combo.tp_bin)))


    # --- Step 3: Consolidate all discovered combinations into a final DataFrame ---
    definitions = []
    # Consolidate semi-dynamic blueprints where SL is dynamic and TP is static
    for sl_level, sl_bin, tp_ratio in semi_sl_binned_combos:
        definitions.append({'type': 'Semi-Dynamic-SL-Binned', 'sl_def': sl_level, 'sl_bin': sl_bin, 'tp_def': tp_ratio, 'tp_bin': np.nan})
    # Consolidate semi-dynamic blueprints where TP is dynamic and SL is static
    for tp_level, tp_bin, sl_ratio in semi_tp_binned_combos:
        definitions.append({'type': 'Semi-Dynamic-TP-Binned', 'sl_def': sl_ratio, 'sl_bin': np.nan, 'tp_def': tp_level, 'tp_bin': tp_bin})

    # --- OPTIONAL: Consolidate Fully Dynamic Blueprints ---
    # Uncomment the block below if you enabled the fully dynamic discovery.
    # 
    # for sl_level, sl_bin, tp_level, tp_bin in fully_dynamic_combos:
    #     definitions.append({'type': 'Fully-Dynamic-Binned', 'sl_def': sl_level, 'sl_bin': sl_bin, 'tp_def': tp_level, 'tp_bin': tp_bin})


    if not definitions:
        print("\nℹ️ No valid combinations were found.")
        return pd.DataFrame()

    # Perform a final de-duplication and reset the index for a clean output file
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
                    print("\nℹ️ No valid combinations were found for this instrument.")

            except Exception as e:
                print(f"\n❌ FAILED to process {instrument_name}. Error: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "="*50 + "\n✅ All combination generation complete.")
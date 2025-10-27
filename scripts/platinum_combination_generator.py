# platinum_combinations_generator.py (V4 - Final Architecture with BPS)

"""
Platinum Layer - Stage 1: The Combination Generator (The Architect)

This script is the first and foundational stage of the Platinum discovery engine.
Its purpose is to scan the vast, enriched trade data from the Silver Layer and
discover every unique strategy "blueprint" that exists within it.

It transforms an infinite problem into a finite one by using a technique called
discretization (binning). Instead of treating every possible SL/TP placement
as unique, it groups them into finite buckets.

This version is enhanced to discover two distinct types of dynamic strategies:
1.  **Proportional Placement (Percentage):** Strategies where the SL/TP is placed
    at a specific percentage of the way to a market level (e.g., "TP at 80% of
    the distance to resistance").
2.  **Fixed Buffer (Basis Points):** Strategies where the SL/TP is placed at a
    standardized fixed distance from a market level (e.g., "SL 15 bps below
    the 50-period SMA").

The output is a master list of all unique blueprints, defining the complete
search space for the subsequent Platinum stages.
"""

import pandas as pd
import os
import numpy as np
from tqdm import tqdm
# import itertools
# import gc
import sys

# --- CONFIGURATION ---
# The size of each bin for basis points. e.g., 5 means bins will be -5 to 0, 0 to 5, etc.
BPS_BIN_SIZE = 5.0

def generate_strategy_definitions(chunked_outcomes_dir):
    """
    Discovers all unique strategy blueprints from an instrument's enriched data chunks.

    This function iterates through a directory of pre-chunked, enriched outcome
    files. It discretizes both the `_placement_pct` and `_dist_bps` features into
    bins and identifies every unique "semi-dynamic" strategy combination, where one
    parameter (SL or TP) is dynamic (binned) and the other is static (a fixed ratio).

    Args:
        chunked_outcomes_dir (str): The path to the directory containing the
                                    CSV chunk files for a single instrument.

    Returns:
        pd.DataFrame: A DataFrame containing all unique strategy blueprints
                      found in the data. Returns an empty DataFrame if none are found.
    """
    print("Scanning chunked outcomes to generate binned strategy definitions...")
    
    # Use sets for automatic de-duplication and performance.
    semi_sl_pct_binned_combos, semi_tp_pct_binned_combos = set(), set()
    semi_sl_bps_binned_combos, semi_tp_bps_binned_combos = set(), set()
    
    # --- OPTIONAL: Set for Fully Dynamic Strategies ---
    # fully_dynamic_combos = set()

    try:
        chunk_files = [os.path.join(chunked_outcomes_dir, f) for f in os.listdir(chunked_outcomes_dir) if f.endswith('.csv')]
        if not chunk_files:
            print("[ERROR] No chunk files found in the directory.")
            return pd.DataFrame()
    except FileNotFoundError:
        print(f"[ERROR] Chunk directory not found: {chunked_outcomes_dir}")
        return pd.DataFrame()

    header_df = pd.read_csv(chunk_files[0], nrows=0)
    pct_levels = sorted([c.replace('sl_placement_pct_to_', '') for c in header_df.columns if c.startswith('sl_placement_pct_to_')])
    bps_levels = sorted([c.replace('sl_dist_to_', '').replace('_bps', '') for c in header_df.columns if c.startswith('sl_dist_to_')])
    all_levels = sorted(list(set(pct_levels) | set(bps_levels)))

    print(f"Found {len(all_levels)} potential placement levels to bin for both PCT and BPS strategies.")

    def to_bin_pct(series):
        """Discretizes a percentage series into 10% bins (e.g., 0.85 -> bin 8)."""
        return np.floor(series * 10).astype('Int64')
    
    def to_bin_bps(series):
        """Discretizes a BPS series into bins of a fixed size (e.g., 17.3 -> bin 15)."""
        return (np.floor(series / BPS_BIN_SIZE) * BPS_BIN_SIZE).astype('Int64')

    for chunk_path in tqdm(chunk_files, desc="Scanning for Combinations"):
        chunk = pd.read_csv(chunk_path)
        if 'sl_ratio' in chunk.columns: chunk['sl_ratio'] = chunk['sl_ratio'].round(5)
        if 'tp_ratio' in chunk.columns: chunk['tp_ratio'] = chunk['tp_ratio'].round(5)
            
        for level in all_levels:
            sl_pct_col, tp_pct_col = f"sl_placement_pct_to_{level}", f"tp_placement_pct_to_{level}"
            sl_bps_col, tp_bps_col = f"sl_dist_to_{level}_bps", f"tp_dist_to_{level}_bps"  

            # --- Find "Semi-Dynamic-SL-Pct-Binned" strategies ---
            if sl_pct_col in chunk.columns and 'tp_ratio' in chunk.columns:
                chunk['sl_bin_pct'] = to_bin_pct(chunk[sl_pct_col])
                sl_binned_chunk = chunk[chunk['sl_bin_pct'].between(-10, 19, inclusive='both')]
                for combo in sl_binned_chunk[['tp_ratio', 'sl_bin_pct']].drop_duplicates().itertuples(index=False):
                    semi_sl_pct_binned_combos.add((level, combo.sl_bin_pct, combo.tp_ratio))

            # --- Find "Semi-Dynamic-TP-Pct-Binned" strategies ---
            if tp_pct_col in chunk.columns and 'sl_ratio' in chunk.columns:
                chunk['tp_bin_pct'] = to_bin_pct(chunk[tp_pct_col])
                tp_binned_chunk = chunk[chunk['tp_bin_pct'].between(-10, 19, inclusive='both')]
                for combo in tp_binned_chunk[['sl_ratio', 'tp_bin_pct']].drop_duplicates().itertuples(index=False):
                    semi_tp_pct_binned_combos.add((level, combo.tp_bin_pct, combo.sl_ratio))
            
            # --- Find "Semi-Dynamic-SL-BPS-Binned" strategies (Buffer style) ---
            if sl_bps_col in chunk.columns and 'tp_ratio' in chunk.columns:
                chunk['sl_bin_bps'] = to_bin_bps(chunk[sl_bps_col])
                sl_binned_chunk = chunk[chunk['sl_bin_bps'].between(-50, 49, inclusive='both')]
                for combo in sl_binned_chunk[['tp_ratio', 'sl_bin_bps']].drop_duplicates().itertuples(index=False):
                    semi_sl_bps_binned_combos.add((level, combo.sl_bin_bps, combo.tp_ratio))

            # --- Find "Semi-Dynamic-TP-BPS-Binned" strategies (Buffer style) ---
            if tp_bps_col in chunk.columns and 'sl_ratio' in chunk.columns:
                chunk['tp_bin_bps'] = to_bin_bps(chunk[tp_bps_col])
                tp_binned_chunk = chunk[chunk['tp_bin_bps'].between(-50, 49, inclusive='both')]
                for combo in tp_binned_chunk[['sl_ratio', 'tp_bin_bps']].drop_duplicates().itertuples(index=False):
                    semi_tp_bps_binned_combos.add((level, combo.tp_bin_bps, combo.sl_ratio))
        
        # --- OPTIONAL: Find "Fully-Dynamic-Binned" strategies ---
        # This section discovers blueprints where both SL and TP are dynamic.
        # WARNING: This can create a massive number of combinations. Uncomment to enable.
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
    for sl_level, sl_bin, tp_ratio in semi_sl_pct_binned_combos:
        definitions.append({'type': 'Semi-Dynamic-SL-Pct-Binned', 'sl_def': sl_level, 'sl_bin': sl_bin, 'tp_def': tp_ratio, 'tp_bin': np.nan})
    for tp_level, tp_bin, sl_ratio in semi_tp_pct_binned_combos:
        definitions.append({'type': 'Semi-Dynamic-TP-Pct-Binned', 'sl_def': sl_ratio, 'sl_bin': np.nan, 'tp_def': tp_level, 'tp_bin': tp_bin})

    for sl_level, sl_bin, tp_ratio in semi_sl_bps_binned_combos:
        definitions.append({'type': 'Semi-Dynamic-SL-BPS-Binned', 'sl_def': sl_level, 'sl_bin': sl_bin, 'tp_def': tp_ratio, 'tp_bin': np.nan})
    for tp_level, tp_bin, sl_ratio in semi_tp_bps_binned_combos:
        definitions.append({'type': 'Semi-Dynamic-TP-BPS-Binned', 'sl_def': sl_ratio, 'sl_bin': np.nan, 'tp_def': tp_level, 'tp_bin': tp_bin})
        
    # --- OPTIONAL: Consolidate Fully Dynamic Blueprints ---
    # Uncomment the block below if you enabled the fully dynamic discovery.
    # 
    # for sl_level, sl_bin, tp_level, tp_bin in fully_dynamic_combos:
    #     definitions.append({'type': 'Fully-Dynamic-Binned', 'sl_def': sl_level, 'sl_bin': sl_bin, 'tp_def': tp_level, 'tp_bin': tp_bin})


    if not definitions:
        print("\n[INFO] No valid combinations were found.")
        return pd.DataFrame()

    return pd.DataFrame(definitions).drop_duplicates().reset_index(drop=True)

if __name__ == "__main__":
    """
    Main execution block for the Platinum Combination Generator.
    Supports two modes for flexibility.
    """
    core_dir = os.path.dirname(os.path.abspath(__file__))
    chunked_outcomes_parent_dir = os.path.abspath(os.path.join(core_dir, '..', 'silver_data', 'chunked_outcomes'))
    combinations_dir = os.path.abspath(os.path.join(core_dir, '..', 'platinum_data', 'combinations'))
    os.makedirs(combinations_dir, exist_ok=True)

    instrument_folders_to_process = []
    target_file_arg = sys.argv[1] if len(sys.argv) > 1 else None
    
    if target_file_arg:
        # --- Targeted Mode ---
        instrument_name = target_file_arg.replace('.csv', '')
        print(f"[TARGET] Targeted Mode: Processing single instrument folder '{instrument_name}'")
        instrument_chunk_dir_check = os.path.join(chunked_outcomes_parent_dir, instrument_name)
        if not os.path.isdir(instrument_chunk_dir_check):
            print(f"[ERROR] Target instrument folder not found: {instrument_name}")
        else:
            instrument_folders_to_process = [instrument_name]
    else:
        # --- Interactive Mode ---
        print("[SCAN] Interactive Mode: Scanning for all new instrument folders...")
        try:
            all_instrument_folders = sorted([d for d in os.listdir(chunked_outcomes_parent_dir) if os.path.isdir(os.path.join(chunked_outcomes_parent_dir, d))])
            new_folders = [f for f in all_instrument_folders if not os.path.exists(os.path.join(combinations_dir, f"{f}.csv"))]
            
            if not new_folders:
                print("[INFO] No new instrument folders to process.")
            else:
                print("\n--- Select Folder(s) to Process ---")
                for i, f in enumerate(new_folders): print(f"  [{i+1}] {f}")
                print("\nYou can select multiple folders by entering numbers separated by commas (e.g., 1,3,5)")
                
                user_input = input("Enter number(s) to process: ").strip()
                if user_input:
                    try:
                        selected_indices = [int(i.strip()) - 1 for i in user_input.split(',')]
                        valid_indices = sorted(list(set(idx for idx in selected_indices if 0 <= idx < len(new_folders))))
                        instrument_folders_to_process = [new_folders[idx] for idx in valid_indices]
                    except ValueError:
                        print("[ERROR] Invalid input. Please enter numbers separated by commas.")
        except FileNotFoundError:
            print(f"[ERROR] Source directory not found: {chunked_outcomes_parent_dir}")

    if not instrument_folders_to_process:
        print("[INFO] No folders selected or found for processing.")
    else:
        print(f"\n[INFO] Queued {len(instrument_folders_to_process)} folder(s) for processing: {instrument_folders_to_process}")
        
        for instrument_name in instrument_folders_to_process:
            instrument_chunk_dir = os.path.join(chunked_outcomes_parent_dir, instrument_name)
            definitions_path = os.path.join(combinations_dir, f"{instrument_name}.csv")

            try:
                print(f"\n{'='*25}\nGenerating combinations for: {instrument_name}\n{'='*25}")
                strategy_definitions = generate_strategy_definitions(instrument_chunk_dir)

                if not strategy_definitions.empty:
                    strategy_definitions.to_csv(definitions_path, index=False)
                    print(f"\n[SUCCESS] Saved {len(strategy_definitions)} combinations to: {definitions_path}")
                else:
                    print(f"\n[INFO] No valid combinations were found for {instrument_name}.")
            except Exception as e:
                print(f"\n[ERROR] FAILED to process {instrument_name}. Error: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "="*50 + "\n[SUCCESS] All combination generation complete.")

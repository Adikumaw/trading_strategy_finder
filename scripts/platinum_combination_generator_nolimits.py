# platinum_combinations_generator_nolimits.py

import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import gc
from multiprocessing import Pool, cpu_count

# --- CONFIGURATION ---
MAX_CPU_USAGE = max(1, cpu_count() - 2)

def generate_strategy_definitions_nolimits(outcomes_df):
    """
    'No Limits' version that scans the entire in-memory DataFrame to find all
    unique, actionable strategy definitions with directionality.
    """
    semi_tp_binned_combos = set()
    semi_sl_binned_combos = set()
    
    placement_cols = sorted([c.replace('sl_placement_pct_to_', '') 
                             for c in outcomes_df.columns 
                             if c.startswith('sl_placement_pct_to_')])
    
    print(f"  Found {len(placement_cols)} potential placement levels to bin.")

    def to_bin(series):
        return np.floor(series * 10).astype('Int64')

    outcomes_df['sl_ratio'] = outcomes_df['sl_ratio'].round(5)
    outcomes_df['tp_ratio'] = outcomes_df['tp_ratio'].round(5)
    
    # Process all levels in a vectorized way where possible
    for level in tqdm(placement_cols, desc="  Generating Binned Combinations", leave=False):
        sl_pct_col = f"sl_placement_pct_to_{level}"
        tp_pct_col = f"tp_placement_pct_to_{level}"

        if sl_pct_col in outcomes_df.columns:
            sl_bin = to_bin(outcomes_df[sl_pct_col])
            valid_sl_bins = outcomes_df[(sl_bin >= -10) & (sl_bin < 20)]
            for combo in valid_sl_bins[['tp_ratio']].assign(sl_bin=sl_bin).drop_duplicates().itertuples(index=False):
                semi_sl_binned_combos.add((level, combo.sl_bin, combo.tp_ratio))

        if tp_pct_col in outcomes_df.columns:
            tp_bin = to_bin(outcomes_df[tp_pct_col])
            valid_tp_bins = outcomes_df[(tp_bin >= -10) & (tp_bin < 20)]
            for combo in valid_tp_bins[['sl_ratio']].assign(tp_bin=tp_bin).drop_duplicates().itertuples(index=False):
                semi_tp_binned_combos.add((level, combo.tp_bin, combo.sl_ratio))

    definitions = []
    for sl_level, sl_bin, tp_ratio in semi_sl_binned_combos:
        definitions.append({'type': 'Semi-Dynamic-SL-Binned', 'sl_def': sl_level, 'sl_bin': sl_bin, 'tp_def': tp_ratio, 'tp_bin': np.nan})
    for tp_level, tp_bin, sl_ratio in semi_tp_binned_combos:
        definitions.append({'type': 'Semi-Dynamic-TP-Binned', 'sl_def': sl_ratio, 'sl_bin': np.nan, 'tp_def': tp_level, 'tp_bin': tp_bin})

    if not definitions:
        return pd.DataFrame()

    return pd.DataFrame(definitions).drop_duplicates().reset_index(drop=True)

def process_file_in_parallel(file_path_tuple):
    """
    Wrapper function for multiprocessing: loads the file, generates definitions, and saves.
    """
    outcomes_path, definitions_path = file_path_tuple
    fname = os.path.basename(outcomes_path)

    try:
        # --- SPEED OPTIMIZATION: Load the entire file into memory ---
        print(f"Loading {fname} into memory...")
        outcomes_df = pd.read_csv(outcomes_path)
        
        print(f"Generating combinations for {fname}...")
        strategy_definitions = generate_strategy_definitions_nolimits(outcomes_df)
        
        if not strategy_definitions.empty:
            strategy_definitions.to_csv(definitions_path, index=False)
            return f"✅ Success! Generated {len(strategy_definitions)} combinations for {fname}."
        else:
            return f"ℹ️ No valid combinations found for {fname}."
            
    except Exception as e:
        return f"❌ FAILED to process {fname}. Error: {e}"

if __name__ == "__main__":
    core_dir = os.path.dirname(os.path.abspath(__file__))
    silver_outcomes_dir = os.path.abspath(os.path.join(core_dir, '..', 'silver_data', 'outcomes'))
    combinations_dir = os.path.abspath(os.path.join(core_dir, '..', 'platinum_data', 'combinations'))
    os.makedirs(combinations_dir, exist_ok=True)

    try:
        all_files = [f for f in os.listdir(silver_outcomes_dir) if f.endswith('.csv')]
    except FileNotFoundError:
        print(f"❌ Error: The directory '{silver_outcomes_dir}' was not found.")
        all_files = []

    if not all_files:
        print("❌ No silver outcome files found to generate combinations from.")
    else:
        files_to_process = [f for f in all_files if not os.path.exists(os.path.join(combinations_dir, f))]
        
        if not files_to_process:
            print("✅ All combination files already exist. Nothing to do.")
        else:
            print(f"Found {len(files_to_process)} new outcome file(s) to process.")
            
            tasks = [(os.path.join(silver_outcomes_dir, fname), os.path.join(combinations_dir, fname)) for fname in files_to_process]
            
            num_processes = min(MAX_CPU_USAGE, len(tasks))
            print(f"\n🚀 Starting parallel processing with {num_processes} workers...")
            
            with Pool(processes=num_processes) as pool:
                # We don't use tqdm here as the parallel processes will print their own status
                results = pool.map(process_file_in_parallel, tasks)

            print("\n--- Processing Summary ---")
            for res in results:
                print(res)

    print("\n" + "="*50 + "\n✅ All combination generation complete.")
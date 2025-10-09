# platinum_combinations_generator.py (Corrected)
import pandas as pd
import os
from tqdm import tqdm
import numpy as np

# --- CONFIGURATION ---
OUTCOMES_CHUNK_SIZE = 1_000_000

def generate_strategy_definitions(outcomes_path):
    """
    Scans the outcomes file to generate all actionable strategy definitions,
    now including binned percentage-based placements with directionality.
    """
    print("Scanning outcomes to generate binned and ratio-based strategy definitions...")
    
    semi_tp_binned_combos = set()
    semi_sl_binned_combos = set()
    
    outcomes_iterator = pd.read_csv(outcomes_path, chunksize=OUTCOMES_CHUNK_SIZE)
    
    header_df = pd.read_csv(outcomes_path, nrows=0)
    placement_cols = sorted([c.replace('sl_placement_pct_to_', '') 
                             for c in header_df.columns 
                             if c.startswith('sl_placement_pct_to_')])

    print(f"Found {len(placement_cols)} potential placement levels to bin.")

    def to_bin(series):
        return np.floor(series * 10).astype('Int64')

    for chunk in tqdm(outcomes_iterator, desc="Scanning for Binned Combinations"):
        chunk['sl_ratio'] = chunk['sl_ratio'].round(5)
        chunk['tp_ratio'] = chunk['tp_ratio'].round(5)
            
        for level in placement_cols:
            sl_pct_col = f"sl_placement_pct_to_{level}"
            tp_pct_col = f"tp_placement_pct_to_{level}"

            if sl_pct_col in chunk.columns:
                chunk['sl_bin'] = to_bin(chunk[sl_pct_col])
                # --- THE FIX ---
                # Accept a wider, directional range of bins.
                # Bins from -10 to 19 allow for placements from -100% (behind entry) to +200% (beyond indicator).
                sl_binned_chunk = chunk[(chunk['sl_bin'] >= -10) & (chunk['sl_bin'] < 20)]
                
                for combo in sl_binned_chunk[['tp_ratio', 'sl_bin']].drop_duplicates().itertuples(index=False):
                    semi_sl_binned_combos.add((level, combo.sl_bin, combo.tp_ratio))

            if tp_pct_col in chunk.columns:
                chunk['tp_bin'] = to_bin(chunk[tp_pct_col])
                # --- THE FIX ---
                # Apply the same directional range for the take-profit bins.
                tp_binned_chunk = chunk[(chunk['tp_bin'] >= -10) & (chunk['tp_bin'] < 20)]

                for combo in tp_binned_chunk[['sl_ratio', 'tp_bin']].drop_duplicates().itertuples(index=False):
                    semi_tp_binned_combos.add((level, combo.tp_bin, combo.sl_ratio))

    definitions = []
    for sl_level, sl_bin, tp_ratio in semi_sl_binned_combos:
        definitions.append({'type': 'Semi-Dynamic-SL-Binned', 'sl_def': sl_level, 'sl_bin': sl_bin, 'tp_def': tp_ratio, 'tp_bin': np.nan})
    for tp_level, tp_bin, sl_ratio in semi_tp_binned_combos:
        definitions.append({'type': 'Semi-Dynamic-TP-Binned', 'sl_def': sl_ratio, 'sl_bin': np.nan, 'tp_def': tp_level, 'tp_bin': tp_bin})

    if not definitions:
        return pd.DataFrame()

    return pd.DataFrame(definitions).drop_duplicates().reset_index(drop=True)

# ... (main execution block is unchanged) ...
if __name__ == "__main__":
    core_dir = os.path.dirname(os.path.abspath(__file__))
    silver_outcomes_dir = os.path.abspath(os.path.join(core_dir, '..', 'silver_data', 'outcomes'))
    combinations_dir = os.path.abspath(os.path.join(core_dir, '..', 'platinum_data', 'combinations'))
    os.makedirs(combinations_dir, exist_ok=True)
    outcome_files = [f for f in os.listdir(silver_outcomes_dir) if f.endswith('.csv')]
    if not outcome_files:
        print("❌ No silver outcome files found to generate combinations from.")
    else:
        for fname in outcome_files:
            outcomes_path = os.path.join(silver_outcomes_dir, fname)
            definitions_path = os.path.join(combinations_dir, fname)
            if os.path.exists(definitions_path):
                print(f"ℹ️ Combinations file already exists for {fname}. Skipping generation.")
                continue
            try:
                print(f"\n{'='*25}\nGenerating combinations for: {fname}\n{'='*25}")
                strategy_definitions = generate_strategy_definitions(outcomes_path)
                if not strategy_definitions.empty:
                    strategy_definitions.to_csv(definitions_path, index=False)
                    print(f"\n✅ Success! Generated and saved {len(strategy_definitions)} strategy definitions to: {definitions_path}")
                else:
                    print("\nℹ️ No valid dynamic combinations were found that met the tolerance criteria.")
            except Exception as e:
                print(f"\n❌ FAILED to process {fname}. Error: {e}")
                import traceback
                traceback.print_exc()
    print("\n" + "="*50 + "\n✅ All combination generation complete.")
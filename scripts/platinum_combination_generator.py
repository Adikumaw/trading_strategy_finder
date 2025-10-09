import pandas as pd
import os
from tqdm import tqdm
import itertools
import gc

# --- CONFIGURATION ---
OUTCOMES_CHUNK_SIZE = 500_000
# KEY CHANGE: Tolerance is now in basis points (1 bps = 0.01%).
# A tolerance of 10 means we're looking for trades where the SL/TP is
# within +/- 0.10% of an indicator level. This is equivalent to the old 0.001 tolerance.
POSITIONING_TOLERANCE_BPS = 10

def generate_strategy_definitions(outcomes_path):
    """
    Scans the large outcomes file chunk by chunk to find all unique, actionable
    strategy definitions (combinations) that exist in the data.
    It focuses exclusively on Semi-Dynamic and Fully-Dynamic strategies.
    """
    print("Scanning outcomes file to generate all actionable (dynamic) strategy definitions...")

    # Use sets to efficiently store unique combinations found
    semi_tp_combos = set()
    semi_sl_combos = set()
    dynamic_combos = set()

    outcomes_iterator = pd.read_csv(outcomes_path, chunksize=OUTCOMES_CHUNK_SIZE)

    # KEY CHANGE: Get the list of positioning columns by looking for the '_bps' suffix.
    header_df = pd.read_csv(outcomes_path, nrows=0)
    positioning_cols = sorted([c.replace('sl_dist_to_', '').replace('_bps', '')
                               for c in header_df.columns
                               if c.startswith('sl_dist_to_') and c.endswith('_bps')])

    if not positioning_cols:
        print("❌ CRITICAL ERROR: No positioning columns with '_bps' suffix found. Did the Silver script run correctly?")
        return pd.DataFrame()

    print(f"Found {len(positioning_cols)} potential positioning levels to test.")

    for chunk in tqdm(outcomes_iterator, desc="Scanning for Combinations"):
        chunk['sl_ratio'] = chunk['sl_ratio'].round(5)
        chunk['tp_ratio'] = chunk['tp_ratio'].round(5)

        # Iterate through each potential positioning level
        for level in positioning_cols:
            # KEY CHANGE: Use the '_bps' column names.
            sl_dist_col = f"sl_dist_to_{level}_bps"
            tp_dist_col = f"tp_dist_to_{level}_bps"

            # Check if columns exist before using them
            if sl_dist_col not in chunk.columns or tp_dist_col not in chunk.columns:
                continue

            # Find trades where SL or TP is positioned at this level using the BPS tolerance.
            sl_positioned_chunk = chunk[chunk[sl_dist_col].abs() < POSITIONING_TOLERANCE_BPS]
            tp_positioned_chunk = chunk[chunk[tp_dist_col].abs() < POSITIONING_TOLERANCE_BPS]

            # --- Type 1: Semi-Dynamic (SL Positioned, TP Ratio) ---
            for tp_ratio in sl_positioned_chunk['tp_ratio'].unique():
                semi_sl_combos.add((level, tp_ratio))

            # --- Type 2: Semi-Dynamic (TP Positioned, SL Ratio) ---
            for sl_ratio in tp_positioned_chunk['sl_ratio'].unique():
                semi_tp_combos.add((level, sl_ratio))

        # --- Type 3: Fully-Dynamic ---
        # This needs to be done by checking pairs of levels
        for sl_level, tp_level in itertools.combinations(positioning_cols, 2):
            # KEY CHANGE: Use the '_bps' column names.
            sl_dist_col = f"sl_dist_to_{sl_level}_bps"
            tp_dist_col = f"tp_dist_to_{tp_level}_bps"
            
            if sl_dist_col not in chunk.columns or tp_dist_col not in chunk.columns:
                continue

            both_positioned_chunk = chunk[
                (chunk[sl_dist_col].abs() < POSITIONING_TOLERANCE_BPS) &
                (chunk[tp_dist_col].abs() < POSITIONING_TOLERANCE_BPS)
            ]
            if not both_positioned_chunk.empty:
                # Sort to ensure (support, resistance) is treated the same as (resistance, support)
                dynamic_combos.add(tuple(sorted((sl_level, tp_level))))

    # --- Convert sets to a final DataFrame for saving ---
    definitions = []
    for sl_level, tp_ratio in semi_sl_combos:
        definitions.append({'type': 'Semi-Dynamic-SL', 'sl_def': sl_level, 'tp_def': tp_ratio})
    for tp_level, sl_ratio in semi_tp_combos:
        definitions.append({'type': 'Semi-Dynamic-TP', 'sl_def': sl_ratio, 'tp_def': tp_level})
    for sl_level, tp_level in dynamic_combos:
        definitions.append({'type': 'Fully-Dynamic', 'sl_def': sl_level, 'tp_def': tp_level})

    if not definitions:
        return pd.DataFrame()

    return pd.DataFrame(definitions).drop_duplicates().reset_index(drop=True)

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
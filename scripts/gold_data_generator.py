import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import re
from tqdm import tqdm
import gc

def downcast_dtypes(df):
    """Downcasts numeric columns for memory efficiency."""
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
    return df

def normalize_feature_slice(feature_slice_df):
    """
    Takes a specific slice of features (for one SL/TP combo) and normalizes it.
    """
    # The 'outcome' is already in the slice, so we separate it.
    y = feature_slice_df.pop('outcome')
    
    # The rest of the logic is the same as our previous gold script
    abs_price_patterns = [
        r'^(open|high|low|close)$', r'^SMA_\d+$', r'^EMA_\d+$',
        r'^BB_(upper|lower)$', r'^(support|resistance)$', r'^ATR_level_.+$'
    ]
    abs_price_cols = []
    for pattern in abs_price_patterns:
        regex = re.compile(pattern)
        abs_price_cols.extend([col for col in feature_slice_df.columns if regex.match(col)])
    
    if 'close' in feature_slice_df.columns:
        close_series = feature_slice_df['close']
        for col in abs_price_cols:
            if col != 'close':
                feature_slice_df[f'{col}_dist_norm'] = (close_series - feature_slice_df[col]) / close_series.replace(0, np.nan)

    cols_to_drop = abs_price_cols + ['time', 'volume']
    feature_slice_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    categorical_cols = ['session', 'trend_regime', 'vol_regime']
    feature_slice_df = pd.get_dummies(feature_slice_df, columns=[c for c in categorical_cols if c in feature_slice_df.columns], drop_first=True)
    
    candle_cols = [col for col in feature_slice_df.columns if col.startswith("CDL")]
    def compress_pattern(v):
        if v >= 80: return 1.0
        elif v > 0: return 0.5
        elif v <= -80: return -1.0
        elif v < 0: return -0.5
        else: return 0.0
    for col in candle_cols:
        feature_slice_df[col] = feature_slice_df[col].fillna(0).apply(compress_pattern)
        
    one_hot_cols = [col for col in feature_slice_df.columns if any(s in col for s in ['session_', 'trend_regime_', 'vol_regime_'])]
    non_scalable_cols = set(candle_cols + one_hot_cols)
    numeric_columns_to_scale = [col for col in feature_slice_df.columns if col not in non_scalable_cols]
    
    if numeric_columns_to_scale:
        scaler = StandardScaler()
        feature_slice_df[numeric_columns_to_scale] = scaler.fit_transform(feature_slice_df[numeric_columns_to_scale].fillna(0))
    
    gold_slice = pd.concat([feature_slice_df, y], axis=1)
    return downcast_dtypes(gold_slice)

if __name__ == "__main__":
    core_dir = os.path.dirname(os.path.abspath(__file__))
    silver_features_dir = os.path.abspath(os.path.join(core_dir, '..', 'silver_data', 'features'))
    silver_outcomes_dir = os.path.abspath(os.path.join(core_dir, '..', 'silver_data', 'outcomes'))
    gold_sliced_dir = os.path.abspath(os.path.join(core_dir, '..', 'gold_data', 'by_sl_tp'))
    
    os.makedirs(gold_sliced_dir, exist_ok=True)

    feature_files = [f for f in os.listdir(silver_features_dir) if f.endswith('.csv')]

    if not feature_files:
        print("❌ No silver feature files found to process.")
    else:
        print(f"Found {len(feature_files)} feature file(s) to process.")
        for fname in feature_files:
            features_path = os.path.join(silver_features_dir, fname)
            outcomes_path = os.path.join(silver_outcomes_dir, fname)
            
            if not os.path.exists(outcomes_path):
                print(f"⚠️ SKIPPING: Corresponding outcomes file for {fname} not found."); continue

            try:
                print(f"\n{'='*25}\nProcessing: {fname}\n{'='*25}")
                
                # --- STEP 1: Load all necessary data ---
                print("Loading silver features and outcomes data...")
                features_df = pd.read_csv(features_path, parse_dates=['time'])
                outcomes_df = pd.read_csv(outcomes_path, parse_dates=['entry_time'])
                outcomes_df['outcome'] = (outcomes_df['outcome'].str.lower().str.strip() == 'win').astype(int)

                # --- STEP 2: Find all unique SL/TP combinations ---
                # Rounding is crucial to group similar floating point ratios
                outcomes_df['sl_ratio'] = outcomes_df['sl_ratio'].round(5)
                outcomes_df['tp_ratio'] = outcomes_df['tp_ratio'].round(5)
                
                unique_combos = outcomes_df[['sl_ratio', 'tp_ratio']].drop_duplicates()
                print(f"Found {len(unique_combos)} unique SL/TP combinations to process.")

                # --- STEP 3: Loop through each combo, create, normalize, and save a slice ---
                for index, combo in tqdm(unique_combos.iterrows(), total=len(unique_combos), desc="Slicing Gold Data by SL/TP"):
                    sl = combo['sl_ratio']
                    tp = combo['tp_ratio']
                    
                    # 1. Filter outcomes for this specific SL/TP combo
                    specific_outcomes = outcomes_df[(outcomes_df['sl_ratio'] == sl) & (outcomes_df['tp_ratio'] == tp)]
                    
                    # 2. Merge with features to create the training slice
                    # This is the key step that combines market context with a specific trade type
                    training_slice = pd.merge(
                        features_df,
                        specific_outcomes[['entry_time', 'outcome']],
                        left_on='time',
                        right_on='entry_time',
                        how='inner'
                    ).drop(columns=['entry_time'])
                    
                    if training_slice.empty:
                        continue
                        
                    # 3. Normalize this specific slice
                    gold_slice = normalize_feature_slice(training_slice)
                    
                    # 4. Save the slice to a dedicated file
                    slice_fname = f"sl_{sl:.5f}_tp_{tp:.5f}.csv"
                    output_path = os.path.join(gold_sliced_dir, fname.replace('.csv', ''), slice_fname)
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    gold_slice.to_csv(output_path, index=False)

                del features_df, outcomes_df, unique_combos; gc.collect()
                print(f"\n✅ Success! Sliced and normalized all gold data for {fname}.")

            except Exception as e:
                print(f"\n❌ FAILED to process {fname}. Error: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "="*50 + "\n✅ All gold data slicing complete.")
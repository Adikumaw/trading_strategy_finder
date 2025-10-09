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

def create_gold_features(features_df):
    """
    Transforms the silver features dataset into a normalized, ML-ready gold feature set.
    """
    print("Beginning transformation of silver features to gold features...")
    
    # --- 1. Transform Absolute Price Features into Relational Features ---
    # --- KEY UPDATE: Added 'ATR_level_.+' to the patterns ---
    abs_price_patterns = [
        r'^(open|high|low|close)$', 
        r'^SMA_\d+$', 
        r'^EMA_\d+$',
        r'^BB_(upper|lower)$', 
        r'^(support|resistance)$', 
        r'^ATR_level_.+$'
    ]
    abs_price_cols = []
    for pattern in abs_price_patterns:
        regex = re.compile(pattern)
        abs_price_cols.extend([col for col in features_df.columns if regex.match(col)])
    
    if 'close' in features_df.columns:
        close_series = features_df['close']
        for col in abs_price_cols:
            if col != 'close':
                features_df[f'{col}_dist_norm'] = (close_series - features_df[col]) / close_series.replace(0, np.nan)

    # --- 2. Drop Original Price Columns and Unnecessary Identifiers ---
    # We KEEP the 'time' column as our primary key.
    cols_to_drop = abs_price_cols + ['volume']
    features_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    print(f"Dropped {len(cols_to_drop)} original absolute price and other columns.")

    # --- 3. Process Categorical, Candle, and Numeric Features ---
    print("Processing categorical, candlestick, and numeric features...")
    
    categorical_cols = ['session', 'trend_regime', 'vol_regime']
    features_df = pd.get_dummies(features_df, columns=[c for c in categorical_cols if c in features_df.columns], drop_first=True)
    
    candle_cols = [col for col in features_df.columns if col.startswith("CDL")]
    def compress_pattern(v):
        if v >= 80: return 1.0
        elif v > 0: return 0.5
        elif v <= -80: return -1.0
        elif v < 0: return -0.5
        else: return 0.0
    for col in candle_cols:
        features_df[col] = features_df[col].fillna(0).apply(compress_pattern)
        
    # Correctly identify which columns to scale
    one_hot_cols = [col for col in features_df.columns if any(s in col for s in ['session_', 'trend_regime_', 'vol_regime_'])]
    non_scalable_cols = set(candle_cols + one_hot_cols + ['time'])
    numeric_columns_to_scale = [col for col in features_df.columns if col not in non_scalable_cols and features_df[col].dtype in [np.int64, np.float64, np.int32, np.float32]]
    
    print(f"Identified {len(numeric_columns_to_scale)} columns for scaling.")
    print(f"Excluding {len(non_scalable_cols)} columns (time, candle patterns, one-hot) from scaling.")
    
    scaler = StandardScaler()
    features_df[numeric_columns_to_scale] = scaler.fit_transform(features_df[numeric_columns_to_scale].fillna(0))
    
    return downcast_dtypes(features_df)

if __name__ == "__main__":
    core_dir = os.path.dirname(os.path.abspath(__file__))
    silver_features_dir = os.path.abspath(os.path.join(core_dir, '..', 'silver_data', 'features'))
    gold_features_dir = os.path.abspath(os.path.join(core_dir, '..', 'gold_data', 'features'))
    
    os.makedirs(gold_features_dir, exist_ok=True)

    feature_files = [f for f in os.listdir(silver_features_dir) if f.endswith('.csv')]

    if not feature_files:
        print("❌ No silver feature files found to process.")
    else:
        print(f"Found {len(feature_files)} feature file(s) to process.")
        for fname in feature_files:
            features_path = os.path.join(silver_features_dir, fname)
            gold_path = os.path.join(gold_features_dir, fname)
            
            try:
                print(f"\n{'='*25}\nProcessing: {fname}\n{'='*25}")
                
                # Load the single features file
                features_df = pd.read_csv(features_path, parse_dates=['time'])
                
                # Create the gold dataset (features only)
                gold_dataset = create_gold_features(features_df)
                
                # Save the final gold features dataset
                gold_dataset.to_csv(gold_path, index=False)
                print(f"\n✅ Success! Normalized 'Gold' feature data saved to: {gold_path}")

            except Exception as e:
                print(f"\n❌ FAILED to process {fname}. Error: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "="*50 + "\n✅ All gold data generation complete.")
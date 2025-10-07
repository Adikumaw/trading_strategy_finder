import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import re
from tqdm import tqdm

# --- CONFIGURATION ---
GOLD_CHUNK_SIZE = 1_000_000  # How many rows to process at a time to manage memory

def downcast_dtypes(df):
    """
    Downcasts numeric columns to more memory-efficient types (float32).
    This is the key to reducing the final file size.
    """
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
    return df

def transform_chunk(df_chunk, is_fitting_pass=False):
    """
    Applies all transformations to convert a raw silver chunk into a gold-ready feature set.
    """
    # --- 1. Separate Target Variable ---
    y_chunk = df_chunk['outcome'].apply(lambda x: 1 if x == 'win' else 0)
    features_df = df_chunk.drop(columns=['outcome'])

    # --- 2. Transform Absolute Price Features into Relational Features ---
    abs_price_patterns = [
        r'^(open|high|low|close)$', r'^SMA_\d+$', r'^EMA_\d+$',
        r'^BB_(upper|lower)$', r'^(support|resistance)$', r'^ATR_level_.+$'
    ]
    abs_price_cols = []
    for pattern in abs_price_patterns:
        regex = re.compile(pattern)
        abs_price_cols.extend([col for col in features_df.columns if regex.match(col)])
    
    if 'close' in features_df.columns:
        # Use .copy() to avoid SettingWithCopyWarning
        close_series = features_df['close'].copy()
        for col in abs_price_cols:
            if col != 'close':
                # Ensure the division is safe from zero-division errors
                features_df.loc[:, f'{col}_dist_norm'] = (close_series - features_df[col]) / close_series.replace(0, np.nan)

    # --- 3. Drop Original and Unnecessary Columns ---
    cols_to_drop = abs_price_cols + ['entry_time', 'exit_time', 'entry_price', 'sl_price', 'tp_price', 'volume']
    features_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # --- 4. Process Categorical, Candle, and Numeric Features ---
    categorical_cols_defs = {
        'trade_type': ['buy', 'sell'],
        'session': ['Asian', 'London', 'London_NY_Overlap', 'New_York'],
        'trend_regime': ['trend', 'range'],
        'vol_regime': ['high_vol', 'low_vol']
    }
    for col, categories in categorical_cols_defs.items():
        if col in features_df.columns:
            features_df[col] = pd.Categorical(features_df[col], categories=categories)
    features_df = pd.get_dummies(features_df, drop_first=True)

    candle_cols = [col for col in features_df.columns if col.startswith("CDL")]
    def compress_pattern(v):
        if v >= 80: return 1.0
        elif v > 0: return 0.5
        elif v <= -80: return -1.0
        elif v < 0: return -0.5
        else: return 0.0

    if candle_cols:
        for col in candle_cols:
            features_df[col] = features_df[col].fillna(0).apply(compress_pattern)
            
    # Return features and target separately
    return features_df, y_chunk

def process_silver_to_gold(silver_path, gold_path):
    """
    Orchestrates the two-pass, chunk-based processing of a single silver file.
    """
    print(f"\n{'='*25}\nProcessing: {os.path.basename(silver_path)}\n{'='*25}")
    
    # --- Pass 1: Fit the Scaler ---
    print("Pass 1: Fitting scaler on transformed data...")
    scaler = StandardScaler()
    final_feature_columns = []

    chunk_iterator = pd.read_csv(silver_path, chunksize=GOLD_CHUNK_SIZE)
    for chunk in tqdm(chunk_iterator, desc="Fitting Scaler"):
        features_transformed, _ = transform_chunk(chunk)
        numeric_cols = features_transformed.select_dtypes(include=np.number).columns
        scaler.partial_fit(features_transformed[numeric_cols].fillna(0))
        if not final_feature_columns:
            final_feature_columns = list(features_transformed.columns)

    # --- Pass 2: Transform and Save the Data ---
    print("\nPass 2: Transforming data and saving to gold file...")
    chunk_iterator = pd.read_csv(silver_path, chunksize=GOLD_CHUNK_SIZE)
    is_first_chunk = True

    if os.path.exists(gold_path):
        os.remove(gold_path)

    for chunk in tqdm(chunk_iterator, desc="Transforming Chunks"):
        features_transformed, y_transformed = transform_chunk(chunk)
        numeric_cols = features_transformed.select_dtypes(include=np.number).columns
        features_transformed.loc[:, numeric_cols] = scaler.transform(features_transformed[numeric_cols].fillna(0))
        
        features_aligned = features_transformed.reindex(columns=final_feature_columns, fill_value=0)
        
        # Combine with target
        processed_chunk = pd.concat([features_aligned, y_transformed], axis=1)
        
        # --- KEY FIX: Downcast dtypes to reduce file size ---
        processed_chunk = downcast_dtypes(processed_chunk)
        
        processed_chunk.to_csv(gold_path, mode='a', header=is_first_chunk, index=False)
        is_first_chunk = False

    print(f"\n✅ Success! Normalized and size-optimized 'Gold' data saved to: {gold_path}")

if __name__ == "__main__":
    core_dir = os.path.dirname(os.path.abspath(__file__))
    silver_dir = os.path.abspath(os.path.join(core_dir, '..', 'silver_data'))
    gold_dir = os.path.abspath(os.path.join(core_dir, '..', 'gold_data'))
    
    os.makedirs(gold_dir, exist_ok=True)
    silver_files = [f for f in os.listdir(silver_dir) if f.endswith('.csv')]

    if not silver_files:
        print("❌ No silver files found to process in the 'silver_data' directory.")
    else:
        print(f"Found {len(silver_files)} silver file(s) to process.")
        for fname in silver_files:
            silver_path = os.path.join(silver_dir, fname)
            gold_path = os.path.join(gold_dir, fname)
            
            try:
                process_silver_to_gold(silver_path, gold_path)
            except Exception as e:
                print(f"\n❌ FAILED to process {fname}. Error: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "="*50 + "\n✅ All gold data generation complete.")
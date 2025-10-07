import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import re
from tqdm import tqdm

# --- CONFIGURATION ---
GOLD_CHUNK_SIZE = 500_000  # How many rows to process at a time to manage memory

def downcast_dtypes(df):
    """
    Downcasts numeric columns to more memory-efficient types (float32).
    """
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
    return df

def transform_chunk(df_chunk):
    """
    Applies all transformations to convert a raw silver chunk into a gold-ready feature set.
    """
    features_df = df_chunk.copy()
    y_chunk = features_df.pop('outcome').apply(lambda x: 1 if x == 'win' else 0)

    abs_price_patterns = [
        r'^(open|high|low|close)$', r'^SMA_\d+$', r'^EMA_\d+$',
        r'^BB_(upper|lower)$', r'^(support|resistance)$', r'^ATR_level_.+$'
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

    cols_to_drop = abs_price_cols + ['entry_time', 'exit_time', 'entry_price', 'sl_price', 'tp_price', 'volume']
    features_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

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
            
    return features_df, y_chunk

def process_silver_to_gold(silver_path, gold_path):
    """
    Orchestrates the two-pass, chunk-based processing of a single silver file.
    """
    print(f"\n{'='*25}\nProcessing: {os.path.basename(silver_path)}\n{'='*25}")
    
    # --- Pass 1: Fit the Scaler and determine final column structure ---
    print("Pass 1: Fitting scaler on transformed data...")
    scaler = StandardScaler()
    final_feature_columns = None
    numeric_columns_to_scale = None
    candle_columns_final = None

    chunk_iterator = pd.read_csv(silver_path, chunksize=GOLD_CHUNK_SIZE)
    for chunk in tqdm(chunk_iterator, desc="Fitting Scaler"):
        features_transformed, _ = transform_chunk(chunk)
        
        if final_feature_columns is None:
            final_feature_columns = list(features_transformed.columns)
            candle_columns_final = [col for col in final_feature_columns if col.startswith("CDL")]
            numeric_columns_to_scale = list(
                features_transformed.select_dtypes(include=np.number).columns.difference(candle_columns_final)
            )

        numeric_chunk = features_transformed.reindex(columns=numeric_columns_to_scale, fill_value=0)
        scaler.partial_fit(numeric_chunk.fillna(0))

    # --- Pass 2: Transform and Save the Data ---
    print("\nPass 2: Transforming data and saving to gold file...")
    chunk_iterator = pd.read_csv(silver_path, chunksize=GOLD_CHUNK_SIZE)
    is_first_chunk = True

    if os.path.exists(gold_path):
        os.remove(gold_path)

    for chunk in tqdm(chunk_iterator, desc="Transforming Chunks"):
        features_transformed, y_transformed = transform_chunk(chunk)
        
        # --- KEY FIX: Rebuild the DataFrame to avoid FutureWarning ---
        # 1. Separate the numeric features to be scaled from the unscaled candle features
        numeric_to_scale_df = features_transformed.reindex(columns=numeric_columns_to_scale, fill_value=0)
        candle_features_df = features_transformed.reindex(columns=candle_columns_final, fill_value=0)
        
        # 2. Scale the numeric data and create a brand new DataFrame from it
        scaled_data = scaler.transform(numeric_to_scale_df.fillna(0))
        scaled_df = pd.DataFrame(scaled_data, index=features_transformed.index, columns=numeric_columns_to_scale)
        
        # 3. Combine the scaled numeric features and the unscaled candle features
        final_features = pd.concat([scaled_df, candle_features_df], axis=1)
        
        # 4. Align to the final column order to handle any discrepancies between chunks
        features_aligned = final_features.reindex(columns=final_feature_columns, fill_value=0)
        
        # 5. Combine with target
        processed_chunk = pd.concat([features_aligned, y_transformed], axis=1)
        
        # 6. Downcast and save
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
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import re

def create_gold_data(
    silver_csv_path,
    output_dir="../gold_data",
):
    """
    Transforms the silver dataset into a normalized, ML-ready "gold" dataset
    by converting all absolute price indicators into relational features.

    This script performs the following key steps:
    1.  Identifies all columns representing absolute price levels (OHLC, SMAs, EMAs, S/R, etc.).
    2.  Creates new relational features by calculating the normalized distance of these
        levels from the entry candle's closing price.
    3.  Drops the original absolute price columns, keeping only the new relational ones.
    4.  Applies One-Hot Encoding to categorical features (e.g., 'session', 'trade_type').
    5.  Bucketizes candlestick pattern columns into a simpler, more effective range.
    6.  Applies StandardScaler (Z-score normalization) to ALL final numeric features
        to prepare them for model training.
    """
    if not os.path.exists(silver_csv_path):
        print(f"❌ ERROR: Silver data file not found at: {silver_csv_path}")
        return

    print(f"Loading silver data from: {os.path.basename(silver_csv_path)}...")
    df = pd.read_csv(silver_csv_path)
    print(f"Loaded {len(df)} rows with {len(df.columns)} columns.")

    # --- 1. Separate Target Variable ---
    TARGET_COL = 'outcome'
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in the silver data!")
    
    y = df[TARGET_COL].apply(lambda x: 1 if x == 'win' else 0)
    features_df = df.drop(columns=[TARGET_COL])

    # --- 2. Transform Absolute Price Features into Relational Features ---
    print("Transforming absolute price features into normalized relational features...")
    
    # Patterns to identify columns that hold absolute price values
    abs_price_patterns = [
        r'^(open|high|low|close)$',
        r'^SMA_\d+$',
        r'^EMA_\d+$',
        r'^BB_(upper|lower)$', # BB_width is already a ratio, so we exclude it
        r'^(support|resistance)$',
        r'^ATR_level_.+$'
    ]
    
    abs_price_cols = []
    for pattern in abs_price_patterns:
        regex = re.compile(pattern)
        abs_price_cols.extend([col for col in features_df.columns if regex.match(col)])
        
    # Ensure 'close' is present for the calculation
    if 'close' not in features_df.columns:
        raise ValueError("'close' column is required for normalization but not found.")

    # Create new relational columns
    for col in abs_price_cols:
        # We don't need to create a relational version of 'close' to itself
        if col == 'close':
            continue
        # The new feature is the normalized distance from the closing price
        features_df[f'{col}_dist_norm'] = (features_df['close'] - features_df[col]) / features_df['close']

    print(f"Created {len(abs_price_cols) - 1} new relational features.")

    # --- 3. Drop Original and Unnecessary Columns ---
    # Now that we have the relational features, we can drop the originals
    # Also drop other identifiers that are not useful for the model
    cols_to_drop = abs_price_cols + ['entry_time', 'exit_time', 'entry_price', 'sl_price', 'tp_price', 'volume']
    
    original_feature_count = len(features_df.columns)
    features_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    print(f"Dropped {original_feature_count - len(features_df.columns)} original absolute price and identifier columns.")

    # --- 4. Identify and Process Remaining Column Types ---
    candle_cols = [col for col in features_df.columns if col.startswith("CDL")]
    categorical_cols = ['trade_type', 'session', 'trend_regime', 'vol_regime']
    categorical_cols = [col for col in categorical_cols if col in features_df.columns]
    
    numeric_cols = list(features_df.select_dtypes(include=np.number).columns.difference(candle_cols))

    print(f"Found {len(numeric_cols)} numeric, {len(categorical_cols)} categorical, and {len(candle_cols)} candle pattern features to process.")

    # --- 5. Process Categorical, Candle, and Numeric Features ---
    if categorical_cols:
        print("Applying One-Hot Encoding to categorical features...")
        features_df = pd.get_dummies(features_df, columns=categorical_cols, drop_first=True)

    def compress_pattern(v):
        if v >= 80: return 1.0
        elif v > 0: return 0.5
        elif v <= -80: return -1.0
        elif v < 0: return -0.5
        else: return 0.0

    if candle_cols:
        print("Compressing candlestick pattern features...")
        for col in candle_cols:
            features_df[col] = features_df[col].fillna(0).apply(compress_pattern)

    # Re-identify numeric columns after one-hot encoding, as dtypes might change
    numeric_cols_final = list(features_df.select_dtypes(include=np.number).columns)
    if numeric_cols_final:
        print(f"Applying StandardScaler to all {len(numeric_cols_final)} numeric features...")
        scaler = StandardScaler()
        features_df[numeric_cols_final] = scaler.fit_transform(features_df[numeric_cols_final])

    # --- 6. Combine and Save the Gold Dataset ---
    gold_df = pd.concat([features_df, y], axis=1)
    
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(silver_csv_path)
    out_path = os.path.join(output_dir, f"gold_{base_name}")
    
    gold_df.to_csv(out_path, index=False)

    print("\n" + "="*50)
    print(f"✅ Success! Normalized 'Gold' data saved to: {out_path}")
    print(f"💡 This dataset is now fully preprocessed and ready for model training.")
    print("="*50)

    return out_path

if __name__ == "__main__":
    # This makes the script easy to run directly.
    # It will look for 'AUDUSD1.csv' in the silver_data folder.
    # You can change this to any other file you want to process.
    silver_file = "../silver_data/AUDUSD1.csv"
    
    create_gold_data(silver_csv_path=silver_file)
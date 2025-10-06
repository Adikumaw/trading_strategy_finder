import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import itertools
import lightgbm as lgb

# --- CONFIGURATION ---
MIN_UNIQUE_SIGNALS = 15
MIN_PROFIT_THRESHOLD = 0.001
MIN_RISK_REWARD = 1.0
N_TOP_FEATURES = 20
STRATEGY_COMPLEXITY = [3, 4]
CHUNK_SIZE = 500000
ML_SAMPLE_SIZE = 1000000
N_QUANTILES = 5

# --- HELPER FUNCTIONS ---
def downcast_dtypes(df):
    for col in df.columns:
        if df[col].dtype == 'float64': df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64': df[col] = df[col].astype('int32')
    return df

def add_exhaustive_features(df):
    new_cols = {}
    for p in [20, 50, 100, 200]: new_cols[f'price_vs_sma{p}_ratio'] = (df['close'] - df[f'SMA_{p}']) / df[f'SMA_{p}']
    for p in [8, 13, 21, 50]: new_cols[f'price_vs_ema{p}_ratio'] = (df['close'] - df[f'EMA_{p}']) / df[f'EMA_{p}']
    new_cols['ema_short_vs_long_ratio'] = (df['EMA_8'] - df['EMA_50']) / df['close']
    new_cols['bb_position'] = (df['close'] - df['BB_lower']) / (df['BB_width'] + 1e-9)
    new_cols['atr_ratio'] = df['ATR_14'] / df['close']
    new_cols['price_vs_support_ratio'] = (df['close'] - df['support']) / df['close']
    new_cols['price_vs_resistance_ratio'] = (df['resistance'] - df['close']) / df['close']
    new_cols['RSI_delta'] = df['RSI_14'].diff()
    new_cols['MACD_hist_delta'] = df['MACD_hist'].diff()
    df = df.assign(**new_cols)

    quantile_labels = [f'q_{i+1}' for i in range(N_QUANTILES)]
    cols_to_bin = [col for col in df.columns if 'ratio' in col or 'delta' in col or 'position' in col or col in ['RSI_14', 'ADX']]
    cols_to_bin += [col for col in df.columns if col.startswith('bullish_ratio_last_')]
    
    binned_cols = {}
    for col in cols_to_bin:
        try:
            binned_cols[f'{col}_qbin'] = pd.qcut(df[col], N_QUANTILES, labels=quantile_labels, duplicates='drop')
        except ValueError:
            continue
    df = df.assign(**binned_cols)
    
    candle_cols = [col for col in df.columns if col.startswith('CDL')]
    is_bullish = df[candle_cols].gt(0).any(axis=1)
    is_bearish = df[candle_cols].lt(0).any(axis=1)
    df['candlestick_type'] = np.select([is_bullish, is_bearish], ['Bullish', 'Bearish'], default='None')
    return df

def get_top_features(df_sample):
    print("\n--- Stage 1: Finding Top Features with Machine Learning ---")
    df_sample['profitable'] = (df_sample['tp_ratio'] > df_sample['sl_ratio']).astype(int)
    feature_cols = [col for col in df_sample.columns if '_qbin' in col or col in ['session', 'trade_type', 'candlestick_type']]
    
    # --- FIX: Exclude features that leak information from the future ---
    features_to_exclude = ['sl_ratio_qbin', 'tp_ratio_qbin']
    feature_cols = [f for f in feature_cols if f not in features_to_exclude]
    # --- END FIX ---
    
    X = df_sample[feature_cols].copy()
    y = df_sample['profitable']
    
    for col in X.select_dtypes(['object', 'category']).columns:
        X[col] = X[col].astype('category').cat.codes
    X.fillna(-1, inplace=True)

    print(f"Training model on {len(X)} samples to rank {len(feature_cols)} features...")
    model = lgb.LGBMClassifier(random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    importances = pd.DataFrame({'feature': feature_cols, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
    top_features = importances.head(N_TOP_FEATURES)['feature'].tolist()
    print(f"Top {len(top_features)} features identified.")
    return top_features

def analyze_strategies_in_chunks(silver_file, gold_file, min_signals, min_profit, min_rr):
    df_sample = pd.read_csv(silver_file, nrows=ML_SAMPLE_SIZE, low_memory=False)
    df_sample = downcast_dtypes(df_sample)
    features_df_sample = add_exhaustive_features(df_sample)
    top_features = get_top_features(features_df_sample)

    print("\n--- Stage 2: Aggregating Base Patterns from Chunks (The 'Super-Group') ---")
    chunk_reader = pd.read_csv(silver_file, chunksize=CHUNK_SIZE, low_memory=False)
    
    aggregated_chunks = []
    for chunk in tqdm(chunk_reader, desc="Processing Chunks"):
        chunk = downcast_dtypes(chunk)
        features_df_chunk = add_exhaustive_features(chunk)

        if not all(f in features_df_chunk.columns for f in top_features):
            continue
        
        super_group = features_df_chunk.groupby(top_features, observed=True).agg(
            trade_count=('entry_time', 'size'), unique_entry_signals=('entry_time', 'nunique'),
            sum_sl_ratio=('sl_ratio', 'sum'), sum_tp_ratio=('tp_ratio', 'sum')
        ).reset_index()
        aggregated_chunks.append(super_group)
    
    print("\nCombining aggregated chunks...")
    if not aggregated_chunks:
        print("\n❌ No data was aggregated from chunks. Cannot proceed.")
        return
        
    final_agg = pd.concat(aggregated_chunks)
    final_super_group = final_agg.groupby(top_features, observed=True).sum().reset_index()
    
    print("\n--- Stage 3: Deriving and Evaluating Sub-Strategies ---")
    combinations_to_test = []
    for c in STRATEGY_COMPLEXITY:
        combinations_to_test.extend(itertools.combinations(top_features, r=c))
        
    print(f"Generated {len(combinations_to_test)} unique strategy combinations to test from top features.")
    
    all_found_strategies = []
    for combo in tqdm(combinations_to_test, desc="Deriving Strategies"):
        combo = list(combo)
        
        sub_group = final_super_group.groupby(combo, observed=True)[
            ['trade_count', 'unique_entry_signals', 'sum_sl_ratio', 'sum_tp_ratio']
        ].sum()
        
        for name, row in sub_group.iterrows():
            if row['unique_entry_signals'] < min_signals:
                continue
            
            if isinstance(name, str): name = [name]
            strategy_def = dict(zip(combo, name))
            
            all_found_strategies.append({
                'strategy_definition': strategy_def, 'trade_count': row['trade_count'],
                'unique_entry_signals': row['unique_entry_signals'], 'sum_sl_ratio': row['sum_sl_ratio'],
                'sum_tp_ratio': row['sum_tp_ratio'],
            })

    if not all_found_strategies:
        print("\n❌ No strategies found that meet the quality criteria.")
        return

    print(f"\nProcessing and ranking {len(all_found_strategies)} discovered patterns...")
    gold_df = pd.DataFrame(all_found_strategies)
    gold_df['suggested_sl_ratio'] = gold_df['sum_sl_ratio'] / gold_df['trade_count']
    gold_df['suggested_tp_ratio'] = gold_df['sum_tp_ratio'] / gold_df['trade_count']
    
    gold_df = gold_df[gold_df['suggested_tp_ratio'] >= min_profit]
    gold_df['risk_reward_ratio'] = gold_df['suggested_tp_ratio'] / gold_df['suggested_sl_ratio']
    gold_df = gold_df[gold_df['risk_reward_ratio'] >= min_rr]

    if gold_df.empty:
        print("\n❌ No strategies found that meet the quality criteria after aggregation.")
        return

    gold_df['strategy_score'] = gold_df['unique_entry_signals'] * gold_df['risk_reward_ratio']
    gold_df.sort_values(by='strategy_score', ascending=False, inplace=True)
    
    final_cols = ['strategy_definition', 'trade_count', 'unique_entry_signals', 'suggested_sl_ratio', 'suggested_tp_ratio', 'risk_reward_ratio', 'strategy_score']
    gold_df = gold_df[final_cols]
    
    gold_df.to_csv(gold_file, index=False)
    print(f"✅ Success! Found and ranked {len(gold_df)} unique strategies. Saved to {os.path.basename(gold_file)}")

if __name__ == "__main__":
    core_dir = os.path.dirname(os.path.abspath(__file__))
    silver_data_dir = os.path.abspath(os.path.join(core_dir, '..', 'silver_data'))
    gold_data_dir = os.path.abspath(os.path.join(core_dir, '..', 'gold_data'))
    os.makedirs(gold_data_dir, exist_ok=True)

    try:
        silver_files = [f for f in os.listdir(silver_data_dir) if f.endswith('.csv')]
    except FileNotFoundError:
        print(f"❌ Error: The directory '{silver_data_dir}' was not found.")
        silver_files = []

    if not silver_files:
        print("❌ No CSV files found in 'silver_data'.")
    else:
        print(f"Found {len(silver_files)} files to process...")

    for filename in silver_files:
        silver_path = os.path.join(silver_data_dir, filename)
        gold_path = os.path.join(gold_data_dir, filename)

        print("\n" + "=" * 50 + f"\nAnalyzing: {filename}")
        try:
            analyze_strategies_in_chunks(
                silver_file=silver_path, gold_file=gold_path,
                min_signals=MIN_UNIQUE_SIGNALS, min_profit=MIN_PROFIT_THRESHOLD,
                min_rr=MIN_RISK_REWARD
            )
        except Exception as e:
            print(f"❌ FAILED to process {filename}. Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 50 + "\nExhaustive gold data generation complete.")
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import itertools

# --- CONFIGURATION ---

# --- Analysis Parameters (Exploratory Mode) ---
# We use more lenient filters to cast a wider net during this exhaustive search.
MIN_UNIQUE_SIGNALS = 15
MIN_PROFIT_THRESHOLD = 0.001
MIN_RISK_REWARD = 1.0

# --- Advanced Feature Engineering ---
N_QUANTILES = 5

# ### --- START OF THE STRATEGY FACTORY --- ###
# 1. Define pools of features based on their logical category.
CONTEXT_FEATURES = ['session', 'trade_type']
TREND_FEATURES = ['price_vs_sma50_ratio', 'price_vs_sma200_ratio', 'ema_short_vs_long_ratio', 'ADX']
MOMENTUM_FEATURES = ['RSI_14', 'RSI_delta', 'MACD_hist_delta']
VOLATILITY_FEATURES = ['bb_position', 'atr_ratio']
PRICE_ACTION_FEATURES = ['price_vs_support_ratio', 'price_vs_resistance_ratio', 'candlestick_type']
RECENT_HISTORY_FEATURES = ['bullish_ratio_last_5', 'bullish_ratio_last_10', 'bullish_ratio_last_20']

# 2. Generate a large list of logical combinations to test.
FEATURE_COMBINATIONS_TO_TEST = []

# Create 4-part strategies (Context + Trend + Momentum + Volatility) - 2*4*3*2 = 48 combos
for combo in itertools.product(CONTEXT_FEATURES, TREND_FEATURES, MOMENTUM_FEATURES, VOLATILITY_FEATURES):
    FEATURE_COMBINATIONS_TO_TEST.append(list(combo))

# Create 3-part strategies (Context + Trend + Price Action) - 2*4*3 = 24 combos
for combo in itertools.product(CONTEXT_FEATURES, TREND_FEATURES, PRICE_ACTION_FEATURES):
    FEATURE_COMBINATIONS_TO_TEST.append(list(combo))

# Create 3-part strategies (Context + Momentum + Recent History) - 2*3*3 = 18 combos
for combo in itertools.product(CONTEXT_FEATURES, MOMENTUM_FEATURES, RECENT_HISTORY_FEATURES):
    FEATURE_COMBINATIONS_TO_TEST.append(list(combo))

# Create 3-part strategies (Context + Volatility + Price Action) - 2*2*3 = 12 combos
for combo in itertools.product(CONTEXT_FEATURES, VOLATILITY_FEATURES, PRICE_ACTION_FEATURES):
    FEATURE_COMBINATIONS_TO_TEST.append(list(combo))
# Total Combinations: 48 + 24 + 18 + 12 = 102
# ### --- END OF THE STRATEGY FACTORY --- ###

# --- CHUNKING CONFIGURATION ---
CHUNK_SIZE = 500000

# --- HELPER FUNCTIONS (Unchanged from previous robust version) ---
def downcast_dtypes(df):
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.columns:
        if df[col].dtype == 'float64': df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64': df[col] = df[col].astype('int32')
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
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


# --- MAIN ANALYSIS FUNCTION ---
def analyze_strategies_in_chunks(silver_file, gold_file, min_signals, min_profit, min_rr):
    chunk_reader = pd.read_csv(silver_file, chunksize=CHUNK_SIZE, low_memory=False)
    all_found_strategies = []
    
    print("Processing file in chunks...")
    for chunk in tqdm(chunk_reader, desc="Analyzing Chunks"):
        chunk = downcast_dtypes(chunk)
        features_df = add_exhaustive_features(chunk)

        for combo in FEATURE_COMBINATIONS_TO_TEST:
            processed_combo = [f"{c}_qbin" if c not in ['session', 'trade_type', 'candlestick_type'] else c for c in combo]
            if not all(c in features_df.columns for c in processed_combo):
                continue

            grouped = features_df.groupby(processed_combo, observed=True)
            for name, group in grouped:
                unique_signals = group['entry_time'].nunique()
                if unique_signals == 0: continue
                
                all_found_strategies.append({
                    'strategy_definition': dict(zip(processed_combo, name)),
                    'trade_count': len(group),
                    'unique_entry_signals': unique_signals,
                    'sum_sl_ratio': group['sl_ratio'].sum(),
                    'sum_tp_ratio': group['tp_ratio'].sum(),
                })

    if not all_found_strategies:
        print("\n❌ No patterns found in any chunk.")
        return

    print(f"\nAggregating results from {len(all_found_strategies)} partial patterns...")
    final_df = pd.DataFrame(all_found_strategies)
    final_df['strategy_definition'] = final_df['strategy_definition'].astype(str)
    aggregated = final_df.groupby('strategy_definition').agg({
        'trade_count': 'sum', 'unique_entry_signals': 'sum',
        'sum_sl_ratio': 'sum', 'sum_tp_ratio': 'sum'
    }).reset_index()

    print("Applying quality filters and ranking final strategies...")
    aggregated['suggested_sl_ratio'] = aggregated['sum_sl_ratio'] / aggregated['trade_count']
    aggregated['suggested_tp_ratio'] = aggregated['sum_tp_ratio'] / aggregated['trade_count']
    
    aggregated = aggregated[aggregated['unique_entry_signals'] >= min_signals]
    aggregated = aggregated[aggregated['suggested_tp_ratio'] >= min_profit]
    aggregated['risk_reward_ratio'] = aggregated['suggested_tp_ratio'] / aggregated['suggested_sl_ratio']
    aggregated = aggregated[aggregated['risk_reward_ratio'] >= min_rr]

    if aggregated.empty:
        print("\n❌ No strategies found that meet the quality criteria after aggregation.")
        return

    aggregated['strategy_score'] = aggregated['unique_entry_signals'] * aggregated['risk_reward_ratio']
    aggregated.sort_values(by='strategy_score', ascending=False, inplace=True)
    gold_df = aggregated[['strategy_definition', 'trade_count', 'unique_entry_signals', 'suggested_sl_ratio', 'suggested_tp_ratio', 'risk_reward_ratio', 'strategy_score']]
    
    gold_df.to_csv(gold_file, index=False)
    print(f"✅ Success! Found and ranked {len(gold_df)} unique strategies. Saved to {os.path.basename(gold_file)}")


# --- MAIN EXECUTION BLOCK ---
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
        print(f"Generated {len(FEATURE_COMBINATIONS_TO_TEST)} combinations to test.")
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
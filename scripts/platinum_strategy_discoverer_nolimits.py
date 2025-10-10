import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, _tree
import os
from tqdm import tqdm
import gc
import hashlib

# --- CONFIGURATION (Unchanged) ---
MIN_CANDLES_FOR_ANALYSIS = 100
MIN_CANDLES_PER_RULE = 50
DECISION_TREE_MAX_DEPTH = 5
MIN_PREDICTED_TRADE_DENSITY = 5.0 

# --- Functions (Unchanged, but renamed one for clarity) ---
def get_rules_from_tree(tree, feature_names):
    tree_ = tree.tree_; feature_name = [feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature]; rules = {}
    def recurse(node, rule_path):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name, threshold = feature_name[node], tree_.threshold[node]
            recurse(tree_.children_left[node], rule_path + [f"`{name}` <= {threshold:.4f}"])
            recurse(tree_.children_right[node], rule_path + [f"`{name}` > {threshold:.4f}"])
        else:
            predicted_density, num_samples = tree_.value[node][0][0], tree_.n_node_samples[node]
            if num_samples >= MIN_CANDLES_PER_RULE and predicted_density > MIN_PREDICTED_TRADE_DENSITY:
                rules[node] = {"rule": " and ".join(rule_path) if rule_path else "all_trades", "predicted_density": predicted_density, "num_candles": int(num_samples)}
    recurse(0, []); return list(rules.values())

def filter_dataframe_for_definition(df, definition): # Renamed from filter_chunk_...
    temp_df = df
    if isinstance(definition['sl_def'], str):
        level, bin_val = definition['sl_def'], definition['sl_bin']; pct_col = f"sl_placement_pct_to_{level}"
        if pct_col in temp_df.columns:
            lower_bound, upper_bound = bin_val / 10.0, (bin_val + 1) / 10.0
            temp_df = temp_df[(temp_df[pct_col] >= lower_bound) & (temp_df[pct_col] < upper_bound)]
    else: temp_df = temp_df[np.isclose(temp_df['sl_ratio'], definition['sl_def'])]
    if isinstance(definition['tp_def'], str):
        level, bin_val = definition['tp_def'], definition['tp_bin']; pct_col = f"tp_placement_pct_to_{level}"
        if pct_col in temp_df.columns:
            lower_bound, upper_bound = bin_val / 10.0, (bin_val + 1) / 10.0
            temp_df = temp_df[(temp_df[pct_col] >= lower_bound) & (temp_df[pct_col] < upper_bound)]
    else: temp_df = temp_df[np.isclose(temp_df['tp_ratio'], definition['tp_def'])]
    return temp_df

# --- Main execution block ---
if __name__ == "__main__":
    core_dir = os.path.dirname(os.path.abspath(__file__))
    gold_features_dir, silver_outcomes_dir, combinations_dir, discovered_dir, blacklist_dir = [os.path.abspath(os.path.join(core_dir, '..', d)) for d in ['gold_data/features', 'silver_data/outcomes', 'platinum_data/combinations', 'platinum_data/discovered_strategy', 'platinum_data/blacklists']]
    os.makedirs(discovered_dir, exist_ok=True); os.makedirs(blacklist_dir, exist_ok=True)

    combination_files = [f for f in os.listdir(combinations_dir) if f.endswith('.csv')]
    if not combination_files:
        print("❌ No combination files found.")
    else:
        for fname in combination_files:
            print(f"\n{'='*25}\nProcessing: {fname}\n{'='*25}")
            combinations_path, gold_path, outcomes_path, discovered_path, blacklist_path = [os.path.join(d, fname) for d in [combinations_dir, gold_features_dir, silver_outcomes_dir, discovered_dir, blacklist_dir]]
            processed_log_path = os.path.join(discovered_dir, f".{fname}.processed_log")
            
            if not os.path.exists(gold_path) or not os.path.exists(outcomes_path): continue
            
            try:
                # --- State Management (Your robust logic is preserved) ---
                all_definitions = pd.read_csv(combinations_path)
                all_definitions['sl_def'] = all_definitions['sl_def'].astype(object); all_definitions['tp_def'] = all_definitions['tp_def'].astype(object)
                if 'sl_bin' in all_definitions.columns: all_definitions['sl_bin'] = all_definitions['sl_bin'].astype('Int64')
                if 'tp_bin' in all_definitions.columns: all_definitions['tp_bin'] = all_definitions['tp_bin'].astype('Int64')
                
                try:
                    with open(processed_log_path, 'r') as f: processed_keys = set(f.read().splitlines())
                except FileNotFoundError: processed_keys = set()
                
                try: blacklist = pd.read_csv(blacklist_path)
                except FileNotFoundError: blacklist = pd.DataFrame()
                
                key_cols = ['type', 'sl_def', 'sl_bin', 'tp_def', 'tp_bin']
                all_definitions['key'] = all_definitions[key_cols].astype(str).sum(axis=1).apply(lambda x: hashlib.sha256(x.encode()).hexdigest())
                if not blacklist.empty:
                    blacklist['sl_def'] = blacklist['sl_def'].astype(object); blacklist['tp_def'] = blacklist['tp_def'].astype(object)
                    blacklist['key'] = blacklist[key_cols].astype(str).sum(axis=1).apply(lambda x: hashlib.sha256(x.encode()).hexdigest())
                
                definitions_to_process = all_definitions[(~all_definitions['key'].isin(processed_keys)) | (all_definitions['key'].isin(blacklist['key'] if not blacklist.empty else []))]
                
                if definitions_to_process.empty:
                    print("✅ All combinations have already been processed."); continue
                
                print(f"Loaded {len(all_definitions)} total combinations. Analyzing {len(definitions_to_process)} new or blacklisted combinations.")

                # --- SPEED OPTIMIZATION: Load all data into RAM once ---
                print(f"Loading data for {fname} into memory...")
                gold_features_df = pd.read_csv(gold_path, parse_dates=['time'])
                outcomes_df = pd.read_csv(outcomes_path, parse_dates=['entry_time'])
                # Convert outcome to integer once
                outcomes_df['outcome'] = 1
                print("✅ Data loaded.")

                # Open the log file and start the high-speed, in-memory loop
                with open(processed_log_path, 'a') as log_file:
                    for _, definition in tqdm(definitions_to_process.iterrows(), total=len(definitions_to_process), desc="Discovering Strategies"):
                        log_file.write(definition['key'] + '\n'); log_file.flush()

                        # SPEED OPTIMIZATION: Filter the in-memory DataFrame directly
                        slice_df = filter_dataframe_for_definition(outcomes_df, definition)
                        if slice_df.empty: continue
                        
                        candle_agg_df = slice_df.groupby('entry_time').agg(trade_count=('outcome', 'size')).reset_index()
                        if len(candle_agg_df) < MIN_CANDLES_FOR_ANALYSIS: continue
                        training_slice = pd.merge(gold_features_df, candle_agg_df, left_on='time', right_on='entry_time', how='inner')
                        if training_slice.empty: continue
                            
                        y, X = training_slice['trade_count'], training_slice.drop(columns=['time', 'entry_time', 'trade_count'])
                        model = DecisionTreeRegressor(max_depth=DECISION_TREE_MAX_DEPTH, min_samples_leaf=MIN_CANDLES_PER_RULE, random_state=42).fit(X, y)
                        rules = get_rules_from_tree(model, list(X.columns))

                        if rules:
                            rules_to_save = []
                            for rule in rules:
                                rule_candle_times = X.query(rule['rule']).index if rule['rule'] != 'all_trades' else X.index
                                actual_avg_density = training_slice.loc[rule_candle_times]['trade_count'].mean()
                                total_trades = training_slice.loc[rule_candle_times]['trade_count'].sum()
                                rules_to_save.append({
                                    'type': definition['type'], 'sl_def': definition['sl_def'], 'sl_bin': definition.get('sl_bin'),
                                    'tp_def': definition['tp_def'], 'tp_bin': definition.get('tp_bin'),
                                    'market_rule': rule['rule'], 'avg_trade_density': round(actual_avg_density, 2),
                                    'num_candles': rule['num_candles'], 'total_trades': int(total_trades)
                                })
                            new_rules_df = pd.DataFrame(rules_to_save)
                            file_exists = os.path.exists(discovered_path) and os.path.getsize(discovered_path) > 0
                            new_rules_df.to_csv(discovered_path, mode='a', header=not file_exists, index=False)
                
                # --- Final Cleanup (Unchanged) ---
                print("\nLoop finished. Performing final cleanup of discovered strategies...")
                try:
                    final_df = pd.read_csv(discovered_path)
                    final_df.drop_duplicates(subset=key_cols + ['market_rule'], keep='last', inplace=True)
                    final_df.sort_values(by=['avg_trade_density', 'num_candles'], ascending=[False, False], inplace=True)
                    final_df.to_csv(discovered_path, index=False)
                    print(f"✅ Cleanup complete. Total unique strategies found: {len(final_df)}")
                except FileNotFoundError:
                    print("ℹ️ No strategies were discovered in this run.")

            except Exception as e:
                print(f"\n❌ FAILED to process {fname}. Error: {e}"); import traceback; traceback.print_exc()

    print("\n" + "="*50 + "\n✅ All strategy discovery complete.")
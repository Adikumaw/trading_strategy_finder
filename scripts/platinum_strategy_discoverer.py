import pandas as pd
import numpy as np
# KEY CHANGE: Switched to DecisionTreeRegressor for predicting win rates
from sklearn.tree import DecisionTreeRegressor, _tree
import os
from tqdm import tqdm
import gc

# --- CONFIGURATION ---
OUTCOMES_CHUNK_SIZE = 1_000_000
MIN_TRADES_PER_DEFINITION = 100
MIN_CANDLES_PER_RULE = 50 # A rule must be based on at least 50 unique market events (candles)
DECISION_TREE_MAX_DEPTH = 4 # Increased depth slightly to handle regression
WIN_PROBABILITY_THRESHOLD = 0.6 # The predicted win rate for a rule's leaf must be above this
POSITIONING_TOLERANCE_BPS = 10

# KEY CHANGE: This function is now adapted for a DecisionTreeRegressor
def get_rules_from_tree(tree, feature_names):
    """Extracts human-readable IF-THEN rules from a trained Decision Tree Regressor."""
    tree_ = tree.tree_
    feature_name = [feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature]
    rules = {}
    
    def recurse(node, rule_path):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name, threshold = feature_name[node], tree_.threshold[node]
            recurse(tree_.children_left[node], rule_path + [f"`{name}` <= {threshold:.4f}"])
            recurse(tree_.children_right[node], rule_path + [f"`{name}` > {threshold:.4f}"])
        else:
            # For a regressor, 'value' is the predicted win rate for this leaf
            predicted_win_rate = tree_.value[node][0][0]
            num_samples = tree_.n_node_samples[node]
            
            # Check if the rule meets our performance criteria
            if num_samples >= MIN_CANDLES_PER_RULE and predicted_win_rate > WIN_PROBABILITY_THRESHOLD:
                rules[node] = {
                    "rule": " and ".join(rule_path) if rule_path else "all_trades",
                    "win_prob": predicted_win_rate,
                    # This is the number of unique candles that fit the rule
                    "num_candles": int(num_samples)
                }
    recurse(0, [])
    return list(rules.values())

# --- filter_chunk_for_definition (CORRECTED) ---
def filter_chunk_for_definition(chunk, definition):
    """
    Filters a chunk based on a strategy definition, now handling all types correctly.
    """
    temp_df = chunk
    
    # --- Filter by SL definition ---
    # Check if the SL definition is a string (i.e., an indicator name)
    if isinstance(definition['sl_def'], str):
        level, bin_val = definition['sl_def'], definition['sl_bin']
        pct_col = f"sl_placement_pct_to_{level}"
        if pct_col in temp_df.columns:
            lower_bound = bin_val / 10.0
            upper_bound = (bin_val + 1) / 10.0
            temp_df = temp_df[(temp_df[pct_col] >= lower_bound) & (temp_df[pct_col] < upper_bound)]
    else: # It's a numeric ratio
        sl_ratio = definition['sl_def']
        temp_df = temp_df[np.isclose(temp_df['sl_ratio'], sl_ratio)]

    # --- Filter by TP definition ---
    # Check if the TP definition is a string
    if isinstance(definition['tp_def'], str):
        level, bin_val = definition['tp_def'], definition['tp_bin']
        pct_col = f"tp_placement_pct_to_{level}"
        if pct_col in temp_df.columns:
            lower_bound = bin_val / 10.0
            upper_bound = (bin_val + 1) / 10.0
            temp_df = temp_df[(temp_df[pct_col] >= lower_bound) & (temp_df[pct_col] < upper_bound)]
    else: # It's a numeric ratio
        tp_ratio = definition['tp_def']
        temp_df = temp_df[np.isclose(temp_df['tp_ratio'], tp_ratio)]
        
    return temp_df


# --- Main execution block ---
if __name__ == "__main__":
    core_dir = os.path.dirname(os.path.abspath(__file__))
    gold_features_dir = os.path.abspath(os.path.join(core_dir, '..', 'gold_data', 'features'))
    silver_outcomes_dir = os.path.abspath(os.path.join(core_dir, '..', 'silver_data', 'outcomes'))
    combinations_dir = os.path.abspath(os.path.join(core_dir, '..', 'platinum_data', 'combinations'))
    discovered_dir = os.path.abspath(os.path.join(core_dir, '..', 'platinum_data', 'discovered_strategy'))
    blacklist_dir = os.path.abspath(os.path.join(core_dir, '..', 'platinum_data', 'blacklists'))
    os.makedirs(discovered_dir, exist_ok=True); os.makedirs(blacklist_dir, exist_ok=True)

    combination_files = [f for f in os.listdir(combinations_dir) if f.endswith('.csv')]

    if not combination_files:
        print("❌ No combination files found. Please run the combinations generator first.")
    else:
        for fname in combination_files:
            print(f"\n{'='*25}\nProcessing: {fname}\n{'='*25}")
            
            combinations_path = os.path.join(combinations_dir, fname)
            gold_path = os.path.join(gold_features_dir, fname)
            outcomes_path = os.path.join(silver_outcomes_dir, fname)
            discovered_path = os.path.join(discovered_dir, fname)
            blacklist_path = os.path.join(blacklist_dir, fname)

            if not os.path.exists(gold_path) or not os.path.exists(outcomes_path):
                print(f"⚠️ SKIPPING: Missing gold features or silver outcomes for {fname}."); continue
            
            try:
                all_definitions = pd.read_csv(combinations_path)
                all_definitions['sl_def'] = all_definitions['sl_def'].astype(object); all_definitions['tp_def'] = all_definitions['tp_def'].astype(object)
                if 'sl_bin' in all_definitions.columns: all_definitions['sl_bin'] = all_definitions['sl_bin'].astype('Int64')
                if 'tp_bin' in all_definitions.columns: all_definitions['tp_bin'] = all_definitions['tp_bin'].astype('Int64')
                
                try: existing_results = pd.read_csv(discovered_path)
                except FileNotFoundError: existing_results = pd.DataFrame()
                
                try: blacklist = pd.read_csv(blacklist_path)
                except FileNotFoundError: blacklist = pd.DataFrame()
                
                key_cols = ['type', 'sl_def', 'sl_bin', 'tp_def', 'tp_bin']
                
                for df in [existing_results, blacklist]:
                    if not df.empty:
                        df['sl_def'] = df['sl_def'].astype(object)
                        df['tp_def'] = df['tp_def'].astype(object)

                if not existing_results.empty: existing_results['key'] = [tuple(x) for x in existing_results[key_cols].values]
                else: existing_results['key'] = pd.Series(dtype='object')
                if not blacklist.empty: blacklist['key'] = [tuple(x) for x in blacklist[key_cols].values]
                else: blacklist['key'] = pd.Series(dtype='object')
                all_definitions['key'] = [tuple(x) for x in all_definitions[key_cols].values]
                
                definitions_to_process = all_definitions[(~all_definitions['key'].isin(existing_results['key'])) | (all_definitions['key'].isin(blacklist['key']))].drop(columns=['key'])
                
                if definitions_to_process.empty:
                    print("✅ All combinations have already been processed."); continue
                
                print(f"Loaded {len(all_definitions)} total combinations. Analyzing {len(definitions_to_process)}.")

                gold_features_df = pd.read_csv(gold_path, parse_dates=['time'])
                all_new_strategies = []

                for _, definition in tqdm(definitions_to_process.iterrows(), total=len(definitions_to_process), desc="Discovering Strategies"):
                    outcomes_iterator = pd.read_csv(outcomes_path, chunksize=OUTCOMES_CHUNK_SIZE, parse_dates=['entry_time'])
                    matching_trades = [filter_chunk_for_definition(chunk, definition) for chunk in outcomes_iterator]
                    slice_df = pd.concat(matching_trades)

                    if slice_df.empty: continue

                    # --- THE FIX IS HERE ---
                    # Correctly chain the .str accessor methods
                    slice_df['outcome'] = (slice_df['outcome'].str.strip().str.lower() == 'win').astype(int)

                    candle_agg_df = slice_df.groupby('entry_time').agg(win_rate=('outcome', 'mean'), trade_count=('outcome', 'size')).reset_index()
                    
                    if len(candle_agg_df) < MIN_TRADES_PER_DEFINITION: continue

                    training_slice = pd.merge(gold_features_df, candle_agg_df, left_on='time', right_on='entry_time', how='inner')
                    
                    if training_slice.empty: continue
                        
                    y = training_slice['win_rate']
                    X = training_slice.drop(columns=['time', 'entry_time', 'win_rate', 'trade_count'])
                    
                    model = DecisionTreeRegressor(max_depth=DECISION_TREE_MAX_DEPTH, min_samples_leaf=MIN_CANDLES_PER_RULE, random_state=42)
                    model.fit(X, y)
                    rules = get_rules_from_tree(model, list(X.columns))

                    for rule in rules:
                        rule_candle_times = X.query(rule['rule']).index if rule['rule'] != 'all_trades' else X.index
                        total_trades = training_slice.loc[rule_candle_times]['trade_count'].sum()

                        all_new_strategies.append({
                            'type': definition['type'], 'sl_def': definition['sl_def'], 'sl_bin': definition.get('sl_bin'),
                            'tp_def': definition['tp_def'], 'tp_bin': definition.get('tp_bin'),
                            'market_rule': rule['rule'], 'win_prob': round(rule['win_prob'], 4),
                            'num_candles': rule['num_candles'], 'total_trades': int(total_trades)
                        })
                
                if all_new_strategies:
                    new_results_df = pd.DataFrame(all_new_strategies)
                    
                    if 'num_trades' in existing_results.columns:
                        existing_results.rename(columns={'num_candles': 'num_candles'}, inplace=True) # Corrected rename target
                    
                    combined_results = pd.concat([existing_results.drop(columns=['key'], errors='ignore'), new_results_df])
                    combined_results.drop_duplicates(subset=['type', 'sl_def', 'sl_bin', 'tp_def', 'tp_bin', 'market_rule'], keep='last', inplace=True)
                    combined_results.sort_values(by=['win_prob', 'num_candles'], ascending=[False, False], inplace=True)
                    
                    combined_results.to_csv(discovered_path, index=False)
                    print(f"\n✅ Success! Discovered {len(new_results_df)} new potential strategy rules.")
                else:
                    print("\nℹ️ No new viable strategies were discovered in this run.")
                
            except Exception as e:
                print(f"\n❌ FAILED to process {fname}. Error: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "="*50 + "\n✅ All strategy discovery complete.")
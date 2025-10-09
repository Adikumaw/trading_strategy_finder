import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, _tree
import os
from tqdm import tqdm
import gc

# --- CONFIGURATION ---
OUTCOMES_CHUNK_SIZE = 1_000_000 # Process silver outcomes in 1M row chunks
MIN_TRADES_PER_DEFINITION = 100 # A strategy definition must have at least this many trades to be analyzed
MIN_TRADES_PER_RULE = 50       # A final discovered rule must be based on at least this many trades
DECISION_TREE_MAX_DEPTH = 3
WIN_PROBABILITY_THRESHOLD = 0.6 # Minimum win probability for a rule to be considered valid
# KEY CHANGE: Tolerance is now in basis points. 10 bps = 0.1%
POSITIONING_TOLERANCE_BPS = 10

def get_rules_from_tree(tree, feature_names):
    """Extracts human-readable IF-THEN rules from a trained Decision Tree."""
    tree_ = tree.tree_
    feature_name = [feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature]
    rules = {}
    
    def recurse(node, rule_path):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name, threshold = feature_name[node], tree_.threshold[node]
            recurse(tree_.children_left[node], rule_path + [f"`{name}` <= {threshold:.4f}"])
            recurse(tree_.children_right[node], rule_path + [f"`{name}` > {threshold:.4f}"])
        else:
            value = tree_.value[node][0]
            # Ensure the node predicts 'win' (class 1)
            if len(value) > 1 and np.argmax(value) == 1:
                num_samples = np.sum(value)
                win_prob = value[1] / num_samples
                # Check against thresholds
                if num_samples >= MIN_TRADES_PER_RULE and win_prob > WIN_PROBABILITY_THRESHOLD:
                    rules[node] = {
                        "rule": " and ".join(rule_path) if rule_path else "all_trades",
                        "win_prob": win_prob,
                        "num_trades": int(num_samples)
                    }
    recurse(0, [])
    return list(rules.values())

def filter_chunk_for_definition(chunk, definition):
    """Filters a single data chunk based on one strategy definition."""
    temp_df = chunk
    
    # Filter by SL definition
    if isinstance(definition['sl_def'], str): # It's a positioning rule
        col_name = f"sl_dist_to_{definition['sl_def']}_bps"
        if col_name in temp_df.columns:
            temp_df = temp_df[temp_df[col_name].abs() < POSITIONING_TOLERANCE_BPS]
    else: # It's a ratio
        temp_df = temp_df[np.isclose(temp_df['sl_ratio'], definition['sl_def'])]

    # Filter by TP definition
    if isinstance(definition['tp_def'], str): # It's a positioning rule
        col_name = f"tp_dist_to_{definition['tp_def']}_bps"
        if col_name in temp_df.columns:
            temp_df = temp_df[temp_df[col_name].abs() < POSITIONING_TOLERANCE_BPS]
    else: # It's a ratio
        temp_df = temp_df[np.isclose(temp_df['tp_ratio'], definition['tp_def'])]
        
    return temp_df

if __name__ == "__main__":
    core_dir = os.path.dirname(os.path.abspath(__file__))
    # --- Define all required paths ---
    gold_features_dir = os.path.abspath(os.path.join(core_dir, '..', 'gold_data', 'features'))
    silver_outcomes_dir = os.path.abspath(os.path.join(core_dir, '..', 'silver_data', 'outcomes'))
    combinations_dir = os.path.abspath(os.path.join(core_dir, '..', 'platinum_data', 'combinations'))
    discovered_dir = os.path.abspath(os.path.join(core_dir, '..', 'platinum_data', 'discovered_strategy'))
    blacklist_dir = os.path.abspath(os.path.join(core_dir, '..', 'platinum_data', 'blacklists'))
    
    os.makedirs(discovered_dir, exist_ok=True)
    os.makedirs(blacklist_dir, exist_ok=True)

    combination_files = [f for f in os.listdir(combinations_dir) if f.endswith('.csv')]

    if not combination_files:
        print("❌ No combination files found in 'platinum_data/combinations'. Run the previous script first.")
    else:
        for fname in combination_files:
            print(f"\n{'='*25}\nProcessing: {fname}\n{'='*25}")
            
            # --- Define paths for the current instrument ---
            combinations_path = os.path.join(combinations_dir, fname)
            gold_path = os.path.join(gold_features_dir, fname)
            outcomes_path = os.path.join(silver_outcomes_dir, fname)
            discovered_path = os.path.join(discovered_dir, fname)
            blacklist_path = os.path.join(blacklist_dir, fname)

            if not os.path.exists(gold_path) or not os.path.exists(outcomes_path):
                print(f"⚠️ SKIPPING: Missing corresponding gold features or silver outcomes file for {fname}."); continue
            
            try:
                # --- Step 1: Load all definitions and state files ---
                all_definitions = pd.read_csv(combinations_path)
                
                try:
                    existing_results = pd.read_csv(discovered_path)
                except FileNotFoundError:
                    existing_results = pd.DataFrame(columns=['type', 'sl_def', 'tp_def'])
                
                try:
                    blacklist = pd.read_csv(blacklist_path)
                except FileNotFoundError:
                    blacklist = pd.DataFrame()

                # --- Step 2: Determine which strategies to process ---
                # Create a unique key for each definition to easily check for existence
                key_cols = ['type', 'sl_def', 'tp_def']
                all_definitions['key'] = [tuple(x) for x in all_definitions[key_cols].values]
                existing_results['key'] = [tuple(x) for x in existing_results[key_cols].values]
                blacklist['key'] = [tuple(x) for x in blacklist[key_cols].values]
                
                # Process a definition if it's NOT already discovered OR if it IS blacklisted
                definitions_to_process = all_definitions[
                    (~all_definitions['key'].isin(existing_results['key'])) | 
                    (all_definitions['key'].isin(blacklist['key']))
                ].drop(columns=['key'])
                
                if definitions_to_process.empty:
                    print("✅ All combinations have already been processed. Nothing new to discover."); continue
                
                print(f"Loaded {len(all_definitions)} total combinations.")
                print(f"Found {len(existing_results)} already discovered. Found {len(blacklist)} blacklisted.")
                print(f"Proceeding to analyze {len(definitions_to_process)} new or blacklisted combinations.")

                # --- Step 3: Load Gold Features (the ML data) ---
                gold_features_df = pd.read_csv(gold_path, parse_dates=['time'])
                
                all_new_strategies = []

                # --- Step 4: Main Discovery Loop ---
                for _, definition in tqdm(definitions_to_process.iterrows(), total=len(definitions_to_process), desc="Discovering Strategies"):
                    # --- Memory-Efficiently find all trades for this definition ---
                    outcomes_iterator = pd.read_csv(outcomes_path, chunksize=OUTCOMES_CHUNK_SIZE, parse_dates=['entry_time'])
                    matching_trades = [filter_chunk_for_definition(chunk, definition) for chunk in outcomes_iterator]
                    slice_df = pd.concat(matching_trades)

                    if len(slice_df) < MIN_TRADES_PER_DEFINITION: continue

                    # --- Prepare training data ---
                    slice_df['outcome'] = (slice_df['outcome'].str.lower().strip() == 'win').astype(int)
                    training_slice = pd.merge(gold_features_df, slice_df[['entry_time', 'outcome']], left_on='time', right_on='entry_time', how='inner')
                    
                    if len(training_slice) < MIN_TRADES_PER_DEFINITION: continue
                        
                    y = training_slice['outcome']
                    X = training_slice.drop(columns=['time', 'entry_time', 'outcome'])
                    
                    # --- Run ML model to find rules ---
                    model = DecisionTreeClassifier(max_depth=DECISION_TREE_MAX_DEPTH, min_samples_leaf=MIN_TRADES_PER_RULE, random_state=42, class_weight='balanced')
                    model.fit(X, y)
                    rules = get_rules_from_tree(model, list(X.columns))

                    for rule in rules:
                        all_new_strategies.append({
                            'type': definition['type'],
                            'sl_def': definition['sl_def'],
                            'tp_def': definition['tp_def'],
                            'market_rule': rule['rule'],
                            'win_prob': round(rule['win_prob'], 4),
                            'num_trades': rule['num_trades']
                        })
                
                # --- Step 5: Save all new results ---
                if all_new_strategies:
                    new_results_df = pd.DataFrame(all_new_strategies)
                    
                    # Combine with old results, drop duplicates (in case a blacklisted item was re-discovered)
                    # and save the master list of discovered strategies.
                    combined_results = pd.concat([existing_results.drop(columns=['key'], errors='ignore'), new_results_df])
                    combined_results.drop_duplicates(subset=['type', 'sl_def', 'tp_def', 'market_rule'], keep='last', inplace=True)
                    combined_results.sort_values(by=['win_prob', 'num_trades'], ascending=[False, False], inplace=True)
                    
                    combined_results.to_csv(discovered_path, index=False)
                    print(f"\n✅ Success! Discovered {len(new_results_df)} new potential strategy rules.")
                    print(f"Total strategies now in file: {len(combined_results)}. Results saved to: {discovered_path}")
                else:
                    print("\nℹ️ No new viable strategies were discovered in this run.")
                
            except Exception as e:
                print(f"\n❌ FAILED to process {fname}. Error: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "="*50 + "\n✅ All strategy discovery complete.")
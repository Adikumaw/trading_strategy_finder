# platinum_strategy_discoverer.py (Final Version with Dynamic Lift & Full Comments)

# Import standard libraries for data manipulation, file operations, and parallel processing
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, _tree
import os
from tqdm import tqdm
import gc
import hashlib
from multiprocessing import Pool, cpu_count
from functools import partial

# --- CONFIGURATION ---
# The blueprint must have at least this many historical examples to be considered for analysis.
MIN_CANDLES_FOR_ANALYSIS = 100 
# A final discovered rule must apply to at least this many of the blueprint's candles. Prevents overfitting.
MIN_CANDLES_PER_RULE = 50      
# Limits the complexity of discovered rules (e.g., max of 5 'AND' conditions).
DECISION_TREE_MAX_DEPTH = 10    
# NEW: The "Smart" Filter. A rule is only significant if its trade density is at least 1.5x the blueprint's average.
DENSITY_LIFT_THRESHOLD = 1.5   
# A safe default for multiprocessing, leaving cores free for the OS.
MAX_CPU_USAGE = max(1, cpu_count() - 2)

# --- FUNCTIONS ---
def get_rules_from_tree(tree, feature_names):
    """
    Extracts all valid rule paths from a trained Decision Tree that meet the minimum sample size.
    It no longer filters by density here; that logic is now handled dynamically.
    """
    # Get the internal structure and feature names from the trained tree model.
    tree_ = tree.tree_
    feature_name = [feature_names[i] if i != _tree.TREE_UNDEFINED else "!" for i in tree_.feature]
    rules = {}
    
    # Define a recursive function to walk down the tree's branches.
    def recurse(node, rule_path):
        # If the current node is a decision split (a branch)...
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name, threshold = feature_name[node], tree_.threshold[node]
            # ...recursively call this function for the left child (condition is True) and right child (condition is False).
            recurse(tree_.children_left[node], rule_path + [f"`{name}` <= {threshold:.4f}"])
            recurse(tree_.children_right[node], rule_path + [f"`{name}` > {threshold:.4f}"])
        # If the current node is a final prediction (a leaf)...
        else:
            num_samples = tree_.n_node_samples[node]
            # ...only consider it a valid rule if it's based on enough historical samples.
            if num_samples >= MIN_CANDLES_PER_RULE:
                # Save the complete rule path and the number of candles it applies to.
                rules[node] = {"rule": " and ".join(rule_path) or "all_trades", "num_candles": int(num_samples)}
    
    # Start the recursion from the root node (node 0) and return all valid rules found.
    recurse(0, []); return list(rules.values())

def process_definition_batch(definition, gold_features_df, targets_dir):
    """
    This is the core work done by each CPU core. It takes one strategy blueprint,
    runs the ML analysis, applies the dynamic "lift" filter, and returns any significant rules.
    """
    # Construct the path to the small, pre-computed target file for this specific blueprint.
    target_file = os.path.join(targets_dir, f"{definition['key']}.csv")
    
    # Abort early if the target file doesn't exist (meaning no trades were ever found for this blueprint).
    if not os.path.exists(target_file): return []
        
    try:
        # Read the pre-computed data, which contains 'entry_time' and 'trade_count'.
        candle_agg_df = pd.read_csv(target_file)
        # Perform safety checks and ensure the time column is a proper datetime object.
        if 'entry_time' not in candle_agg_df.columns or candle_agg_df.empty: return []
        candle_agg_df['entry_time'] = pd.to_datetime(candle_agg_df['entry_time'])
    except (pd.errors.EmptyDataError, ValueError):
        # Gracefully handle cases where the file is empty or corrupted.
        return []

    # Abort if the blueprint doesn't have enough historical examples to be statistically meaningful.
    if len(candle_agg_df) < MIN_CANDLES_FOR_ANALYSIS: return []
    
    # Join the target data (the 'y') with the feature data (the 'X').
    training_slice = pd.merge(gold_features_df, candle_agg_df, left_on='time', right_on='entry_time', how='inner')
    if training_slice.empty: return []
        
    # Separate the data into features (X) and the target variable to predict (y = trade_count).
    y, X = training_slice['trade_count'], training_slice.drop(columns=['time', 'entry_time', 'trade_count'])
    
    # Train the Decision Tree Regressor model. This is the core ML step.
    model = DecisionTreeRegressor(max_depth=DECISION_TREE_MAX_DEPTH, min_samples_leaf=MIN_CANDLES_PER_RULE, random_state=42).fit(X, y)
    
    # Extract all valid rule paths from the trained model.
    rules = get_rules_from_tree(model, list(X.columns))

    rules_found = []
    if rules:
        # --- NEW DYNAMIC FILTERING LOGIC ---
        # 1. Calculate the baseline average density for this entire blueprint.
        baseline_density = training_slice['trade_count'].mean()
        
        for rule in rules:
            # For each rule, find the subset of data it applies to.
            rule_candle_times = X.query(rule['rule']).index if rule['rule'] != 'all_trades' else X.index
            # Calculate the actual historical trade density for just that subset.
            actual_avg_density = training_slice.loc[rule_candle_times]['trade_count'].mean()
            
            # 2. Apply the "Lift" filter: Is this rule's density significantly better than the baseline?
            if actual_avg_density / baseline_density >= DENSITY_LIFT_THRESHOLD:
                # If yes, calculate the total trades and package the full strategy for saving.
                total_trades = training_slice.loc[rule_candle_times]['trade_count'].sum()
                rules_found.append({
                    'type': definition['type'], 'sl_def': definition['sl_def'], 'sl_bin': definition.get('sl_bin'),
                    'tp_def': definition['tp_def'], 'tp_bin': definition.get('tp_bin'),
                    'market_rule': rule['rule'], 'avg_trade_density': round(actual_avg_density, 2),
                    'num_candles': rule['num_candles'], 'total_trades': int(total_trades)
                })
    return rules_found

if __name__ == "__main__":
    # --- Setup and State Management (Unchanged and Correct) ---
    core_dir = os.path.dirname(os.path.abspath(__file__))
    gold_features_dir, combinations_dir, discovered_dir, blacklist_dir, targets_dir = [os.path.abspath(os.path.join(core_dir, '..', d)) for d in ['gold_data/features', 'platinum_data/combinations', 'platinum_data/discovered_strategy', 'platinum_data/blacklists', 'platinum_data/targets']]
    os.makedirs(discovered_dir, exist_ok=True); os.makedirs(blacklist_dir, exist_ok=True)
    
    combination_files = [f for f in os.listdir(combinations_dir) if f.endswith('.csv')]
    if not combination_files: print("❌ No combination files found.")
    else:
        print(f"Found {len(combination_files)} instrument(s) to process.")
        use_multiprocessing = input("Use multiprocessing? (y/n): ").strip().lower() == 'y'
        if use_multiprocessing: num_processes = MAX_CPU_USAGE
        else:
            try:
                num_processes = int(input(f"Enter number of processes to use (1-{cpu_count()}): ").strip())
                if num_processes < 1 or num_processes > cpu_count(): raise ValueError
            except ValueError: print("Invalid input. Defaulting to 1 process."); num_processes = 1
        
        for fname in combination_files:
            print(f"\n{'='*25}\nProcessing: {fname}\n{'='*25}")
            instrument_name = fname.replace('.csv', '')
            combinations_path, gold_path, discovered_path, blacklist_path = [os.path.join(d, fname) for d in [combinations_dir, gold_features_dir, discovered_dir, blacklist_dir]]
            instrument_target_dir = os.path.join(targets_dir, instrument_name)
            processed_log_path = os.path.join(discovered_dir, f".{fname}.processed_log")

            if not os.path.exists(instrument_target_dir):
                print(f"❌ Target files not found for {instrument_name}. Run extractor first."); continue
            
            all_definitions = pd.read_csv(combinations_path)
            all_definitions['sl_def'] = all_definitions['sl_def'].astype(object)
            all_definitions['tp_def'] = all_definitions['tp_def'].astype(object)
            if 'sl_bin' in all_definitions.columns: 
                all_definitions['sl_bin'] = all_definitions['sl_bin'].astype('Int64')
            if 'tp_bin' in all_definitions.columns: 
                all_definitions['tp_bin'] = all_definitions['tp_bin'].astype('Int64')
            
            try:
                with open(processed_log_path, 'r') as f: processed_keys = set(f.read().splitlines())
            except FileNotFoundError: processed_keys = set()
            try: blacklist = pd.read_csv(blacklist_path)
            except FileNotFoundError: blacklist = pd.DataFrame()
            
            key_cols = ['type', 'sl_def', 'sl_bin', 'tp_def', 'tp_bin']
            all_definitions['key'] = all_definitions[key_cols].astype(str).sum(axis=1).apply(lambda x: hashlib.sha256(x.encode()).hexdigest())
            if not blacklist.empty:
                blacklist['sl_def'] = blacklist['sl_def'].astype(object)
                blacklist['tp_def'] = blacklist['tp_def'].astype(object)
                if 'sl_bin' in blacklist.columns: 
                    blacklist['sl_bin'] = blacklist['sl_bin'].astype('Int64')
                if 'tp_bin' in blacklist.columns: 
                    blacklist['tp_bin'] = blacklist['tp_bin'].astype('Int64')
                blacklist['key'] = blacklist[key_cols].astype(str).sum(axis=1).apply(lambda x: hashlib.sha256(x.encode()).hexdigest())
                blacklisted_keys = set(blacklist['key'])
                processed_keys -= blacklisted_keys
                try:
                    existing_results = pd.read_csv(discovered_path)
                    existing_results['key'] = existing_results[key_cols].astype(str).sum(axis=1).apply(lambda x: hashlib.sha256(x.encode()).hexdigest())
                    cleaned_results = existing_results[~existing_results['key'].isin(blacklisted_keys)]
                    cleaned_results.drop(columns=['key']).to_csv(discovered_path, index=False)
                    print(f"Removed {len(existing_results) - len(cleaned_results)} old rule(s) for blacklisted blueprints.")
                except FileNotFoundError: 
                    pass
            
            definitions_to_process = all_definitions[~all_definitions['key'].isin(processed_keys)]
            if definitions_to_process.empty:
                print("✅ All combinations have already been processed for this instrument."); continue
            
            print(f"Analyzing {len(definitions_to_process)} new or blacklisted combinations.")
            temp_df = pd.read_csv(gold_path, nrows=1)
            dtype_map = {col: 'float32' for col in temp_df.columns if col not in ['time']}
            gold_features_df = pd.read_csv(gold_path, parse_dates=['time'], dtype=dtype_map)

            tasks = definitions_to_process.to_dict('records')
            effective_num_processes = min(num_processes, len(tasks))
            print(f"\n🚀 Starting discovery with {effective_num_processes} workers...")

            func = partial(process_definition_batch, gold_features_df=gold_features_df, targets_dir=instrument_target_dir)
            if effective_num_processes > 1:
                with Pool(processes=effective_num_processes) as pool:
                    results_nested = list(tqdm(pool.imap_unordered(func, tasks), total=len(tasks), desc="Discovering Strategies"))
            else:
                results_nested = [func(task) for task in tqdm(tasks, desc="Discovering Strategies")]
            
            all_rules = [rule for sublist in results_nested for rule in sublist]
            if all_rules:
                new_rules_df = pd.DataFrame(all_rules)
                file_exists = os.path.exists(discovered_path) and os.path.getsize(discovered_path) > 0
                new_rules_df.to_csv(discovered_path, mode='a', header=not file_exists, index=False)
            
            with open(processed_log_path, 'a') as log_file:
                for key in definitions_to_process['key']:
                    log_file.write(key + '\n')

            print("\nLoop finished. Performing final cleanup...")
            try:
                final_df = pd.read_csv(discovered_path, dtype={'sl_def': object, 'tp_def': object})
                final_df.drop_duplicates(subset=key_cols + ['market_rule'], keep='last', inplace=True)
                final_df.sort_values(by=['avg_trade_density', 'num_candles'], ascending=[False, False], inplace=True)
                final_df.to_csv(discovered_path, index=False)
                print(f"✅ Cleanup complete. Total unique strategies: {len(final_df)}")
            except FileNotFoundError:
                print("ℹ️ No strategies were discovered.")
                
    print("\n" + "="*50 + "\n✅ All strategy discovery complete.")
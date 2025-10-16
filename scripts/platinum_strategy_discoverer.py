# platinum_strategy_discoverer.py (Corrected Blacklist Logic)

"""
Platinum Layer - Stage 3: The Strategy Discoverer (The Rule Miner)

This script is the final, high-speed machine learning stage of the Platinum
discovery engine. It is the intelligent heart of the entire pipeline, where
raw data patterns become actionable intelligence.

Its purpose is to act as a powerful noise filter and pattern detector. It
iterates through each strategy blueprint from Stage 1, loads the pre-computed
target data from Stage 2, and uses a Decision Tree model to find explicit,
human-readable trading rules that identify market conditions with a high
density of historical winning trades.

Key features of this stage include:
- Interpretability: Uses a `DecisionTreeRegressor` specifically because its
  output is a set of simple, human-readable rules (e.g., `RSI <= 30`), not a
  "black box" prediction.
- Statistical Significance: Implements a "lift" filter to ensure that it only
  saves rules that identify market conditions where a strategy performs
  significantly better than its own baseline average.
- Intelligent Feedback Loop: Leverages a blacklist fed back from the
  Diamond/Zircon layers to ignore blueprints that have already been proven
  unprofitable, making the system smarter and more efficient over time.
"""

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

# The minimum number of historical candles with trade data required for a
# blueprint to be considered for ML analysis. Prevents modeling on sparse data.
MIN_CANDLES_FOR_ANALYSIS = 100

# The minimum number of candles a final rule must apply to. This prevents
# overfitting by ignoring rules based on very few historical examples.
MIN_CANDLES_PER_RULE = 50

# The maximum depth of the Decision Tree. A lower number (e.g., 4-5) encourages
# simpler, more generalizable rules, while a higher number allows for more
# complex, specific rules.
DECISION_TREE_MAX_DEPTH = 10

# The core statistical filter. A rule is only saved if the density of winning
# trades it identifies is at least this many times greater than the baseline
# density of its parent blueprint. (e.g., 1.5 means a 50% improvement).
DENSITY_LIFT_THRESHOLD = 1.5

# Sets the maximum number of CPU cores to use for multiprocessing.
MAX_CPU_USAGE = max(1, cpu_count() - 2)


# --- CORE FUNCTIONS ---

def get_rules_from_tree(tree, feature_names):
    """
    Extracts all valid rule paths from a trained Decision Tree.

    This function recursively walks through the tree's nodes and constructs a
    human-readable rule string for each path from the root to a leaf node that
    meets the `MIN_CANDLES_PER_RULE` requirement.

    Args:
        tree (DecisionTreeClassifier): The trained scikit-learn tree model.
        feature_names (list): A list of the feature names used to train the tree.

    Returns:
        list: A list of dictionaries, where each dictionary represents a valid rule.
    """
    tree_ = tree.tree_
    feature_name = [feature_names[i] if i != _tree.TREE_UNDEFINED else "!" for i in tree_.feature]
    rules = {}
    
    def recurse(node, rule_path):
        """A recursive helper function to traverse the tree."""
        # If the node is a split node (not a leaf)
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name, threshold = feature_name[node], tree_.threshold[node]
            # Recurse down the left branch (condition is True)
            recurse(tree_.children_left[node], rule_path + [f"`{name}` <= {threshold:.4f}"])
            # Recurse down the right branch (condition is False)
            recurse(tree_.children_right[node], rule_path + [f"`{name}` > {threshold:.4f}"])
        # If the node is a leaf node
        else:
            num_samples = tree_.n_node_samples[node]
            # Only consider the rule valid if it's based on enough samples
            if num_samples >= MIN_CANDLES_PER_RULE:
                rules[node] = {"rule": " and ".join(rule_path) or "all_trades", "num_candles": int(num_samples)}
    
    recurse(0, [])
    return list(rules.values())

def process_definition_batch(definition, gold_features_df, targets_dir):
    """
    The core worker function for multiprocessing. It processes a single
    strategy blueprint, runs the ML analysis, applies the "lift" filter,
    and returns any statistically significant rules found.

    Args:
        definition (dict): A dictionary representing one strategy blueprint.
        gold_features_df (pd.DataFrame): The full ML-ready features dataset.
        targets_dir (str): The directory containing the pre-computed target files.

    Returns:
        list: A list of dictionaries, where each dictionary is a complete,
              discovered strategy ready to be saved.
    """
    target_file = os.path.join(targets_dir, f"{definition['key']}.csv")
    
    if not os.path.exists(target_file):
        return [] # Skip if no trades were ever found for this blueprint.
    try:
        candle_agg_df = pd.read_csv(target_file)
        if 'entry_time' not in candle_agg_df.columns or candle_agg_df.empty:
            return []
        candle_agg_df['entry_time'] = pd.to_datetime(candle_agg_df['entry_time'])
    except (pd.errors.EmptyDataError, ValueError):
        return [] # Handle empty or corrupted target files.

    if len(candle_agg_df) < MIN_CANDLES_FOR_ANALYSIS:
        return [] # Skip if not enough historical data.
    
    # --- Join target data (y) with feature data (X) ---
    training_slice = pd.merge(gold_features_df, candle_agg_df, left_on='time', right_on='entry_time', how='inner')
    if training_slice.empty:
        return []
        
    y, X = training_slice['trade_count'], training_slice.drop(columns=['time', 'entry_time', 'trade_count'])
    
    # --- Train the Decision Tree Model ---
    model = DecisionTreeRegressor(
        max_depth=DECISION_TREE_MAX_DEPTH,
        min_samples_leaf=MIN_CANDLES_PER_RULE,
        random_state=42
    ).fit(X, y)
    
    rules = get_rules_from_tree(model, list(X.columns))
    rules_found = []
    if rules:
        # --- DYNAMIC LIFT FILTERING ---
        # 1. Calculate the baseline average trade density for the entire blueprint.
        baseline_density = training_slice['trade_count'].mean()
        for rule in rules:
            # 2. For each rule, calculate the density for just the subset of data it applies to.
            rule_candle_times = X.query(rule['rule']).index if rule['rule'] != 'all_trades' else X.index
            actual_avg_density = training_slice.loc[rule_candle_times]['trade_count'].mean()
            
            # 3. Apply the Lift filter: Is this rule's density significantly better?
            if actual_avg_density / baseline_density >= DENSITY_LIFT_THRESHOLD:
                total_trades = training_slice.loc[rule_candle_times]['trade_count'].sum()
                # If it passes, package the full strategy definition for saving.
                rules_found.append({
                    'type': definition['type'], 'sl_def': definition['sl_def'], 'sl_bin': definition.get('sl_bin'),
                    'tp_def': definition['tp_def'], 'tp_bin': definition.get('tp_bin'),
                    'market_rule': rule['rule'], 'avg_trade_density': round(actual_avg_density, 2),
                    'num_candles': rule['num_candles'], 'total_trades': int(total_trades)
                })
    return rules_found

if __name__ == "__main__":
    # --- Setup Project Directories ---
    core_dir = os.path.dirname(os.path.abspath(__file__))
    gold_features_dir, combinations_dir, discovered_dir, blacklist_dir, targets_dir = [os.path.abspath(os.path.join(core_dir, '..', d)) for d in ['gold_data/features', 'platinum_data/combinations', 'platinum_data/discovered_strategy', 'platinum_data/blacklists', 'platinum_data/targets']]
    os.makedirs(discovered_dir, exist_ok=True); os.makedirs(blacklist_dir, exist_ok=True)
    
    combination_files = [f for f in os.listdir(combinations_dir) if f.endswith('.csv')]
    if not combination_files:
        print("❌ No combination files found.")
    else:
        # --- Configure Multiprocessing ---
        print(f"Found {len(combination_files)} instrument(s) to process.")
        use_multiprocessing = input("Use multiprocessing? (y/n): ").strip().lower() == 'y'
        if use_multiprocessing:
            num_processes = MAX_CPU_USAGE
        else:
            try: num_processes = int(input(f"Enter number of processes to use (1-{cpu_count()}): ").strip())
            except ValueError: num_processes = 1
        
        # --- Main Loop: Process each instrument ---
        for fname in combination_files:
            print(f"\n{'='*25}\nProcessing: {fname}\n{'='*25}")
            instrument_name = fname.replace('.csv', '')
            combinations_path, gold_path, discovered_path, blacklist_path = [os.path.join(d, fname) for d in [combinations_dir, gold_features_dir, discovered_dir, blacklist_dir]]
            instrument_target_dir = os.path.join(targets_dir, instrument_name)
            processed_log_path = os.path.join(discovered_dir, f".{fname}.processed_log")

            if not os.path.exists(instrument_target_dir):
                print(f"❌ Target files not found for {instrument_name}. Run extractor first."); continue
            
            all_definitions = pd.read_csv(combinations_path)
            # Ensure correct dtypes for key generation
            all_definitions['sl_def'] = all_definitions['sl_def'].astype(object); all_definitions['tp_def'] = all_definitions['tp_def'].astype(object)
            if 'sl_bin' in all_definitions.columns: all_definitions['sl_bin'] = all_definitions['sl_bin'].astype('Int64')
            if 'tp_bin' in all_definitions.columns: all_definitions['tp_bin'] = all_definitions['tp_bin'].astype('Int64')
            key_cols = ['type', 'sl_def', 'sl_bin', 'tp_def', 'tp_bin']
            all_definitions['key'] = all_definitions[key_cols].astype(str).sum(axis=1).apply(lambda x: hashlib.sha256(x.encode()).hexdigest())

            # --- Blacklist and State Management Logic ---

            # 1. Load the blacklist of failed blueprints from the backtesting layers.
            try:
                blacklist = pd.read_csv(blacklist_path)
                blacklist['sl_def'] = blacklist['sl_def'].astype(object); blacklist['tp_def'] = blacklist['tp_def'].astype(object)
                if 'sl_bin' in blacklist.columns: blacklist['sl_bin'] = blacklist['sl_bin'].astype('Int64')
                if 'tp_bin' in blacklist.columns: blacklist['tp_bin'] = blacklist['tp_bin'].astype('Int64')
                blacklist['key'] = blacklist[key_cols].astype(str).sum(axis=1).apply(lambda x: hashlib.sha256(x.encode()).hexdigest())
                blacklisted_keys = set(blacklist['key'])
                print(f"Found {len(blacklisted_keys)} blacklisted blueprints to ignore.")
            except FileNotFoundError:
                blacklisted_keys = set()
                print("No blacklist file found. Processing all blueprints.")

            # 2. Load the log of blueprints already processed in previous runs for resumability.
            try:
                with open(processed_log_path, 'r') as f: processed_keys = set(f.read().splitlines())
            except FileNotFoundError:
                processed_keys = set()

            # 3. Self-healing: Remove discovered rules if their parent blueprint is now blacklisted.
            if blacklisted_keys:
                try:
                    existing_results = pd.read_csv(discovered_path)
                    existing_results['key'] = existing_results[key_cols].astype(str).sum(axis=1).apply(lambda x: hashlib.sha256(x.encode()).hexdigest())
                    cleaned_results = existing_results[~existing_results['key'].isin(blacklisted_keys)]
                    if len(existing_results) > len(cleaned_results):
                        cleaned_results.drop(columns=['key']).to_csv(discovered_path, index=False)
                        print(f"Sanitized results: Removed {len(existing_results) - len(cleaned_results)} old rule(s) for now-blacklisted blueprints.")
                except FileNotFoundError:
                    pass # No existing results to clean.

            # 4. Determine the final list of definitions to process in this run.
            # First, filter out any blueprints on the blacklist.
            definitions_to_process = all_definitions[~all_definitions['key'].isin(blacklisted_keys)]
            # Then, from the remaining valid list, filter out those already processed.
            definitions_to_process = definitions_to_process[~definitions_to_process['key'].isin(processed_keys)]
            
            if definitions_to_process.empty:
                print("✅ All valid combinations have already been processed for this instrument."); continue
            
            print(f"Found {len(definitions_to_process)} new or updated combinations to analyze.")
            
            # --- Execute Processing ---
            gold_features_df = pd.read_csv(gold_path, parse_dates=['time'])
            tasks = definitions_to_process.to_dict('records')
            effective_num_processes = min(num_processes, len(tasks))
            print(f"\n🚀 Starting discovery with {effective_num_processes} workers...")

            func = partial(process_definition_batch, gold_features_df=gold_features_df, targets_dir=instrument_target_dir)
            if effective_num_processes > 1:
                with Pool(processes=effective_num_processes) as pool:
                    results_nested = list(tqdm(pool.imap_unordered(func, tasks), total=len(tasks), desc="Discovering Strategies"))
            else:
                results_nested = [func(task) for task in tqdm(tasks, desc="Discovering Strategies")]
            
            # --- Save Results and Log Progress ---
            all_rules = [rule for sublist in results_nested for rule in sublist if sublist]
            if all_rules:
                new_rules_df = pd.DataFrame(all_rules)
                file_exists = os.path.exists(discovered_path) and os.path.getsize(discovered_path) > 0
                new_rules_df.to_csv(discovered_path, mode='a', header=not file_exists, index=False)
            
            # Log the keys of the blueprints that were just processed.
            with open(processed_log_path, 'a') as log_file:
                for key in definitions_to_process['key']:
                    log_file.write(key + '\n')

            # --- Final Cleanup ---
            print("\nLoop finished. Performing final cleanup...")
            try:
                final_df = pd.read_csv(discovered_path, dtype={'sl_def': object, 'tp_def': object})
                final_df.drop_duplicates(subset=key_cols + ['market_rule'], keep='last', inplace=True)
                final_df.sort_values(by=['avg_trade_density', 'num_candles'], ascending=[False, False], inplace=True)
                final_df.to_csv(discovered_path, index=False)
                print(f"✅ Cleanup complete. Total unique strategies: {len(final_df)}")
            except FileNotFoundError:
                print("ℹ️ No new strategies were discovered in this run.")
                
    print("\n" + "="*50 + "\n✅ All strategy discovery complete.")
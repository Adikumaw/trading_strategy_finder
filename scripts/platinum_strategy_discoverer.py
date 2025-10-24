# platinum_strategy_discoverer.py (Upgraded with Key-Based Logic)

"""
Platinum Layer - Stage 3: The Strategy Discoverer (The Rule Miner)

This script is the final, high-speed machine learning stage of the Platinum
discovery engine. It is the intelligent heart of the entire pipeline, where
raw data patterns become actionable intelligence.

Its purpose is to act as a powerful noise filter and pattern detector. It
iterates through each strategy blueprint (identified by a unique 'key'), loads
its pre-computed target data, and uses a Decision Tree model to find explicit,
human-readable trading rules that identify market conditions with a high
density of historical winning trades.

Key features of this stage include:
- Interpretability: Uses a `DecisionTreeRegressor` specifically because its
  output is a set of simple, human-readable rules (e.g., `RSI <= 30`), not a
  "black box" prediction.
- Statistical Significance: Implements a "lift" filter to ensure that it only
  saves rules that identify market conditions where a strategy performs
  significantly better than its own baseline average.
- Key-Based Feedback Loop: Reads blueprint keys directly from the combinations
  and blacklist files. This robust "single source of truth" approach makes the
  system efficient and eliminates brittle, repeated hashing logic.
- Self-Healing: Automatically purges old results if their parent blueprint
  is added to the blacklist, ensuring the output remains clean.
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, _tree
import os
from tqdm import tqdm
import gc
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
# simpler, more generalizable rules.
DECISION_TREE_MAX_DEPTH = 10

# A rule is only saved if the density of winning trades it identifies is at
# least this many times greater than the blueprint's baseline density.
DENSITY_LIFT_THRESHOLD = 1.5

# Sets the maximum number of CPU cores to use for multiprocessing.
MAX_CPU_USAGE = max(1, cpu_count() - 2)


# --- CORE FUNCTIONS ---

def get_rules_from_tree(tree, feature_names):
    """
    Extracts human-readable rule paths from a trained scikit-learn Decision Tree.

    This function recursively traverses the nodes of a fitted Decision Tree.
    For each path from the root to a leaf node that represents a sufficient
    number of samples (as defined by `MIN_CANDLES_PER_RULE`), it constructs a
    string representing the sequence of decisions (e.g., "RSI <= 35 and ADX > 25").

    Args:
        tree (DecisionTreeRegressor): The trained scikit-learn tree model object.
        feature_names (list): A list of the feature names (X.columns) that the
                              tree was trained on, in the correct order.

    Returns:
        list: A list of dictionaries, where each dictionary contains the 'rule'
              string and the 'num_candles' (samples) that the rule applies to.
    """
    tree_ = tree.tree_
    feature_name = [feature_names[i] if i != _tree.TREE_UNDEFINED else "!" for i in tree_.feature]
    rules = {}
    
    def recurse(node, rule_path):
        """A recursive helper function to traverse the tree."""
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name, threshold = feature_name[node], tree_.threshold[node]
            recurse(tree_.children_left[node], rule_path + [f"`{name}` <= {threshold:.4f}"])
            recurse(tree_.children_right[node], rule_path + [f"`{name}` > {threshold:.4f}"])
        else:
            num_samples = tree_.n_node_samples[node]
            if num_samples >= MIN_CANDLES_PER_RULE:
                rules[node] = {"rule": " and ".join(rule_path) or "all_trades", "num_candles": int(num_samples)}
    
    recurse(0, [])
    return list(rules.values())

def process_definition_batch(definition, gold_features_df, targets_dir):
    """
    Performs the complete ML analysis for a single strategy blueprint.

    This is the core worker function executed in parallel. For a given strategy
    blueprint, it loads the pre-computed target data, merges it with the market
    features, trains a Decision Tree Regressor, extracts the rules, and applies
    a statistical "lift" filter to identify only the most promising rules.

    Args:
        definition (dict): A dictionary representing one strategy blueprint,
                           which MUST include the 'key'.
        gold_features_df (pd.DataFrame): The full, ML-ready market features dataset (X).
        targets_dir (str): The directory containing the pre-computed target files
                           (y), which are named by their corresponding key.

    Returns:
        list: A list of dictionaries, where each dictionary is a complete,
              statistically significant discovered strategy. Returns an empty
              list if no significant rules are found.
    """
    target_file = os.path.join(targets_dir, f"{definition['key']}.csv")
    
    if not os.path.exists(target_file):
        return []
    try:
        candle_agg_df = pd.read_csv(target_file, parse_dates=['entry_time'])
        if candle_agg_df.empty:
            return []
    except (pd.errors.EmptyDataError, ValueError):
        return []

    if len(candle_agg_df) < MIN_CANDLES_FOR_ANALYSIS:
        return []
    
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
        baseline_density = training_slice['trade_count'].mean()
        if baseline_density == 0: return [] # Avoid division by zero if blueprint has no wins
        
        for rule in rules:
            rule_subset = X.query(rule['rule']) if rule['rule'] != 'all_trades' else X
            if rule_subset.empty: continue
            
            actual_avg_density = training_slice.loc[rule_subset.index, 'trade_count'].mean()
            
            # Apply the Lift filter: Is this rule's density significantly better?
            if (actual_avg_density / baseline_density) >= DENSITY_LIFT_THRESHOLD:
                total_trades = training_slice.loc[rule_subset.index, 'trade_count'].sum()
                # Package the full strategy definition, including its key, for saving.
                rules_found.append({
                    'key': definition['key'], 'type': definition['type'], 'sl_def': definition['sl_def'], 
                    'sl_bin': definition.get('sl_bin'), 'tp_def': definition['tp_def'], 'tp_bin': definition.get('tp_bin'),
                    'market_rule': rule['rule'], 'avg_trade_density': round(actual_avg_density, 2),
                    'num_candles': rule['num_candles'], 'total_trades': int(total_trades)
                })
    return rules_found

if __name__ == "__main__":
    # --- Setup Project Directories ---
    core_dir = os.path.dirname(os.path.abspath(__file__))
    gold_features_dir, combinations_dir, discovered_dir, blacklist_dir, targets_dir = [os.path.abspath(os.path.join(core_dir, '..', d)) for d in ['gold_data/features', 'platinum_data/combinations', 'platinum_data/discovered_strategy', 'platinum_data/blacklists', 'platinum_data/targets']]
    os.makedirs(discovered_dir, exist_ok=True); os.makedirs(blacklist_dir, exist_ok=True)
    
    try:
        combination_files = [f for f in os.listdir(combinations_dir) if f.endswith('.csv')]
    except FileNotFoundError:
        print(f"❌ Directory not found: {combinations_dir}"); combination_files = []

    if not combination_files:
        print("❌ No combination files found.")
    else:
        # --- Configure Multiprocessing ---
        print(f"Found {len(combination_files)} instrument(s) to process.")
        use_multiprocessing = input("Use multiprocessing? (y/n): ").strip().lower() == 'y'
        if use_multiprocessing: num_processes = MAX_CPU_USAGE
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
            
            # --- Load Blueprints and Validate Keys ---
            all_definitions = pd.read_csv(combinations_path)
            if 'key' not in all_definitions.columns:
                print(f"❌ FATAL ERROR: 'key' column not found in {fname}. Run the target extractor script first to generate keys.")
                continue

            # --- Blacklist and State Management Logic ---
            # 1. Load the blacklist of failed blueprint KEYS. This is now highly efficient.
            try:
                # The blacklist file is now expected to contain just one column: 'key'.
                blacklist = pd.read_csv(blacklist_path)
                blacklisted_keys = set(blacklist['key'])
                print(f"Found {len(blacklisted_keys)} blacklisted blueprints to ignore.")
            except (FileNotFoundError, pd.errors.EmptyDataError):
                blacklisted_keys = set()
                print("No blacklist file found or file is empty. Processing all blueprints.")

            # 2. Load the log of blueprints already processed in previous runs.
            try:
                with open(processed_log_path, 'r') as f: processed_keys = set(f.read().splitlines())
            except FileNotFoundError:
                processed_keys = set()

            # 3. Self-healing: Remove discovered rules if their parent blueprint is now blacklisted.
            if blacklisted_keys:
                try:
                    existing_results = pd.read_csv(discovered_path)
                    # The 'key' column is now expected to exist in the discovered strategies file.
                    if 'key' in existing_results.columns:
                        cleaned_results = existing_results[~existing_results['key'].isin(blacklisted_keys)]
                        if len(existing_results) > len(cleaned_results):
                            cleaned_results.to_csv(discovered_path, index=False)
                            print(f"Sanitized results: Removed {len(existing_results) - len(cleaned_results)} old rule(s) for now-blacklisted blueprints.")
                except FileNotFoundError:
                    pass # No existing results to clean.

            # 4. Determine the final list of definitions to process in this run.
            definitions_to_process = all_definitions[~all_definitions['key'].isin(blacklisted_keys)]
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
                file_exists = os.path.exists(discovered_path)
                new_rules_df.to_csv(discovered_path, mode='a', header=not file_exists, index=False)
            
            with open(processed_log_path, 'a') as log_file:
                for key in definitions_to_process['key']:
                    log_file.write(key + '\n')

            # --- Final Cleanup ---
            print("\nLoop finished. Performing final cleanup...")
            try:
                final_df = pd.read_csv(discovered_path)
                # Define columns that uniquely identify a strategy rule
                unique_cols = ['key', 'market_rule']
                final_df.drop_duplicates(subset=unique_cols, keep='last', inplace=True)
                final_df.sort_values(by=['avg_trade_density', 'num_candles'], ascending=[False, False], inplace=True)
                final_df.to_csv(discovered_path, index=False)
                print(f"✅ Cleanup complete. Total unique strategies: {len(final_df)}")
            except FileNotFoundError:
                print("ℹ️ No new strategies were discovered in this run.")
                
    print("\n" + "="*50 + "\n✅ All strategy discovery complete.")
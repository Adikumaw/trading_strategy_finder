# platinum_strategy_discoverer.py (V2.0 - Two-Phase Learning Architecture)

"""
Platinum Layer - Strategy Discoverer: The Rule Miner

This is the intelligent heart of the discovery pipeline. It uses a machine
learning model (DecisionTreeRegressor) to find explicit, human-readable trading
rules within the vast dataset of successful trade blueprints.

This script embodies a "Two-Phase Learning" architecture for maximum efficiency
and continuous improvement:

Phase 1: Discovery
- A comprehensive, one-time run to find initial rules for all new and
  unprocessed strategy blueprints. This phase is fully resumable. It finds
  high-quality patterns in blueprints that have never been seen before.

Phase 2: Iterative Improvement
- A fast, targeted run that ONLY re-evaluates blueprints for which the
  backtester has provided new negative feedback (i.e., new blacklisted rules).
  This is the core of the iterative learning loop. It uses a "data pruning"
  technique to force the Decision Tree to find novel, alternative rules,
  avoiding patterns that are known to be unprofitable or have already been found.

The entire process is highly parallelized, resumable, and designed to become
smarter and more efficient over time as it learns from backtesting results.
"""

import os
import re
import sys
import shutil
import traceback
from multiprocessing import Pool, cpu_count, current_process
from collections import defaultdict
from typing import Dict, List, Set, Tuple
from functools import partial

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm

# --- CONFIGURATION ---
MAX_CPU_USAGE: int = max(1, cpu_count() - 2)
# Pre-filtering: Ignore blueprints with fewer than this many unique candles.
MIN_CANDLE_LIMIT: int = 100
# ML Model: The max depth of the Decision Tree. Deeper trees can find more complex rules.
DECISION_TREE_MAX_DEPTH: int = 7
# Rule Quality: A discovered rule must apply to at least this many candles.
MIN_CANDLES_PER_RULE: int = 50
# Rule Quality: A rule's avg trade density must be this much better than the blueprint's baseline.
DENSITY_LIFT_THRESHOLD: float = 1.5
# Batch Size: Number of keys to process per worker task.
BATCH_SIZE: int = 20
# Housekeeping: How many batches to process before flushing results to disk.
BUFFER_FLUSH_SIZE: int = 5

# --- WORKER-SPECIFIC GLOBALS & INITIALIZER ---
worker_gold_features_df: pd.DataFrame

def init_worker(gold_features_df: pd.DataFrame):
    """Initializer for each worker process in the Pool."""
    global worker_gold_features_df
    worker_gold_features_df = gold_features_df

# --- NEW: ROBUST FILE LOADING HELPER ---
def safe_load_series(filepath: str, column_name: str = None, has_header: bool = True) -> pd.Series:
    """
    Safely loads a single column from a CSV file into a Pandas Series.
    Returns an empty Series if the file is empty, preventing errors.
    """
    try:
        if os.path.getsize(filepath) > 0:
            header = 0 if has_header else None
            df = pd.read_csv(filepath, header=header)
            if not df.empty:
                col = column_name if has_header else 0
                return df[col].dropna()
    except (FileNotFoundError, pd.errors.EmptyDataError):
        pass # The file doesn't exist or is empty, which is a valid state.
    except Exception as e:
        print(f"[WARNING] Could not read file {filepath}. Error: {e}")
    
    # Return an empty Series with a specified dtype to ensure consistency
    return pd.Series(dtype='object')

# --- CORE ML & RULE EXTRACTION LOGIC ---

def get_rule_from_tree(tree, feature_names):
    """A helper function to recursively traverse a trained Decision Tree and extract rules."""
    rules = []
    
    def recurse(node, path):
        if tree.feature[node] != -2: # Not a leaf
            name = feature_names[tree.feature[node]]
            threshold = round(tree.threshold[node], 5)
            # Recurse left (<=)
            left_path = path + [f"`{name}` <= {threshold}"]
            recurse(tree.children_left[node], left_path)
            # Recurse right (>)
            right_path = path + [f"`{name}` > {threshold}"]
            recurse(tree.children_right[node], right_path)
        else: # Is a leaf
            rules.append({
                'rule': " and ".join(path),
                'n_candles': tree.n_node_samples[node],
                'avg_density': tree.value[node][0][0]
            })

    recurse(0, [])
    return rules

def find_rules_with_decision_tree(
    training_df: pd.DataFrame, 
    exclusion_rules: Set[str]
) -> Dict:
    """
    Trains a Decision Tree on the provided data to find high-quality, novel rules.

    This function implements the "Intelligent Pruning" logic. It first "poisons"
    the training data by setting the target variable to zero for all candles that
    match a known rule from the exclusion list. It then trains the model and
    extracts any new, high-quality rules that it discovers.

    Args:
        training_df: A DataFrame containing the merged Gold features and target data.
        exclusion_rules: A set of rule strings that should be ignored.

    Returns:
        A dictionary indicating the status ('success' or 'exhausted') and a list
        of any valid new rules that were discovered.
    """
    # --- Data Pruning / "Poisoning" ---
    pruned_df = training_df.copy()
    baseline_avg_density = pruned_df['trade_count'].mean()
    if baseline_avg_density == 0:
        return {'status': 'exhausted'}

    for rule_str in exclusion_rules:
        try:
            indices = pruned_df.query(rule_str).index
            if not indices.empty:
                pruned_df.loc[indices, 'trade_count'] = 0
        except Exception:
            # Ignore rules that may be malformed or no longer apply
            pass

    X = pruned_df.drop(columns=['time', 'trade_count'])
    y = pruned_df['trade_count']
    
    if y.sum() == 0: # All potential good candles were pruned
        return {'status': 'exhausted'}

    # --- Model Training ---
    model = DecisionTreeRegressor(
        max_depth=DECISION_TREE_MAX_DEPTH,
        min_samples_leaf=MIN_CANDLES_PER_RULE,
        random_state=42
    )
    model.fit(X, y)

    # --- Rule Extraction and Quality Filtering ---
    all_rules = get_rule_from_tree(model.tree_, X.columns)
    valid_new_rules = []
    for rule_info in all_rules:
        # Quality Check 1: Does the rule's performance exceed the lift threshold?
        if rule_info['avg_density'] >= (baseline_avg_density * DENSITY_LIFT_THRESHOLD):
            # Quality Check 2: Is the rule novel (not in the exclusion list)?
            if rule_info['rule'] and rule_info['rule'] not in exclusion_rules:
                valid_new_rules.append(rule_info)
    
    if not valid_new_rules:
        return {'status': 'exhausted'}
    
    return {'status': 'success', 'rules': valid_new_rules}

# --- PARALLEL WORKER FUNCTION ---

def process_key_batch(
    task_tuple: Tuple[List[str], Dict[str, Set[str]], Dict[str, str]]
) -> Dict:
    key_batch, exclusion_rules_by_key, base_dirs = task_tuple
    global worker_gold_features_df
    
    discovered_in_batch, exhausted_in_batch = [], []
    merge_failures = 0
    
    for key in key_batch:
        try:
            target_path = os.path.join(base_dirs['platinum_final'], f"{key}.csv")
            target_df = pd.read_csv(target_path, parse_dates=['entry_time'])
            if target_df.empty:
                merge_failures += 1
                continue

            # --- MODIFIED: Correct merge logic for different column names ---
            training_df = pd.merge(
                worker_gold_features_df, 
                target_df, 
                left_on='time', 
                right_on='entry_time', 
                how='inner'
            ).drop(columns=['entry_time']) # Drop redundant time column
            
            if training_df.empty:
                merge_failures += 1
                continue

            exclusion_rules_for_this_key = exclusion_rules_by_key.get(key, set())
            result = find_rules_with_decision_tree(training_df, exclusion_rules_for_this_key)
            
            if result['status'] == 'success':
                for rule_info in result['rules']:
                    discovered_in_batch.append({'key': key, 'market_rule': rule_info['rule'], 'n_candles': rule_info['n_candles'], 'avg_density': rule_info['avg_density']})
            elif result['status'] == 'exhausted':
                exhausted_in_batch.append(key)
        
        except FileNotFoundError:
            merge_failures += 1
            continue
        except Exception:
            print(f"[WORKER ERROR] Failed to process key {key}.")
            traceback.print_exc()
            
    return {'strategies': discovered_in_batch, 'exhausted_keys': exhausted_in_batch, 'processed_keys': key_batch, 'merge_failures': merge_failures}

# --- HELPER & ORCHESTRATION FUNCTIONS ---
def flush_buffers(buffers: Dict, paths: Dict, is_discovery: bool):
    if buffers['strategies']:
        pd.DataFrame(buffers['strategies']).to_csv(paths['discovered'], mode='a', header=False, index=False)
        buffers['strategies'].clear()
    if buffers['exhausted']:
        pd.Series(buffers['exhausted'], name='key').to_csv(paths['exhausted'], mode='a', header=False, index=False)
        buffers['exhausted'].clear()
    if is_discovery and buffers['processed']:
        pd.Series(buffers['processed'], name='key').to_csv(paths['processed'], mode='a', header=False, index=False)
        buffers['processed'].clear()

def execute_parallel_processing(
    keys_to_process: List[str],
    exclusion_rules_by_key: Dict[str, Set[str]],
    base_dirs: Dict[str, str],
    is_discovery_phase: bool
):
    """The generic parallel processing engine."""
    if not keys_to_process:
        print("[INFO] No keys to process in this phase.")
        return

    # Split keys into manageable batches for workers
    num_batches = int(np.ceil(len(keys_to_process) / BATCH_SIZE))
    key_batches = np.array_split(keys_to_process, num_batches) if num_batches > 0 else []
    
    tasks = [(batch.tolist(), exclusion_rules_by_key, base_dirs) for batch in key_batches]
    
    buffers = {'strategies': [], 'exhausted': [], 'processed': []}
    paths = {
        'discovered': os.path.join(base_dirs['platinum_strategies'], f"{base_dirs['instrument_name']}.csv"),
        'exhausted': os.path.join(base_dirs['platinum_exhausted'], f"{base_dirs['instrument_name']}.csv"),
        'processed': os.path.join(base_dirs['platinum_logs'], f"{base_dirs['instrument_name']}.processed")
    }

    batches_processed = 0
    with Pool(processes=MAX_CPU_USAGE, initializer=init_worker, initargs=(worker_gold_features_df,)) as pool:
        with tqdm(total=len(tasks), desc="Processing Batches") as pbar:
            for result in pool.imap_unordered(process_key_batch, tasks):
                if result['strategies']:
                    buffers['strategies'].extend(result['strategies'])
                if result['exhausted_keys']:
                    buffers['exhausted'].extend(result['exhausted_keys'])
                
                # Add all keys from the processed batch to the log buffer
                # We log the whole batch upon return, confirming its completion.
                original_keys_in_batch = [key for key in result['strategies']]
                buffers['processed'].extend(key['key'] for key in original_keys_in_batch)
                buffers['processed'].extend(result['exhausted_keys'])
                
                batches_processed += 1
                if batches_processed % BUFFER_FLUSH_SIZE == 0:
                    flush_buffers(buffers, paths, is_discovery_phase)
                
                pbar.update(1)

    # Final flush for any remaining items
    flush_buffers(buffers, paths, is_discovery_phase)

def run_discovery_for_instrument(instrument_name: str, base_dirs: Dict[str, str]):
    print("[SETUP] Loading all required data...")
    global worker_gold_features_df
    
    # --- MODIFIED: Create all necessary directories at the start ---
    for subdir_key in ['platinum_strategies', 'platinum_blacklists', 'platinum_exhausted', 'platinum_logs']:
        os.makedirs(base_dirs[subdir_key], exist_ok=True)
    
    paths = {
        'gold': os.path.join(base_dirs['gold'], f"{instrument_name}.csv"),
        'combinations': os.path.join(base_dirs['platinum_combo'], f"{instrument_name}.csv"),
        'discovered': os.path.join(base_dirs['platinum_strategies'], f"{instrument_name}.csv"),
        'blacklist': os.path.join(base_dirs['platinum_blacklists'], f"{instrument_name}.csv"),
        'exhausted': os.path.join(base_dirs['platinum_exhausted'], f"{instrument_name}.csv"),
        'processed': os.path.join(base_dirs['platinum_logs'], f"{instrument_name}.processed"),
    }
    
    for path_key, path in paths.items():
        if not os.path.exists(path):
            if '.processed' in path: open(path, 'w').close()
            elif 'blacklist' in path: pd.DataFrame(columns=['key', 'market_rule']).to_csv(path, index=False)
            elif 'discovered' in path: pd.DataFrame(columns=['key', 'market_rule', 'n_candles', 'avg_density']).to_csv(path, index=False)
            elif 'exhausted' in path: pd.DataFrame(columns=['key']).to_csv(path, index=False)
        
    try:
        worker_gold_features_df = pd.read_csv(paths['gold'], parse_dates=['time'])
        all_blueprints_df = pd.read_csv(paths['combinations'])
    except FileNotFoundError as e:
        print(f"[ERROR] A required input file is missing: {e}. Aborting.")
        return

    exhausted_keys_set = set(safe_load_series(paths['exhausted'], 'key'))
    exclusion_rules_by_key = defaultdict(set)
    df_discovered = pd.read_csv(paths['discovered'])
    df_blacklist = pd.read_csv(paths['blacklist'])
    
    for _, row in pd.concat([df_discovered, df_blacklist]).iterrows():
        exclusion_rules_by_key[row['key']].add(row['market_rule'])

    base_dirs['instrument_name'] = instrument_name

    print("\n--- Starting Phase 1: Discovery of New Blueprints ---")
    processed_keys_set = set(safe_load_series(paths['processed'], has_header=False))
    df = all_blueprints_df.copy()
    df = df[df['num_candles'] >= MIN_CANDLE_LIMIT]
    df = df[~df['key'].isin(exhausted_keys_set)]
    df = df[~df['key'].isin(processed_keys_set)]
    keys_for_discovery = df['key'].tolist()
    
    print(f"Found {len(keys_for_discovery)} new blueprints to process.")
    execute_parallel_processing(keys_for_discovery, exclusion_rules_by_key, base_dirs, is_discovery_phase=True)
    
    print("\n--- Starting Phase 2: Iterative Improvement from Blacklist ---")
    keys_with_new_feedback = set(safe_load_series(paths['blacklist'], 'key'))
    df = all_blueprints_df[all_blueprints_df['key'].isin(keys_with_new_feedback)]
    df = df[df['num_candles'] >= MIN_CANDLE_LIMIT]
    df = df[~df['key'].isin(exhausted_keys_set)]
    keys_for_improvement = df['key'].tolist()
    
    print(f"Found {len(keys_for_improvement)} blueprints with new feedback to re-process.")
    execute_parallel_processing(keys_for_improvement, exclusion_rules_by_key, base_dirs, is_discovery_phase=False)

# --- MAIN FUNCTION ---
def main():
    core_dir = os.path.dirname(os.path.abspath(__file__))
    base_dirs = {
        'gold': os.path.abspath(os.path.join(core_dir, '..', 'gold_data', 'features')),
        'platinum_combo': os.path.abspath(os.path.join(core_dir, '..', 'platinum_data', 'combinations')),
        'platinum_strategies': os.path.abspath(os.path.join(core_dir, '..', 'platinum_data', 'discovered_strategies')),
        'platinum_blacklists': os.path.abspath(os.path.join(core_dir, '..', 'platinum_data', 'blacklists')),
        'platinum_exhausted': os.path.abspath(os.path.join(core_dir, '..', 'platinum_data', 'exhausted_keys')),
        'platinum_logs': os.path.abspath(os.path.join(core_dir, '..', 'platinum_data', 'discovery_log')),
        'platinum_final': os.path.abspath(os.path.join(core_dir, '..', 'platinum_data', 'targets')),
    }
    
    # Standard discovery and orchestration logic
    instrument_folders_to_process = []
    target_arg = sys.argv[1] if len(sys.argv) > 1 else None
    
    if target_arg:
        instrument_name = target_arg.replace('.csv', '')
        print(f"[TARGET] Targeted Mode: Processing '{instrument_name}'")
        if os.path.exists(os.path.join(base_dirs['platinum_combo'], f"{instrument_name}.csv")):
            instrument_folders_to_process = [instrument_name]
        else:
            print(f"[ERROR] Combination file not found for: {instrument_name}")
    else:
        print("[SCAN] Interactive Mode: Scanning for new instruments...")
        try:
            all_files = sorted([f.replace('.csv', '') for f in os.listdir(base_dirs['platinum_combo']) if f.endswith('.csv')])
            if not all_files:
                print("[INFO] No combination files found to process.")
            else:
                print("\n--- Select Instrument(s) to Process ---")
                for i, f in enumerate(all_files): print(f"  [{i+1}] {f}")
                user_input = input("Enter number(s) to process: ").strip()
                if user_input:
                    try:
                        indices = [int(i.strip()) - 1 for i in user_input.split(',')]
                        instrument_folders_to_process = [all_files[idx] for idx in sorted(set(indices)) if 0 <= idx < len(all_files)]
                    except ValueError:
                        print("[ERROR] Invalid input.")
        except FileNotFoundError:
            print(f"[ERROR] Source directory not found: {base_dirs['platinum_combo']}")
    
    if not instrument_folders_to_process:
        print("[INFO] No instruments selected for processing.")
        return
        
    print(f"\n[QUEUE] Queued {len(instrument_folders_to_process)} instrument(s): {instrument_folders_to_process}")
    for instrument_name in instrument_folders_to_process:
        try:
            print(f"\n{'='*50}\nProcessing Instrument: {instrument_name}\n{'='*50}")
            run_discovery_for_instrument(instrument_name, base_dirs)
        except Exception:
            print(f"\n[FATAL ORCHESTRATOR ERROR] An unhandled exception occurred in main loop for {instrument_name}.")
            traceback.print_exc()

    print("\n" + "="*50 + "\n[COMPLETE] All strategy discovery tasks are finished.")

if __name__ == "__main__":
    main()
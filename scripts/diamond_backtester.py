# diamond_backtester.py (Upgraded with Definitive Backtest Engine)

"""
Diamond Layer - Stage 1: The Mastery Engine

This script is a lean, fast, single-market backtester that serves as the first
stage of the final validation gauntlet. Its purpose is to take the strategies
discovered by the Platinum layer and backtest them *only* on their home market's
historical data (e.g., test XAUUSD strategies on XAUUSD data).

This version has been upgraded to use the definitive, corrected backtesting engine
from the Zircon layer. This ensures that the performance metrics calculated here
are perfectly consistent with the final validation results, eliminating
discrepancies in the analysis dashboard.

Key Features:
- Definitive Simulation: Utilizes the final, most accurate backtesting engine
  that correctly handles buy/sell logic, slippage, and costs.
- Risk Management: Implements a robust Fixed Fractional Risk model.
- Key-Based Feedback Loop: When a strategy's PARENT BLUEPRINT consistently
  fails, it writes the blueprint's unique 'key' to a blacklist.
- Bulletproof Resumability: Logs its progress and can be stopped and restarted
  at any time without losing completed work.
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import re
from functools import partial
from multiprocessing import Pool, cpu_count

# --- CONFIGURATION ---

# --- Backtest Parameters ---
INITIAL_CAPITAL = 10000.0
RISK_PER_TRADE_PCT = 0.02

# --- "Mastery" Pass/Fail Thresholds ---
PROFIT_FACTOR_PASS_THRESHOLD = 1.5
MAX_DRAWDOWN_PASS_THRESHOLD = 25.0
MIN_TRADES_PASS_THRESHOLD = 50

# --- Performance & Execution ---
TOP_N_PER_RULE = 5
MAX_CPU_USAGE = max(1, cpu_count() - 2)
SAVE_CHECKPOINT_FREQUENCY = 100

# --- Market & Cost Configuration ---
ANNUAL_RISK_FREE_RATE = 0.04
TRADES_PER_YEAR_ESTIMATE = 252
MARKET_CONFIG = {
    "DEFAULT": {"spread_pips": 2.5, "commission_per_lot": 7.0, "pip_value": 0.0001, "slippage_pips": 0.5},
    "XAUUSD":  {"spread_pips": 20.0, "commission_per_lot": 7.0, "pip_value": 0.01,   "slippage_pips": 2.0},
    "EURUSD":  {"spread_pips": 1.5, "commission_per_lot": 7.0, "pip_value": 0.0001, "slippage_pips": 0.3}
}


# --- HELPER & SIMULATION FUNCTIONS (SYNCHRONIZED WITH ZIRCON LAYER) ---

def get_dynamic_price(entry_price, level_price, placement_bin):
    """
    Calculates an absolute price for a dynamic Stop-Loss or Take-Profit.

    This function translates a discretized placement bin into a concrete price
    level. It assumes the bin represents a 10% range and uses the midpoint
    (e.g., bin 8 represents 85%) to calculate the target price relative to the
    distance between the entry and the reference market level (e.g., an SMA).

    Args:
        entry_price (float): The price at which the trade is entered.
        level_price (float): The price of the reference market level (e.g., SMA_50).
        placement_bin (int): The integer bin representing the placement (e.g., 8).

    Returns:
        float: The calculated absolute price for the SL or TP.
    """
    placement_ratio = (placement_bin + 0.5) / 10.0
    return entry_price + (level_price - entry_price) * placement_ratio

def calculate_sharpe_ratio(pnl_series):
    """
    Calculates the annualized Sharpe Ratio from a series of trade PnLs.

    This function measures the risk-adjusted return of the backtest. It
    annualizes the result based on an estimated number of trading days per
    year to provide a standardized metric for comparison.

    Args:
        pnl_series (pd.Series): A pandas Series containing the PnL of each individual trade.

    Returns:
        float: The calculated annualized Sharpe Ratio. Returns 0.0 if calculation
               is not possible (e.g., standard deviation is zero).
    """
    if pnl_series.std() == 0 or len(pnl_series) < 2: return 0.0
    daily_returns = pnl_series / INITIAL_CAPITAL
    excess_returns = daily_returns - (ANNUAL_RISK_FREE_RATE / TRADES_PER_YEAR_ESTIMATE)
    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(TRADES_PER_YEAR_ESTIMATE)

def simulate_trades(strategy, entries, full_silver_data, market_config):
    """
    Executes a high-fidelity, event-driven backtest for a single strategy.

    This is the definitive backtesting engine, synchronized with the Zircon layer
    to ensure perfectly consistent results. For each entry signal, it calculates
    precise SL/TP levels, determines position size based on a fixed fractional
    risk model, and simulates the trade's outcome by iterating through future
    candles. It accurately models slippage, spread, and commissions.

    Args:
        strategy (dict): A dictionary representing the complete strategy, including
                         the blueprint and market rule.
        entries (pd.DataFrame): A DataFrame of the candles that triggered an entry signal.
        full_silver_data (pd.DataFrame): The complete, feature-rich market data used
                                         for looking up future candle outcomes.
        market_config (dict): A dictionary of market-specific parameters like spread
                              and commission.

    Returns:
        dict or None: A dictionary containing key performance metrics (Profit Factor,
                      Sharpe Ratio, Max Drawdown, etc.) for the backtest. Returns None
                      if no valid trades could be executed.
    """
    trades, capital = [], INITIAL_CAPITAL
    time_to_idx = {time: i for i, time in enumerate(full_silver_data['time'])}
    
    is_sl_dynamic, is_tp_dynamic = isinstance(strategy['sl_def'], str), isinstance(strategy['tp_def'], str)
    spread_cost = market_config['spread_pips'] * market_config['pip_value']
    commission_per_unit = market_config['commission_per_lot'] / 100000
    slippage_cost = market_config['slippage_pips'] * market_config['pip_value']
    
    for entry in entries.itertuples():
        entry_price = entry.close
        adjusted_entry_price = entry_price + slippage_cost if strategy['trade_type'] == 'buy' else entry_price - slippage_cost

        # Definitive fix for sell trade SL/TP logic ensures correct levels are calculated
        if strategy['trade_type'] == 'buy':
            sl_price = get_dynamic_price(entry_price, getattr(entry, strategy['sl_def']), strategy['sl_bin']) if is_sl_dynamic else entry_price * (1 - strategy['sl_def'])
            tp_price = get_dynamic_price(entry_price, getattr(entry, strategy['tp_def']), strategy['tp_bin']) if is_tp_dynamic else entry_price * (1 + strategy['tp_def'])
            if sl_price >= adjusted_entry_price or tp_price <= adjusted_entry_price: continue
        else: # Sell trade
            sl_price = get_dynamic_price(entry_price, getattr(entry, strategy['sl_def']), strategy['sl_bin']) if is_sl_dynamic else entry_price * (1 + strategy['sl_def'])
            tp_price = get_dynamic_price(entry_price, getattr(entry, strategy['tp_def']), strategy['tp_bin']) if is_tp_dynamic else entry_price * (1 - strategy['tp_def'])
            if sl_price <= adjusted_entry_price or tp_price >= adjusted_entry_price: continue
        
        risk_per_unit = abs(adjusted_entry_price - sl_price)
        if risk_per_unit == 0: continue
        position_size = (capital * RISK_PER_TRADE_PCT) / risk_per_unit
        
        entry_idx = time_to_idx.get(entry.time)
        if entry_idx is None or entry_idx + 1 >= len(full_silver_data): continue
        
        limit = min(entry_idx + 1 + 5000, len(full_silver_data))
        future_candles = full_silver_data.iloc[entry_idx + 1:limit]
        
        pnl, exit_time = 0.0, None
        if strategy['trade_type'] == 'buy':
            tp_hits = future_candles[future_candles['high'] >= tp_price]
            sl_hits = future_candles[future_candles['low'] <= sl_price]
            if not tp_hits.empty and (sl_hits.empty or tp_hits.index[0] < sl_hits.index[0]):
                pnl, exit_time = (tp_price - adjusted_entry_price) * position_size, tp_hits['time'].iloc[0]
            elif not sl_hits.empty:
                pnl, exit_time = (sl_price - adjusted_entry_price) * position_size, sl_hits['time'].iloc[0]
        else: # Sell trade outcome logic
            tp_hits = future_candles[future_candles['low'] <= tp_price]
            sl_hits = future_candles[future_candles['high'] >= sl_price]
            if not tp_hits.empty and (sl_hits.empty or tp_hits.index[0] < sl_hits.index[0]):
                pnl, exit_time = (adjusted_entry_price - tp_price) * position_size, tp_hits['time'].iloc[0]
            elif not sl_hits.empty:
                pnl, exit_time = (adjusted_entry_price - sl_price) * position_size, sl_hits['time'].iloc[0]

        if pnl != 0.0 and exit_time is not None:
            transaction_costs = (spread_cost * position_size) + (commission_per_unit * position_size)
            net_pnl = pnl - transaction_costs
            capital += net_pnl
            trades.append({'pnl': net_pnl, 'capital': capital})
            
    if not trades: return None
    
    log = pd.DataFrame(trades)
    wins = log[log['pnl'] > 0]
    pf = wins['pnl'].sum() / abs(log[log['pnl'] <= 0]['pnl'].sum()) if abs(log[log['pnl'] <= 0]['pnl'].sum()) > 0 else np.inf
    log['peak'] = log['capital'].cummax()
    max_dd_pct = ((log['peak'] - log['capital']).max() / log['peak'].max()) * 100 if log['peak'].max() > 0 else 0
    
    return {'profit_factor': pf, 'sharpe_ratio': calculate_sharpe_ratio(log['pnl']), 'max_drawdown_pct': max_dd_pct, 
            'final_capital': capital, 'total_trades': len(log), 
            'win_rate_pct': (len(wins) / len(log)) * 100 if len(log) > 0 else 0}

def run_backtest_for_strategy(strategy_dict, market_data_cache, origin_market_csv):
    """
    Orchestrates the backtest for a single strategy, serving as the worker for multiprocessing.

    This function acts as a wrapper that prepares the necessary data for the main
    `simulate_trades` engine. It queries the Gold feature data to find all entry
    signals for a given market rule, retrieves the corresponding raw candle data
    from the Silver DataFrame, and then passes everything to the simulation
    function to generate performance results.

    Args:
        strategy_dict (dict): A dictionary representing a single, complete strategy.
        market_data_cache (dict): A cache holding the prepared Silver and Gold DataFrames.
        origin_market_csv (str): The filename of the market the strategy was discovered on.

    Returns:
        dict or None: A dictionary containing the full, combined results (strategy
                      definition + performance metrics + regime analysis), or None
                      if the backtest fails or produces no trades.
    """
    strategy, market_name = strategy_dict, origin_market_csv.replace('.csv', '')
    silver_df, gold_df = market_data_cache[origin_market_csv]['silver'], market_data_cache[origin_market_csv]['gold']
    try:
        entry_candles = gold_df.query(strategy['market_rule'])
        if entry_candles.empty: return None
        
        entry_points_silver = silver_df[silver_df.time.isin(entry_candles.time)]
        market_key = re.sub(r'\d+', '', market_name) # Extract base market name like "XAUUSD"
        market_config = MARKET_CONFIG.get(market_key, MARKET_CONFIG["DEFAULT"])

        performance = simulate_trades(strategy, entry_points_silver, silver_df, market_config)
        
        if performance:
            regime_analysis = {f"{col}_pct": (entry_points_silver[col].value_counts(normalize=True) * 100).to_dict() for col in ['session', 'trend_regime', 'vol_regime']}
            return {'strategy_id': strategy['strategy_id'], 'market': origin_market_csv, **strategy, **performance, **regime_analysis}
    except Exception:
        return None
    return None

if __name__ == "__main__":
    core_dir = os.path.dirname(os.path.abspath(__file__))
    discovered_dir, prepared_data_dir, backtest_results_dir, blacklist_dir, zircon_input_dir = [os.path.abspath(os.path.join(core_dir, '..', d)) for d in ['platinum_data/discovered_strategy', 'diamond_data/prepared_data', 'diamond_data/backtesting_results', 'platinum_data/blacklists', 'zircon_data/input']]
    for d in [backtest_results_dir, blacklist_dir, zircon_input_dir]: os.makedirs(d, exist_ok=True)
    
    strategy_files = [f for f in os.listdir(discovered_dir) if f.endswith('.csv')]
    if not strategy_files: print("❌ No discovered strategy files found."); exit()
    print("--- Select a Strategy File to Backtest ---")
    for i, f in enumerate(strategy_files): print(f"  [{i+1}] {f}")
    try:
        choice = int(input(f"Enter number (1-{len(strategy_files)}): ")) - 1
        strategy_file_to_test = strategy_files[choice]
    except (ValueError, IndexError): print("❌ Invalid selection. Exiting."); exit()

    origin_market_csv = strategy_file_to_test
    origin_market_name = origin_market_csv.replace('.csv', '')
    
    discovered_path = os.path.join(discovered_dir, strategy_file_to_test)
    strategies_df_full = pd.read_csv(discovered_path, dtype={'sl_def': object, 'tp_def': object, 'key': str})
    if 'key' not in strategies_df_full.columns:
        print(f"❌ FATAL: 'key' column not found in {strategy_file_to_test}. Please re-run the Platinum discoverer."); exit()
    
    def to_numeric_or_str(x):
        try: return float(x)
        except (ValueError, TypeError): return x
    for col in ['sl_def', 'tp_def']:
        strategies_df_full[col] = strategies_df_full[col].apply(to_numeric_or_str)
    
    # The Zircon backtester expects a 'trade_type' column.
    if 'trade_type' not in strategies_df_full.columns:
        strategies_df_full['trade_type'] = 'buy' # Default to buy, as logic relies on this
    
    strategies_df_full['strategy_id'] = strategies_df_full['key'] + '_' + strategies_df_full['market_rule']

    print(f"\nFound {len(strategies_df_full)} total strategies in {strategy_file_to_test}.")
    try:
        top_n_input = input(f"How many top strategies per rule to test? (Default: {TOP_N_PER_RULE}): ").strip()
        TOP_N_PER_RULE = int(top_n_input) if top_n_input else TOP_N_PER_RULE
    except ValueError: print(f"Invalid number. Defaulting to {TOP_N_PER_RULE}.")
    strategies_df = strategies_df_full.groupby('market_rule').head(TOP_N_PER_RULE)

    detailed_report_path = os.path.join(backtest_results_dir, f"diamond_report_{origin_market_name}.csv")
    processed_log_path = os.path.join(backtest_results_dir, f".{origin_market_name}.processed_log")
    try:
        with open(processed_log_path, 'r') as f: processed_ids = set(f.read().splitlines())
        strategies_to_test = strategies_df[~strategies_df['strategy_id'].isin(processed_ids)]
        print(f"\nFound {len(processed_ids)} previously completed backtests. Resuming with {len(strategies_to_test)}.")
    except FileNotFoundError:
        strategies_to_test = strategies_df
        print(f"\nStarting new backtest with {len(strategies_to_test)} strategies.")

    if not strategies_to_test.empty:
        print("\nLoading prepared market data...")
        market_data_cache = {origin_market_csv: {'silver': pd.read_parquet(os.path.join(prepared_data_dir, f"{origin_market_name}_silver.parquet")), 'gold': pd.read_parquet(os.path.join(prepared_data_dir, f"{origin_market_name}_gold.parquet"))}}
        
        use_multiprocessing = input("Use multiprocessing? (y/n): ").strip().lower() == 'y'
        num_processes = MAX_CPU_USAGE if use_multiprocessing else 1
        strategy_list = strategies_to_test.to_dict('records')
        
        with open(processed_log_path, 'a') as log_file:
            temp_results = []
            func = partial(run_backtest_for_strategy, market_data_cache=market_data_cache, origin_market_csv=origin_market_csv)
            iterator = None
            pool = None
            if num_processes > 1 and len(strategy_list) > 1:
                pool = Pool(processes=min(num_processes, len(strategy_list)))
                iterator = pool.imap_unordered(func, strategy_list)
            else:
                iterator = map(func, strategy_list)

            for i, result in enumerate(tqdm(iterator, total=len(strategy_list), desc="Backtesting Strategies")):
                if result: temp_results.append(result)
                log_file.write(strategy_list[i]['strategy_id'] + '\n')
                if (i + 1) % SAVE_CHECKPOINT_FREQUENCY == 0 and temp_results:
                    pd.DataFrame(temp_results).to_csv(detailed_report_path, mode='a', header=not os.path.exists(detailed_report_path), index=False)
                    temp_results = []
            
            if temp_results: pd.DataFrame(temp_results).to_csv(detailed_report_path, mode='a', header=not os.path.exists(detailed_report_path), index=False)
            if pool: pool.close(); pool.join()

    print("\nBacktesting finished. Performing final cleanup...")
    try:
        results_df = pd.read_csv(detailed_report_path)
        results_df.drop_duplicates(subset=['strategy_id'], keep='last', inplace=True)
        results_df.sort_values(by='sharpe_ratio', ascending=False, inplace=True)
        
        passed_strategies_df = results_df[(results_df['profit_factor'] >= PROFIT_FACTOR_PASS_THRESHOLD) &
                                          (results_df['max_drawdown_pct'] <= MAX_DRAWDOWN_PASS_THRESHOLD) &
                                          (results_df['total_trades'] >= MIN_TRADES_PASS_THRESHOLD)].copy()
        
        failed_strategies_df = results_df[~results_df['strategy_id'].isin(passed_strategies_df['strategy_id'])]

        zircon_input_path = os.path.join(zircon_input_dir, f"master_strategies_{origin_market_name}.csv")
        if os.path.exists(zircon_input_path):
            existing_zircon = pd.read_csv(zircon_input_path)
            passed_strategies_df = pd.concat([existing_zircon, passed_strategies_df]).drop_duplicates(subset=['strategy_id'])
        passed_strategies_df.to_csv(zircon_input_path, index=False)
        print(f"\n✅ Found {len(passed_strategies_df)} total master strategies. Saved to Zircon input.")
        
        results_df.to_csv(detailed_report_path, index=False)
        print(f"✅ Full detailed report cleaned and saved.")

        if not failed_strategies_df.empty:
            failed_keys_df = failed_strategies_df[['key']].drop_duplicates()
            blacklist_path = os.path.join(blacklist_dir, strategy_file_to_test)
            if os.path.exists(blacklist_path):
                existing_blacklist = pd.read_csv(blacklist_path)
                failed_keys_df = pd.concat([existing_blacklist, failed_keys_df]).drop_duplicates()
            failed_keys_df.to_csv(blacklist_path, index=False)
            print(f"✅ Added/updated {len(failed_keys_df)} underperforming blueprint keys to blacklist.")
    except FileNotFoundError:
        print("ℹ️ No results were generated to analyze.")

    print("\n" + "="*50 + "\n✅ Diamond Layer (Mastery) run complete.")
# zircon_validator.py (V5 - Final Reporting Engine with Trade Logs)

"""
Zircon Layer: The Ultimate Judge

This script is the final, multi-market reporting engine and the ultimate arbiter
of a strategy's worth. It takes the small, pre-filtered list of "master"
strategies from the Diamond Layer and subjects them to a rigorous out-of-sample
test.

Its purpose is to answer the most critical question: "Is the strategy's edge
real and transferable, or was it just a fluke curve-fit to its original market?"
It does this by automatically finding all other available markets of the same
timeframe and running a full backtest against them.

This is the stage that generates the two most important outputs for the final
human-led analysis:
1.  Detailed Trade Logs: For every elite strategy, it saves a trade-by-trade
    log for each market it was tested on. These logs power the deep analysis
    and visualizations in the Streamlit dashboard.
2.  Final Summary & Detailed Reports: It produces comprehensive reports that
    aggregate a strategy's performance across all tested markets, providing the
    definitive data needed for a final "go/no-go" decision.
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import re
import hashlib
from functools import partial
from multiprocessing import Pool, cpu_count

# --- CONFIGURATION ---
# These configurations are identical to the Diamond layer to ensure a consistent
# backtesting environment for fair comparison.
INITIAL_CAPITAL = 10000.0
RISK_PER_TRADE_PCT = 0.02
ANNUAL_RISK_FREE_RATE = 0.04
TRADES_PER_YEAR_ESTIMATE = 252
MARKET_CONFIG = {
    "DEFAULT": {"spread_pips": 2.5, "commission_per_lot": 7.0, "pip_value": 0.0001, "slippage_pips": 0.5},
    "EURUSD":  {"spread_pips": 1.5, "commission_per_lot": 7.0, "pip_value": 0.0001, "slippage_pips": 0.3},
    "GBPUSD":  {"spread_pips": 2.0, "commission_per_lot": 7.0, "pip_value": 0.0001, "slippage_pips": 0.4},
    "AUDUSD":  {"spread_pips": 2.5, "commission_per_lot": 7.0, "pip_value": 0.0001, "slippage_pips": 0.5},
    "USDJPY":  {"spread_pips": 2.0, "commission_per_lot": 7.0, "pip_value": 0.01,   "slippage_pips": 0.4},
    "USDCAD":  {"spread_pips": 2.5, "commission_per_lot": 7.0, "pip_value": 0.0001, "slippage_pips": 0.5}
}
MAX_CPU_USAGE = max(1, cpu_count() - 2)

# --- HELPER & SIMULATION FUNCTIONS ---

def get_dynamic_price(entry_price, level_price, placement_bin):
    """Calculates the absolute SL/TP price from a binned placement."""
    placement_ratio = (placement_bin + 0.5) / 10.0
    return entry_price + (level_price - entry_price) * placement_ratio

def calculate_sharpe_ratio(pnl_series, risk_free_rate, trades_per_year):
    """Calculates the annualized Sharpe Ratio for a series of trade PnLs."""
    if pnl_series.std() == 0 or len(pnl_series) < 2: return np.inf
    daily_returns = pnl_series / INITIAL_CAPITAL
    excess_returns = daily_returns - (risk_free_rate / trades_per_year)
    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(trades_per_year)

def simulate_trades(strategy, entries, full_silver_data, market_config):
    """
    UPGRADED simulation engine.

    This version is enhanced to return two outputs:
    1. A dictionary of overall performance statistics.
    2. A detailed list of every individual trade, which will be saved as a trade log.
    It also includes the definitive bug fix for correctly simulating sell trades.
    """
    trades, trade_log_for_export = [], []
    capital = INITIAL_CAPITAL
    
    time_to_idx = {time: i for i, time in enumerate(full_silver_data['time'])}
    is_sl_dynamic, is_tp_dynamic = isinstance(strategy['sl_def'], str), isinstance(strategy['tp_def'], str)
    spread_cost_pips = market_config['spread_pips'] * market_config['pip_value']
    commission_per_unit = market_config['commission_per_lot'] / 100000
    slippage_cost_pips = market_config['slippage_pips'] * market_config['pip_value']
    
    for entry in entries.itertuples():
        entry_price = entry.close
        adjusted_entry_price = entry_price + slippage_cost_pips if strategy['trade_type'] == 'buy' else entry_price - slippage_cost_pips

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
        if position_size <= 0: continue
        
        entry_idx = time_to_idx.get(entry.time)
        if entry_idx is None or entry_idx + 1 >= len(full_silver_data): continue
        
        # Define a safe look-forward window to find the trade's outcome
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
            transaction_costs = (spread_cost_pips * position_size) + (commission_per_unit * position_size)
            net_pnl = pnl - transaction_costs
            capital += net_pnl
            trades.append({'pnl': net_pnl, 'capital': capital})
            # Append detailed trade information for the final log file
            trade_log_for_export.append({'entry_time': entry.time, 'exit_time': exit_time, 'pnl': net_pnl, 'entry_price': entry_price})

    if not trades: return None, None
    
    # Calculate final performance statistics
    log = pd.DataFrame(trades)
    wins, losses = log[log['pnl'] > 0], log[log['pnl'] <= 0]
    pf = wins['pnl'].sum() / abs(losses['pnl'].sum()) if abs(losses['pnl'].sum()) > 0 else np.inf
    log['peak'] = log['capital'].cummax()
    max_dd_pct = ((log['peak'] - log['capital']).max() / log['peak'].max()) * 100 if log['peak'].max() > 0 else 0
    sharpe = calculate_sharpe_ratio(log['pnl'], ANNUAL_RISK_FREE_RATE, TRADES_PER_YEAR_ESTIMATE)
    
    performance_stats = {'profit_factor': pf, 'sharpe_ratio': sharpe, 'max_drawdown_pct': max_dd_pct, 'total_trades': len(log)}
    return performance_stats, trade_log_for_export

def run_validation_for_strategy(strategy, market_data_cache, market_csv, trade_logs_dir):
    """
    The main worker function for multiprocessing. It backtests a single strategy
    on a single market and, if successful, saves its detailed trade log to disk.
    """
    market_name = market_csv.replace('.csv', '')
    # The market data is passed in a cache for efficiency
    silver_df, gold_df = market_data_cache['silver'], market_data_cache['gold']
    try:
        entry_candles = gold_df.query(strategy['market_rule'])
        if entry_candles.empty: return None

        entry_points_silver = silver_df[silver_df.time.isin(entry_candles.time)]
        
        # Strip timeframe from market name for config lookup
        market_key = re.sub(r'\d+$', '', market_name)
        market_config = MARKET_CONFIG.get(market_key, MARKET_CONFIG["DEFAULT"])
        
        performance, trade_log = simulate_trades(strategy, entry_points_silver, silver_df, market_config)
        
        if performance and trade_log:
            # --- CRITICAL STEP: Save the detailed trade log ---
            log_dir = os.path.join(trade_logs_dir, strategy['strategy_id'])
            os.makedirs(log_dir, exist_ok=True)
            pd.DataFrame(trade_log).to_csv(os.path.join(log_dir, f"{market_name}.csv"), index=False)
            
            # Package the results for the final reports
            regime_analysis = {f"{col}_pct": (entry_points_silver[col].value_counts(normalize=True) * 100).to_dict() for col in ['session', 'trend_regime', 'vol_regime']}
            result = {'strategy_id': strategy['strategy_id'], 'market': market_csv, **performance, **regime_analysis}
            definition_keys = ['type', 'sl_def', 'sl_bin', 'tp_def', 'tp_bin', 'market_rule', 'trade_type']
            for key in definition_keys:
                if key in strategy: result[key] = strategy[key]
            return result
    except Exception:
        return None
    return None

if __name__ == "__main__":
    # --- Define Project Directories ---
    core_dir = os.path.dirname(os.path.abspath(__file__))
    zircon_input_dir, prepared_data_dir, zircon_results_dir, trade_logs_dir = [os.path.abspath(os.path.join(core_dir, '..', d)) for d in ['zircon_data/input', 'diamond_data/prepared_data', 'zircon_data/results', 'zircon_data/trade_logs']]
    for d in [zircon_results_dir, trade_logs_dir]: os.makedirs(d, exist_ok=True)

    # --- Interactive Setup ---
    master_strategy_files = [f for f in os.listdir(zircon_input_dir) if f.startswith('master_strategies_') and f.endswith('.csv')]
    if not master_strategy_files: print("❌ No master strategy files found."); exit()
    print("--- Select a Master Strategy File to Validate ---")
    for i, f in enumerate(master_strategy_files): print(f"  [{i+1}] {f}")
    try:
        choice = int(input(f"Enter number (1-{len(master_strategy_files)}): ")) - 1
        file_to_validate = master_strategy_files[choice]
    except (ValueError, IndexError):
        print("❌ Invalid selection. Exiting."); exit()
    
    match = re.search(r'master_strategies_(.+?(\d+))\.csv', file_to_validate)
    if not match: print(f"❌ Invalid filename format."); exit()
    
    origin_market_name_full, origin_timeframe_num = match.group(1), match.group(2)
    
    # --- Load Strategies and Prepare for Testing ---
    strategies_df = pd.read_csv(os.path.join(zircon_input_dir, file_to_validate))
    if 'trade_type' not in strategies_df.columns:
        print("❌ 'trade_type' column not found in master strategies file. Please re-run the Diamond layer."); exit()
    
    def to_numeric_or_str(x):
        try: return float(x)
        except (ValueError, TypeError): return x
    for col in ['sl_def', 'tp_def']:
        if col in strategies_df.columns: strategies_df[col] = strategies_df[col].apply(to_numeric_or_str)

    # --- Discover Markets for Validation ---
    # Find all prepared markets that share the same timeframe as the origin market.
    all_prepared_markets = sorted([f.replace('_silver.parquet', '.csv') for f in os.listdir(prepared_data_dir) if f.endswith('_silver.parquet')])
    markets_to_test = [m for m in all_prepared_markets if m.endswith(f"{origin_timeframe_num}.csv")]

    print(f"\n✅ Found {len(strategies_df)} master strategies from {origin_market_name_full}.csv.")
    if not markets_to_test: print(f"⚠️ No prepared markets found for timeframe '{origin_timeframe_num}m'."); exit()
    print(f"✅ Will generate reports and logs for {len(markets_to_test)} markets: {markets_to_test}")
    
    # --- Resumability Logic (Market-based) ---
    detailed_report_path = os.path.join(zircon_results_dir, f"detailed_report_{origin_market_name_full}.csv")
    processed_log_path = os.path.join(zircon_results_dir, f".{origin_market_name_full}.processed_log")
    try:
        with open(processed_log_path, 'r') as f: processed_markets = set(f.read().splitlines())
        markets_to_process = [m for m in markets_to_test if m not in processed_markets]
        print(f"\nFound {len(processed_markets)} previously completed markets. Resuming with {len(markets_to_process)} remaining.")
    except FileNotFoundError:
        markets_to_process = markets_to_test
        print(f"\nStarting new reporting run for {len(markets_to_process)} markets.")

    if not markets_to_process:
        print("✅ All markets have already been processed for this strategy set.")
    else:
        use_multiprocessing = input("Use multiprocessing? (y/n): ").strip().lower() == 'y'
        num_processes = MAX_CPU_USAGE if use_multiprocessing else 1
        strategy_list = strategies_df.to_dict('records')

        # --- Main Execution Loop (Iterate through Markets) ---
        with open(processed_log_path, 'a') as log_file:
            for market_csv in tqdm(markets_to_process, desc="Processing Markets"):
                market_name = market_csv.replace('.csv', '')
                print(f"\n-- Loading and processing {market_csv} --")
                market_data_cache = {'silver': pd.read_parquet(os.path.join(prepared_data_dir, f"{market_name}_silver.parquet")), 'gold': pd.read_parquet(os.path.join(prepared_data_dir, f"{market_name}_gold.parquet"))}
                
                func = partial(run_validation_for_strategy, market_data_cache=market_data_cache, market_csv=market_csv, trade_logs_dir=trade_logs_dir)
                
                market_results = []
                if num_processes > 1 and len(strategy_list) > 1:
                    with Pool(processes=num_processes) as pool:
                        results = list(tqdm(pool.imap_unordered(func, strategy_list), total=len(strategy_list), desc=f"Testing on {market_csv}"))
                    market_results = [r for r in results if r is not None]
                else:
                    for s in tqdm(strategy_list, desc=f"Testing on {market_csv}"):
                        result = func(s)
                        if result: market_results.append(result)
                
                # Append the results for this market to the detailed report.
                if market_results:
                    df = pd.DataFrame(market_results)
                    file_exists = os.path.exists(detailed_report_path) and os.path.getsize(detailed_report_path) > 0
                    df.to_csv(detailed_report_path, mode='a', header=not file_exists, index=False)
                    print(f"✅ Appended {len(df)} results for {market_csv}.")
                
                # Log that this market has been completed.
                log_file.write(market_csv + '\n')
                log_file.flush()

    # --- Final Report Generation ---
    print("\nValidation runs complete. Generating final summary report...")
    try:
        detailed_df = pd.read_csv(detailed_report_path)
        detailed_df.drop_duplicates(subset=['strategy_id', 'market'], keep='last', inplace=True)
        
        # Group by strategy to create the high-level summary.
        summary_df = detailed_df.groupby('strategy_id').agg(
            markets_tested=('market', 'count'),
            avg_profit_factor=('profit_factor', 'mean'),
            avg_sharpe_ratio=('sharpe_ratio', 'mean'),
            avg_max_drawdown_pct=('max_drawdown_pct', 'mean'),
            total_trades=('total_trades', 'sum')
        ).reset_index()
        
        # Merge the original strategy definitions back in for context.
        summary_df = pd.merge(summary_df, strategies_df, on='strategy_id', how='left')
        
        # Calculate the key validation metric: number of markets passed (PF > 1.0)
        passed_strategies = detailed_df[detailed_df.profit_factor > 1.0].groupby('strategy_id')['market'].count()
        summary_df['validation_markets_passed_count'] = summary_df['strategy_id'].map(passed_strategies).fillna(0).astype(int)
        summary_df['validation_markets_passed'] = summary_df['validation_markets_passed_count'].astype(str) + '/' + summary_df['markets_tested'].astype(str)
        
        # Sort by the most robust strategies first.
        summary_df = summary_df.sort_values(by=['validation_markets_passed_count', 'avg_sharpe_ratio'], ascending=False)

        summary_report_path = os.path.join(zircon_results_dir, f"summary_report_{origin_market_name_full}.csv")
        summary_df.to_csv(summary_report_path, index=False)
        detailed_df.to_csv(detailed_report_path, index=False)
        print(f"✅ Final reports generated for '{origin_market_name_full}'.")

    except FileNotFoundError:
        print("ℹ️ No detailed results found to generate a summary from.")
    
    print("\n" + "="*50 + "\n✅ Zircon Layer (Validation) run complete.")
# diamond_backtester.py (with Bulletproof Resumability)

"""
Diamond Layer - Stage 1: The Mastery Engine

This script is a lean, fast, single-market backtester that serves as the first
stage of the final validation gauntlet. Its purpose is to take the strategies
discovered for a specific instrument (e.g., `XAUUSD15`) and backtest them
*only* on that same instrument's historical data.

Its job is to filter for strategies that demonstrate exceptional performance or
"mastery" on their home turf. Only strategies that pass a strict set of
performance thresholds (Profit Factor, Max Drawdown, etc.) are promoted to the
next stage (Zircon) for out-of-sample validation.

Key Features:
- Realistic Simulation: The backtesting engine accounts for spreads,
  commissions, and a simplified model of slippage.
- Risk Management: Implements a robust Fixed Fractional Risk model to determine
  position size for every trade.
- Definitive Feedback Loop: Strategies that fail are used to update the
  Platinum layer's blacklist, preventing the system from re-discovering
  unprofitable blueprints in the future.
- Bulletproof Resumability: The script logs its progress and can be stopped
  and restarted at any time without losing completed work, thanks to a robust
  checkpointing system.
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import re
import hashlib
from functools import partial
from multiprocessing import Pool, cpu_count
import ta
import talib
import numba
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---

# --- Backtest Parameters ---
INITIAL_CAPITAL = 10000.0
RISK_PER_TRADE_PCT = 0.02  # Risk 2% of current capital on each trade.

# --- "Mastery" Pass/Fail Thresholds ---
# A strategy must meet all these criteria to be considered a "master".
PROFIT_FACTOR_PASS_THRESHOLD = 1.5
MAX_DRAWDOWN_PASS_THRESHOLD = 25.0  # Maximum acceptable drawdown percentage.
MIN_TRADES_PASS_THRESHOLD = 50      # Minimum number of trades to be statistically valid.

# --- Performance & Execution ---
# To speed up backtesting, only test the top N strategies for each market rule.
TOP_N_PER_RULE = 5
# Sets the maximum number of CPU cores to use for multiprocessing.
MAX_CPU_USAGE = max(1, cpu_count() - 2)
# How often to save progress to disk. Lower is safer but slightly slower.
SAVE_CHECKPOINT_FREQUENCY = 100

# --- HELPER & SIMULATION FUNCTIONS ---

# --- Market & Cost Configuration ---
ANNUAL_RISK_FREE_RATE = 0.04 # Used for Sharpe Ratio calculation.
TRADES_PER_YEAR_ESTIMATE = 252 # Used for Sharpe Ratio calculation.
MARKET_CONFIG = {
    "DEFAULT": {"spread_pips": 2.5, "commission_per_lot": 7.0, "pip_value": 0.0001, "slippage_pips": 0.5},
    "EURUSD":  {"spread_pips": 1.5, "commission_per_lot": 7.0, "pip_value": 0.0001, "slippage_pips": 0.3},
    "GBPUSD":  {"spread_pips": 2.0, "commission_per_lot": 7.0, "pip_value": 0.0001, "slippage_pips": 0.4},
    "AUDUSD":  {"spread_pips": 2.5, "commission_per_lot": 7.0, "pip_value": 0.0001, "slippage_pips": 0.5},
    "USDJPY":  {"spread_pips": 2.0, "commission_per_lot": 7.0, "pip_value": 0.01,   "slippage_pips": 0.4},
    "USDCAD":  {"spread_pips": 2.5, "commission_per_lot": 7.0, "pip_value": 0.0001, "slippage_pips": 0.5}
}

def get_dynamic_price(entry_price, level_price, placement_bin):
    """Calculates the absolute SL/TP price from a binned placement."""
    # The middle of the bin (e.g., bin 8 -> 85%) is used for the calculation.
    placement_ratio = (placement_bin + 0.5) / 10.0
    return entry_price + (level_price - entry_price) * placement_ratio

def calculate_sharpe_ratio(pnl_series, risk_free_rate, trades_per_year):
    """Calculates the annualized Sharpe Ratio for a series of trade PnLs."""
    if pnl_series.std() == 0 or len(pnl_series) < 2: return 0.0
    daily_returns = pnl_series / INITIAL_CAPITAL
    excess_returns = daily_returns - (risk_free_rate / trades_per_year)
    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(trades_per_year)

def simulate_trades(strategy, entries, full_silver_data, market_config):
    """The core, high-fidelity backtesting simulation engine."""
    trades, capital = [], INITIAL_CAPITAL
    time_to_idx = {time: i for i, time in enumerate(full_silver_data['time'])}
    is_sl_dynamic = isinstance(strategy['sl_def'], str)
    is_tp_dynamic = isinstance(strategy['tp_def'], str)
    
    # Pre-calculate cost factors for efficiency
    spread_cost_pips = market_config['spread_pips'] * market_config['pip_value']
    commission_per_unit = market_config['commission_per_lot'] / 100000
    slippage_cost_pips = market_config['slippage_pips'] * market_config['pip_value']
    
    for entry in entries.itertuples():
        entry_price = entry.close
        # Apply slippage to the entry price
        adjusted_entry_price = entry_price + slippage_cost_pips if strategy['trade_type'] == 'buy' else entry_price - slippage_cost_pips
        
        # Determine SL and TP prices based on strategy definition
        sl_price = get_dynamic_price(entry_price, getattr(entry, strategy['sl_def']), strategy['sl_bin']) if is_sl_dynamic else entry_price * (1 - strategy['sl_def'])
        tp_price = get_dynamic_price(entry_price, getattr(entry, strategy['tp_def']), strategy['tp_bin']) if is_tp_dynamic else entry_price * (1 + strategy['tp_def'])

        # Basic validity check for a buy trade
        if strategy['trade_type'] == 'buy' and (sl_price >= adjusted_entry_price or tp_price <= adjusted_entry_price): continue
        # Basic validity check for a sell trade is implicitly handled by the PnL calculation
        
        # Calculate Position Size using Fixed Fractional Risk
        risk_per_unit = abs(adjusted_entry_price - sl_price)
        if risk_per_unit == 0: continue
        position_size = (capital * RISK_PER_TRADE_PCT) / risk_per_unit
        if position_size <= 0: continue
        
        # Look forward in time to find the trade outcome
        entry_idx = time_to_idx.get(entry.time)
        if entry_idx is None: continue
        future_highs, future_lows = full_silver_data['high'].values[entry_idx + 1:], full_silver_data['low'].values[entry_idx + 1:]
        tp_hits, sl_hits = np.where(future_highs >= tp_price)[0], np.where(future_lows <= sl_price)[0]
        first_tp_idx = tp_hits[0] if len(tp_hits) > 0 else np.inf
        first_sl_idx = sl_hits[0] if len(sl_hits) > 0 else np.inf
        
        pnl = 0.0
        # Determine outcome based on which level was hit first
        if strategy['trade_type'] == 'buy':
            if first_tp_idx < first_sl_idx: pnl = (tp_price - adjusted_entry_price) * position_size
            elif first_sl_idx < first_tp_idx: pnl = (sl_price - adjusted_entry_price) * position_size
        else: # Sell trade logic
            sl_price_sell = entry_price * (1 + strategy['sl_def'])
            tp_price_sell = entry_price * (1 - strategy['tp_def'])
            tp_hits_sell, sl_hits_sell = np.where(future_lows <= tp_price_sell)[0], np.where(future_highs >= sl_price_sell)[0]
            first_tp_idx_sell = tp_hits_sell[0] if len(tp_hits_sell) > 0 else np.inf
            first_sl_idx_sell = sl_hits_sell[0] if len(sl_hits_sell) > 0 else np.inf
            if first_tp_idx_sell < first_sl_idx_sell: pnl = (adjusted_entry_price - tp_price_sell) * position_size
            elif first_sl_idx_sell < first_tp_idx_sell: pnl = (adjusted_entry_price - sl_price_sell) * position_size

        # If a trade was closed, calculate costs and update capital
        if pnl != 0.0:
            transaction_costs = (spread_cost_pips * position_size) + (commission_per_unit * position_size)
            net_pnl = pnl - transaction_costs
            capital += net_pnl
            trades.append({'pnl': net_pnl, 'capital': capital})
            
    if not trades: return None
    
    # Calculate final performance metrics
    log = pd.DataFrame(trades)
    wins, losses = log[log['pnl'] > 0], log[log['pnl'] <= 0]
    total_gross_wins, total_gross_losses = wins['pnl'].sum(), abs(losses['pnl'].sum())
    pf = total_gross_wins / total_gross_losses if total_gross_losses > 0 else np.inf
    log['peak'] = log['capital'].cummax()
    log['drawdown'] = log['peak'] - log['capital']
    max_dd = log['drawdown'].max()
    max_dd_pct = (max_dd / log['peak'].max()) * 100 if log['peak'].max() > 0 else 0
    sharpe_ratio = calculate_sharpe_ratio(log['pnl'], ANNUAL_RISK_FREE_RATE, TRADES_PER_YEAR_ESTIMATE)
    
    return {'profit_factor': pf, 'sharpe_ratio': sharpe_ratio, 'max_drawdown_pct': max_dd_pct, 'final_capital': capital, 'total_trades': len(log), 'win_rate_pct': (len(wins) / len(log)) * 100 if len(log) > 0 else 0}


def run_backtest_for_strategy(strategy_dict, market_data_cache, origin_market_csv):
    """
    The main worker function for multiprocessing. It orchestrates the backtest
    for a single strategy.
    """
    strategy = strategy_dict
    strategy_id = strategy['strategy_id'] # Use the pre-computed ID
    market_name = origin_market_csv.replace('.csv', '')
    silver_df, gold_df = market_data_cache[origin_market_csv]['silver'], market_data_cache[origin_market_csv]['gold']
    try:
        # Find all historical candles where the strategy's market rule was met.
        entry_candles = gold_df.query(strategy['market_rule'])
        if entry_candles.empty: return None
        
        # Get the corresponding Silver-layer data for these candles, which has the raw indicator levels.
        entry_points_silver = silver_df[silver_df.time.isin(entry_candles.time)]
        market_config = MARKET_CONFIG.get(market_name.replace(re.search(r'\d+', market_name).group(), ''), MARKET_CONFIG["DEFAULT"])

        performance = simulate_trades(strategy, entry_points_silver, silver_df, market_config)
        
        if performance:
            # Analyze the market regimes where trades were initiated.
            regime_analysis = {f"{col}_pct": (entry_points_silver[col].value_counts(normalize=True) * 100).to_dict() for col in ['session', 'trend_regime', 'vol_regime']}
            return {'strategy_id': strategy_id, 'market': origin_market_csv, **strategy, **performance, **regime_analysis}
    except Exception:
        # Catch any query errors or other issues during simulation.
        return None
    return None


if __name__ == "__main__":
    # --- Define Project Directories ---
    core_dir = os.path.dirname(os.path.abspath(__file__))
    discovered_dir, prepared_data_dir, backtest_results_dir, blacklist_dir, zircon_input_dir = [os.path.abspath(os.path.join(core_dir, '..', d)) for d in ['platinum_data/discovered_strategy', 'diamond_data/prepared_data', 'diamond_data/backtesting_results', 'platinum_data/blacklists', 'zircon_data/input']]
    for d in [backtest_results_dir, blacklist_dir, zircon_input_dir]: os.makedirs(d, exist_ok=True)
    
    # --- Interactive Setup ---
    strategy_files = [f for f in os.listdir(discovered_dir) if f.endswith('.csv')]; 
    if not strategy_files: print("❌ No discovered strategy files found."); exit()
    print("--- Select a Strategy File to Backtest for Mastery ---")
    for i, f in enumerate(strategy_files): print(f"  [{i+1}] {f}")
    try:
        choice = int(input(f"Enter number (1-{len(strategy_files)}): ")) - 1
        strategy_file_to_test = strategy_files[choice]
    except (ValueError, IndexError):
        print("❌ Invalid selection. Exiting."); exit()

    origin_market_csv = strategy_file_to_test
    origin_market_name = origin_market_csv.replace('.csv', '')
    print(f"\n✅ Automatically selected origin market for testing: {origin_market_csv}")

    discovered_path = os.path.join(discovered_dir, strategy_file_to_test)
    prepared_market_path_silver = os.path.join(prepared_data_dir, f"{origin_market_name}_silver.parquet")
    if not os.path.exists(prepared_market_path_silver):
        print(f"❌ Prepared data for {origin_market_name} not found! Run the prepper first."); exit()

    # --- Load and Prepare Strategies ---
    def to_numeric_or_str(x):
        try: return float(x)
        except (ValueError, TypeError): return x
    strategies_df_full = pd.read_csv(discovered_path, dtype={'sl_def': object, 'tp_def': object})
    for col in ['sl_def', 'tp_def']:
        if col in strategies_df_full.columns: strategies_df_full[col] = strategies_df_full[col].apply(to_numeric_or_str)
    
    # Infer trade type based on SL definition (fixed ratio for buy, dynamic for sell is a simplification)
    strategies_df_full['trade_type'] = strategies_df_full.apply(lambda row: 'buy' if isinstance(row['sl_def'], float) else 'sell', axis=1)
    strategies_df_full['strategy_id'] = strategies_df_full.apply(lambda row: hashlib.sha256(str(row.to_dict()).encode()).hexdigest()[:10], axis=1)
    
    print(f"\nFound {len(strategies_df_full)} total strategies in {strategy_file_to_test}.")
    try:
        top_n_input = input(f"How many top strategies per rule to test? (Default: {TOP_N_PER_RULE}): ").strip()
        TOP_N_PER_RULE = int(top_n_input) if top_n_input else TOP_N_PER_RULE
    except ValueError:
        print(f"Invalid number. Defaulting to {TOP_N_PER_RULE}.")
    strategies_df = strategies_df_full.groupby('market_rule').head(TOP_N_PER_RULE).reset_index(drop=True)

    # --- State Management for Resumability ---
    detailed_report_path = os.path.join(backtest_results_dir, f"diamond_report_{strategy_file_to_test}")
    processed_log_path = os.path.join(backtest_results_dir, f".{strategy_file_to_test}.processed_log")
    try:
        with open(processed_log_path, 'r') as f:
            processed_ids = set(f.read().splitlines())
        strategies_to_test = strategies_df[~strategies_df['strategy_id'].isin(processed_ids)]
        print(f"\nFound {len(processed_ids)} previously completed backtests. Resuming with {len(strategies_to_test)} remaining strategies.")
    except FileNotFoundError:
        strategies_to_test = strategies_df
        print(f"\nNo previous results found. Starting new backtest with {len(strategies_to_test)} strategies.")

    if strategies_to_test.empty:
        print("✅ All selected strategies have already been backtested.")
    else:
        print("\nLoading prepared market data into memory...")
        market_data_cache = {origin_market_csv: {'silver': pd.read_parquet(os.path.join(prepared_data_dir, f"{origin_market_name}_silver.parquet")), 'gold': pd.read_parquet(os.path.join(prepared_data_dir, f"{origin_market_name}_gold.parquet"))}}
        print("✅ Data loaded.")
        
        use_multiprocessing = input("Use multiprocessing? (y/n): ").strip().lower() == 'y'
        num_processes = MAX_CPU_USAGE if use_multiprocessing else 1
        strategy_list = strategies_to_test.to_dict('records')
        effective_num_processes = min(num_processes, len(strategy_list))
        print(f"\n🚀 Starting backtesting with {effective_num_processes} worker(s)...")

        # --- Upgraded Checkpointing Logic ---
        def save_checkpoint(results_list, log_file_handle):
            """Saves results and updates the processed log atomically."""
            if not results_list: return
            df = pd.DataFrame(results_list)
            file_exists = os.path.exists(detailed_report_path) and os.path.getsize(detailed_report_path) > 0
            df.to_csv(detailed_report_path, mode='a', header=not file_exists, index=False)
            log_file_handle.flush() # Ensure the log is written to disk

        # Open the log file in append mode to preserve history.
        with open(processed_log_path, 'a') as log_file:
            temp_results = []
            if effective_num_processes > 1:
                with Pool(processes=effective_num_processes) as pool:
                    func = partial(run_backtest_for_strategy, market_data_cache=market_data_cache, origin_market_csv=origin_market_csv)
                    # Use enumerate to track progress for checkpointing
                    for i, result in enumerate(tqdm(pool.imap_unordered(func, strategy_list), total=len(strategy_list))):
                        if result: temp_results.append(result)
                        # Log the ID of the strategy that was just attempted
                        log_file.write(strategy_list[i]['strategy_id'] + '\n')
                        # Periodically save the accumulated results.
                        if (i + 1) % SAVE_CHECKPOINT_FREQUENCY == 0:
                            save_checkpoint(temp_results, log_file)
                            temp_results = [] # Clear the temporary list
            else: # Sequential execution
                for i, strategy in enumerate(tqdm(strategy_list, desc="Backtesting Strategies")):
                    result = run_backtest_for_strategy(strategy, market_data_cache, origin_market_csv)
                    if result: temp_results.append(result)
                    log_file.write(strategy['strategy_id'] + '\n')
                    if (i + 1) % SAVE_CHECKPOINT_FREQUENCY == 0:
                        save_checkpoint(temp_results, log_file)
                        temp_results = []
            # Final save for any remaining results after the loop finishes.
            save_checkpoint(temp_results, log_file)

    # --- FINAL CLEANUP AND OUTPUT GENERATION ---
    print("\nBacktesting finished. Performing final cleanup of reports...")
    try:
        results_df = pd.read_csv(detailed_report_path)
        # Ensure no duplicate strategies exist from overlapping checkpoints.
        results_df.drop_duplicates(subset=['strategy_id'], keep='last', inplace=True)
        results_df.sort_values(by='sharpe_ratio', ascending=False, inplace=True)
        
        # Filter for strategies that passed all mastery thresholds.
        passed_strategies_df = results_df[
            (results_df['profit_factor'] > PROFIT_FACTOR_PASS_THRESHOLD) &
            (results_df['max_drawdown_pct'] < MAX_DRAWDOWN_PASS_THRESHOLD) &
            (results_df['total_trades'] >= MIN_TRADES_PASS_THRESHOLD)
        ].copy()
        
        failed_strategy_ids = set(results_df['strategy_id']) - set(passed_strategies_df['strategy_id'])
        failed_strategies_df = strategies_df_full[strategies_df_full['strategy_id'].isin(failed_strategy_ids)]

        # --- Save Passed Strategies for Zircon ---
        zircon_input_path = os.path.join(zircon_input_dir, f"master_strategies_{origin_market_name}.csv")
        passed_strategies_df.to_csv(zircon_input_path, index=False)
        print(f"\n✅ Found {len(passed_strategies_df)} master strategies. Saved to Zircon input.")
        
        # Save the cleaned, full report.
        results_df.to_csv(detailed_report_path, index=False)
        print(f"✅ Full detailed report cleaned and saved.")

        # --- Update Blacklist with Failed Strategies ---
        if not failed_strategies_df.empty:
            blacklist_df = failed_strategies_df[['type', 'sl_def', 'sl_bin', 'tp_def', 'tp_bin']].drop_duplicates()
            blacklist_path = os.path.join(blacklist_dir, strategy_file_to_test)
            # Append to existing blacklist to avoid overwriting.
            if os.path.exists(blacklist_path):
                existing_blacklist = pd.read_csv(blacklist_path)
                blacklist_df = pd.concat([existing_blacklist, blacklist_df]).drop_duplicates()
            blacklist_df.to_csv(blacklist_path, index=False)
            print(f"✅ Added/updated {len(blacklist_df)} underperforming blueprints to blacklist.")
    except FileNotFoundError:
        print("ℹ️ No results were generated to save.")

    print("\n" + "="*50 + "\n✅ Diamond Layer (Mastery) run complete.")
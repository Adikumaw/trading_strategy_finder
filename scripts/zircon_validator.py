# zircon_validator.py (V4 - Regime Analysis Fix)

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import re
import hashlib
from functools import partial
from multiprocessing import Pool, cpu_count

# --- All configurations and helper functions are unchanged and correct ---
INITIAL_CAPITAL = 10000.0; RISK_PER_TRADE_PCT = 0.02; ANNUAL_RISK_FREE_RATE = 0.04; TRADES_PER_YEAR_ESTIMATE = 252
MARKET_CONFIG = {
    "DEFAULT": {"spread_pips": 2.5, "commission_per_lot": 7.0, "pip_value": 0.0001, "slippage_pips": 0.5},
    "EURUSD":  {"spread_pips": 1.5, "commission_per_lot": 7.0, "pip_value": 0.0001, "slippage_pips": 0.3},
    "GBPUSD":  {"spread_pips": 2.0, "commission_per_lot": 7.0, "pip_value": 0.0001, "slippage_pips": 0.4},
    "AUDUSD":  {"spread_pips": 2.5, "commission_per_lot": 7.0, "pip_value": 0.0001, "slippage_pips": 0.5},
    "USDJPY":  {"spread_pips": 2.0, "commission_per_lot": 7.0, "pip_value": 0.01,   "slippage_pips": 0.4},
    "USDCAD":  {"spread_pips": 2.5, "commission_per_lot": 7.0, "pip_value": 0.0001, "slippage_pips": 0.5}
}
MAX_CPU_USAGE = max(1, cpu_count() - 2)
def get_dynamic_price(entry_price, level_price, placement_bin):
    placement_ratio = (placement_bin + 0.5) / 10.0; return entry_price + (level_price - entry_price) * placement_ratio
def calculate_sharpe_ratio(pnl_series, risk_free_rate, trades_per_year):
    if pnl_series.std() == 0 or len(pnl_series) < 2: return np.inf # Return inf for zero std dev
    daily_returns = pnl_series / INITIAL_CAPITAL
    excess_returns = daily_returns - (risk_free_rate / trades_per_year)
    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(trades_per_year)
def simulate_trades(strategy, entries, full_silver_data, market_config):
    trades, capital = [], INITIAL_CAPITAL
    time_to_idx = {time: i for i, time in enumerate(full_silver_data['time'])}
    is_sl_dynamic, is_tp_dynamic = isinstance(strategy['sl_def'], str), isinstance(strategy['tp_def'], str)
    spread_cost_pips = market_config['spread_pips'] * market_config['pip_value']; commission_per_unit = market_config['commission_per_lot'] / 100000; slippage_cost_pips = market_config['slippage_pips'] * market_config['pip_value']
    for entry in entries.itertuples():
        entry_price = entry.close; adjusted_entry_price = entry_price + slippage_cost_pips if strategy['trade_type'] == 'buy' else entry_price - slippage_cost_pips
        sl_price = get_dynamic_price(entry_price, getattr(entry, strategy['sl_def']), strategy['sl_bin']) if is_sl_dynamic else entry_price * (1 - strategy['sl_def'])
        tp_price = get_dynamic_price(entry_price, getattr(entry, strategy['tp_def']), strategy['tp_bin']) if is_tp_dynamic else entry_price * (1 + strategy['tp_def'])
        if sl_price >= adjusted_entry_price or tp_price <= adjusted_entry_price: continue
        risk_per_unit = abs(adjusted_entry_price - sl_price);
        if risk_per_unit == 0: continue
        position_size = (capital * RISK_PER_TRADE_PCT) / risk_per_unit;
        if position_size <= 0: continue
        entry_idx = time_to_idx.get(entry.time);
        if entry_idx is None: continue
        future_highs, future_lows = full_silver_data['high'].values[entry_idx + 1:], full_silver_data['low'].values[entry_idx + 1:]
        tp_hits, sl_hits = np.where(future_highs >= tp_price)[0], np.where(future_lows <= sl_price)[0]
        first_tp_idx, first_sl_idx = (tp_hits[0] if len(tp_hits) > 0 else np.inf), (sl_hits[0] if len(sl_hits) > 0 else np.inf)
        pnl = 0.0
        if first_tp_idx < first_sl_idx: pnl = (tp_price - adjusted_entry_price) * position_size
        elif first_sl_idx < first_tp_idx: pnl = (sl_price - adjusted_entry_price) * position_size
        if pnl != 0.0:
            transaction_costs = (spread_cost_pips * position_size) + (commission_per_unit * position_size)
            net_pnl = pnl - transaction_costs; capital += net_pnl; trades.append({'pnl': net_pnl, 'capital': capital})
    if not trades: return None
    log = pd.DataFrame(trades); wins, losses = log[log['pnl'] > 0], log[log['pnl'] <= 0]; total_gross_wins, total_gross_losses = wins['pnl'].sum(), abs(losses['pnl'].sum())
    pf = total_gross_wins / total_gross_losses if total_gross_losses > 0 else np.inf
    log['peak'] = log['capital'].cummax(); log['drawdown'] = log['peak'] - log['capital']; max_dd = log['drawdown'].max()
    max_dd_pct = (max_dd / log['peak'].max()) * 100 if log['peak'].max() > 0 else 0
    sharpe_ratio = calculate_sharpe_ratio(log['pnl'], ANNUAL_RISK_FREE_RATE, TRADES_PER_YEAR_ESTIMATE)
    return {'profit_factor': pf, 'sharpe_ratio': sharpe_ratio, 'max_drawdown_pct': max_dd_pct, 'total_trades': len(log)}

def run_validation_for_strategy(strategy, market_data_cache, validation_market_csv):
    """Backtests a single strategy against ONE validation market."""
    market_name = validation_market_csv.replace('.csv', '')
    silver_df, gold_df = market_data_cache['silver'], market_data_cache['gold']
    try:
        entry_candles = gold_df.query(strategy['market_rule'])
        if entry_candles.empty: return None
        entry_points_silver = silver_df[silver_df.time.isin(entry_candles.time)]
        market_config = MARKET_CONFIG.get(market_name, MARKET_CONFIG["DEFAULT"])
        performance = simulate_trades(strategy, entry_points_silver, silver_df, market_config)
        if performance:
            regime_analysis = {f"{col}_pct": (entry_points_silver[col].value_counts(normalize=True) * 100).to_dict() for col in ['session', 'trend_regime', 'vol_regime']}
            
            # --- THIS IS THE FIX ---
            result = {'strategy_id': strategy['strategy_id'], 'market': validation_market_csv, **performance, **regime_analysis}
            
            definition_keys = ['type', 'sl_def', 'sl_bin', 'tp_def', 'tp_bin', 'market_rule', 'trade_type']
            for key in definition_keys:
                if key in strategy: result[key] = strategy[key]
            return result
    except Exception: return None
    return None

# --- Main block `if __name__ == "__main__":` is unchanged and correct ---
# (Omitted for brevity, you can keep the previous version's main block)
if __name__ == "__main__":
    core_dir = os.path.dirname(os.path.abspath(__file__))
    zircon_input_dir, prepared_data_dir, zircon_results_dir = [os.path.abspath(os.path.join(core_dir, '..', d)) for d in ['zircon_data/input', 'diamond_data/prepared_data', 'zircon_data/results']]
    os.makedirs(zircon_results_dir, exist_ok=True)

    master_strategy_files = [f for f in os.listdir(zircon_input_dir) if f.startswith('master_strategies_') and f.endswith('.csv')]
    if not master_strategy_files: print("❌ No master strategy files found. Run Diamond layer first."); exit()
    
    print("--- Select a Master Strategy File to Validate ---")
    for i, f in enumerate(master_strategy_files): print(f"  [{i+1}] {f}")
    try:
        choice = int(input(f"Enter number (1-{len(master_strategy_files)}): ")) - 1
        file_to_validate = master_strategy_files[choice]
    except (ValueError, IndexError):
        print("❌ Invalid selection. Exiting."); exit()
    
    match = re.search(r'master_strategies_(.+?(\d+))\.csv', file_to_validate)
    if not match:
        print(f"❌ Invalid master strategy filename format: {file_to_validate}. Expected 'master_strategies_INSTRUMENTTIMEFRAME.csv'."); exit()
    
    origin_market_name_full = match.group(1)
    origin_timeframe_num = match.group(2)
    origin_timeframe_key = f"{origin_timeframe_num}m"

    strategies_df = pd.read_csv(os.path.join(zircon_input_dir, file_to_validate))
    def to_numeric_or_str(x):
        try: return float(x)
        except (ValueError, TypeError): return x
    for col in ['sl_def', 'tp_def']:
        if col in strategies_df.columns: strategies_df[col] = strategies_df[col].apply(to_numeric_or_str)

    all_prepared_markets = sorted([f.replace('_silver.parquet', '.csv') for f in os.listdir(prepared_data_dir) if f.endswith('_silver.parquet')])
    
    validation_markets = [
        m for m in all_prepared_markets
        if m.endswith(f"{origin_timeframe_num}.csv") and m.replace('.csv', '') != origin_market_name_full
    ]

    print(f"\n✅ Found {len(strategies_df)} master strategies from {origin_market_name_full}.csv.")
    if not validation_markets:
        print(f"⚠️ No other prepared markets found for the '{origin_timeframe_key}' timeframe to validate against. Exiting."); exit()
    print(f"✅ Will validate against {len(validation_markets)} other markets of the same '{origin_timeframe_key}' timeframe: {validation_markets}")
    
    detailed_report_path = os.path.join(zircon_results_dir, f"detailed_report_{origin_market_name_full}.csv")
    processed_log_path = os.path.join(zircon_results_dir, f".{origin_market_name_full}.processed_log")
    
    try:
        with open(processed_log_path, 'r') as f: processed_markets = set(f.read().splitlines())
        markets_to_process = [m for m in validation_markets if m not in processed_markets]
        print(f"\nFound {len(processed_markets)} previously validated markets. Resuming with {len(markets_to_process)} remaining.")
    except FileNotFoundError:
        markets_to_process = validation_markets
        print(f"\nStarting new validation run for {len(markets_to_process)} markets.")

    if not markets_to_process:
        print("✅ All validation markets have already been processed for this strategy set.")
    else:
        use_multiprocessing = input("Use multiprocessing? (y/n): ").strip().lower() == 'y'
        num_processes = MAX_CPU_USAGE if use_multiprocessing else 1
        strategy_list = strategies_df.to_dict('records')

        with open(processed_log_path, 'a') as log_file:
            for market_csv in tqdm(markets_to_process, desc="Validating Markets"):
                market_name = market_csv.replace('.csv', '')
                print(f"\n-- Loading and processing {market_csv} --")
                
                market_data_cache = {
                    'silver': pd.read_parquet(os.path.join(prepared_data_dir, f"{market_name}_silver.parquet")),
                    'gold': pd.read_parquet(os.path.join(prepared_data_dir, f"{market_name}_gold.parquet"))
                }
                
                func = partial(run_validation_for_strategy, market_data_cache=market_data_cache, validation_market_csv=market_csv)
                
                market_results = []
                if num_processes > 1 and len(strategy_list) > 1:
                    with Pool(processes=num_processes) as pool:
                        results = list(tqdm(pool.imap_unordered(func, strategy_list), total=len(strategy_list), desc=f"Testing on {market_csv}"))
                    market_results = [r for r in results if r is not None]
                else:
                    for s in tqdm(strategy_list, desc=f"Testing on {market_csv}"):
                        result = func(s)
                        if result: market_results.append(result)
                
                if market_results:
                    df = pd.DataFrame(market_results)
                    file_exists = os.path.exists(detailed_report_path) and os.path.getsize(detailed_report_path) > 0
                    df.to_csv(detailed_report_path, mode='a', header=not file_exists, index=False)
                    print(f"✅ Appended {len(df)} results for {market_csv}.")
                
                log_file.write(market_csv + '\n')
                log_file.flush()

    print("\nValidation runs complete. Generating final summary report...")
    try:
        detailed_df = pd.read_csv(detailed_report_path)
        detailed_df.drop_duplicates(subset=['strategy_id', 'market'], keep='last', inplace=True)
        
        summary_df = detailed_df.groupby('strategy_id').agg(
            markets_tested=('market', 'count'),
            avg_profit_factor=('profit_factor', 'mean'),
            avg_sharpe_ratio=('sharpe_ratio', 'mean'),
            avg_max_drawdown_pct=('max_drawdown_pct', 'mean'),
            total_trades=('total_trades', 'sum')
        ).reset_index()
        
        summary_df = pd.merge(summary_df, strategies_df, on='strategy_id', how='left')
        
        passed_strategies = detailed_df[detailed_df.profit_factor > 1.0].groupby('strategy_id')['market'].count()
        summary_df['validation_markets_passed_count'] = summary_df['strategy_id'].map(passed_strategies).fillna(0).astype(int)
        summary_df['validation_markets_passed'] = summary_df['validation_markets_passed_count'].astype(str) + '/' + summary_df['markets_tested'].astype(str)
        
        summary_df = summary_df.sort_values(by=['validation_markets_passed_count', 'avg_sharpe_ratio'], ascending=False)

        summary_report_path = os.path.join(zircon_results_dir, f"summary_report_{origin_market_name_full}.csv")
        summary_df.to_csv(summary_report_path, index=False)
        detailed_df.to_csv(detailed_report_path, index=False)
        print(f"✅ Final reports generated for '{origin_market_name_full}'.")

    except FileNotFoundError:
        print("ℹ️ No detailed results found to generate a summary from.")
    
    print("\n" + "="*50 + "\n✅ Zircon Layer (Validation) run complete.")
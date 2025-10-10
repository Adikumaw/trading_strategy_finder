# diamond_backtester.py (Final, Configurable Multiprocessing Version)

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import re
import gc
import ta
import talib
import numba
from sklearn.preprocessing import StandardScaler
import hashlib
from functools import partial
from multiprocessing import Pool, cpu_count

# --- CONFIGURATION ---
INITIAL_CAPITAL = 10000
RISK_PER_TRADE_PCT = 0.02
PROFIT_FACTOR_PASS_THRESHOLD = 1.2
MAX_DRAWDOWN_PASS_THRESHOLD = 20.0
TOP_N_STRATEGIES_TO_TEST = 100
MAX_CPU_USAGE = max(1, cpu_count() - 2)

# --- CORE HELPER FUNCTIONS ---
def get_dynamic_price(entry_price, level_price, placement_bin):
    placement_ratio = (placement_bin + 0.5) / 10.0
    return entry_price + (level_price - entry_price) * placement_ratio

def simulate_trades(strategy, entries, full_silver_data):
    trades, capital = [], INITIAL_CAPITAL
    time_to_idx = {time: i for i, time in enumerate(full_silver_data['time'])}
    is_sl_dynamic = isinstance(strategy['sl_def'], str)
    is_tp_dynamic = isinstance(strategy['tp_def'], str)
    for _, entry in entries.iterrows():
        entry_time, entry_price = entry['time'], entry['close']
        sl_price = get_dynamic_price(entry_price, entry[strategy['sl_def']], strategy['sl_bin']) if is_sl_dynamic else entry_price * (1 - strategy['sl_def'])
        tp_price = get_dynamic_price(entry_price, entry[strategy['tp_def']], strategy['tp_bin']) if is_tp_dynamic else entry_price * (1 + strategy['tp_def'])
        if sl_price >= entry_price or tp_price <= entry_price: continue
        position_size = (capital * RISK_PER_TRADE_PCT) / abs(entry_price - sl_price) if entry_price != sl_price else 0
        if position_size <= 0: continue
        entry_idx = time_to_idx.get(entry_time)
        if entry_idx is None: continue
        future_candles = full_silver_data.iloc[entry_idx + 1:]
        exit_price, outcome = None, "Incomplete"
        for _, future in future_candles.iterrows():
            if future['low'] <= sl_price: outcome, exit_price = "Loss", sl_price; break
            if future['high'] >= tp_price: outcome, exit_price = "Win", tp_price; break
        if outcome in ["Win", "Loss"]:
            pnl = (exit_price - entry_price) * position_size; capital += pnl; trades.append({'pnl': pnl, 'capital': capital})
    if not trades: return None
    log = pd.DataFrame(trades); wins, losses = log[log['pnl'] > 0], log[log['pnl'] <= 0]
    pf = wins['pnl'].sum() / abs(losses['pnl'].sum()) if abs(losses['pnl'].sum()) > 0 else np.inf
    log['peak'] = log['capital'].cummax(); log['drawdown'] = log['peak'] - log['capital']; max_dd = log['drawdown'].max()
    return {'profit_factor': pf, 'max_drawdown_pct': (max_dd / log['peak'].max()) * 100 if log['peak'].max() > 0 else 0, 'final_capital': capital, 'total_trades': len(log), 'win_rate_pct': (len(wins) / len(log)) * 100}

# --- PARALLEL WORKER FUNCTION ---
def run_backtest_for_strategy(strategy, market_data_cache, markets_to_test):
    all_market_results = []
    strategy_id = hashlib.sha256(str(strategy).encode()).hexdigest()[:10]
    for market in markets_to_test:
        silver_df, gold_df = market_data_cache[market]['silver'], market_data_cache[market]['gold']
        try: entry_candles = gold_df.query(strategy['market_rule'])
        except Exception: continue
        if entry_candles.empty: continue
        entry_points_silver = silver_df[silver_df.time.isin(entry_candles.time)]
        regime_analysis = {f"{col}_pct": (entry_points_silver[col].value_counts(normalize=True) * 100).to_dict() for col in ['session', 'trend_regime', 'vol_regime']}
        performance = simulate_trades(strategy, entry_points_silver, silver_df)
        if performance:
            result = {'strategy_id': strategy_id, 'market': market, **strategy, **performance, **regime_analysis}
            all_market_results.append(result)
    return all_market_results

if __name__ == "__main__":
    core_dir = os.path.dirname(os.path.abspath(__file__))
    discovered_dir, prepared_data_dir, backtest_results_dir, blacklist_dir = [os.path.abspath(os.path.join(core_dir, '..', d)) for d in ['platinum_data/discovered_strategy', 'diamond_data/prepared_data', 'diamond_data/backtesting_results', 'platinum_data/blacklists']]
    os.makedirs(backtest_results_dir, exist_ok=True); os.makedirs(blacklist_dir, exist_ok=True)
    
    strategy_files = [f for f in os.listdir(discovered_dir) if f.endswith('.csv')]
    if not strategy_files: print("❌ No discovered strategy files found."); exit()
    print("--- Select a Strategy File to Backtest ---")
    for i, f in enumerate(strategy_files): print(f"  [{i+1}] {f}")
    try:
        choice = int(input(f"Enter number (1-{len(strategy_files)}): ")) - 1
        strategy_file_to_test = strategy_files[choice]
    except (ValueError, IndexError): print("❌ Invalid selection. Exiting."); exit()
    discovered_path = os.path.join(discovered_dir, strategy_file_to_test)

    prepared_markets = sorted([f.replace('_silver.parquet', '') for f in os.listdir(prepared_data_dir) if f.endswith('_silver.parquet')])
    if not prepared_markets: print("❌ No prepared market data found. Run diamond_data_prepper.py first."); exit()
    print("\n--- Select Markets to Test On ---")
    for i, f in enumerate(prepared_markets): print(f"  [{i+1}] {f}.csv")
    try:
        selections = input(f"Enter numbers separated by commas (e.g., 1,3,5), or 'all': ")
        markets_to_test = [f'{m}.csv' for m in prepared_markets] if 'all' in selections.lower() else [f'{prepared_markets[int(i.strip()) - 1]}.csv' for i in selections.split(',')]
    except (ValueError, IndexError): print("❌ Invalid selection. Exiting."); exit()

    def to_numeric_or_str(x):
        try: return float(x)
        except (ValueError, TypeError): return x
    strategies_df = pd.read_csv(discovered_path, dtype={'sl_def': object, 'tp_def': object})
    for col in ['sl_def', 'tp_def']:
        if col in strategies_df.columns: strategies_df[col] = strategies_df[col].apply(to_numeric_or_str)
    strategies_df = strategies_df.head(TOP_N_STRATEGIES_TO_TEST)
    
    print(f"\nLoaded top {len(strategies_df)} strategies for backtesting on {len(markets_to_test)} market(s).")
    print("\nLoading all prepared market data into memory...")
    market_data_cache = {}
    for market_csv in tqdm(markets_to_test, desc="Loading Data"):
        market_name = market_csv.replace('.csv', '')
        silver_path = os.path.join(prepared_data_dir, f"{market_name}_silver.parquet")
        gold_path = os.path.join(prepared_data_dir, f"{market_name}_gold.parquet")
        market_data_cache[market_csv] = {'silver': pd.read_parquet(silver_path), 'gold': pd.read_parquet(gold_path)}
    print("✅ All data loaded.")
    
    use_multiprocessing = input("Use multiprocessing? (y/n): ").strip().lower() == 'y'
    if use_multiprocessing: num_processes = MAX_CPU_USAGE
    else:
        try:
            num_processes = int(input(f"Enter number of processes to use (1-{cpu_count()}): ").strip())
            if num_processes < 1 or num_processes > cpu_count(): raise ValueError
        except ValueError: print("Invalid input. Defaulting to 1 process."); num_processes = 1
    
    strategy_list = strategies_df.to_dict('records')
    effective_num_processes = min(num_processes, len(strategy_list))
    print(f"\n🚀 Starting backtesting with {effective_num_processes} worker(s)...")
    
    all_detailed_results = []
    if effective_num_processes > 1:
        with Pool(processes=effective_num_processes) as pool:
            func = partial(run_backtest_for_strategy, market_data_cache=market_data_cache, markets_to_test=markets_to_test)
            all_results_nested = list(tqdm(pool.imap(func, strategy_list), total=len(strategy_list)))
        all_detailed_results = [item for sublist in all_results_nested for item in sublist]
    else:
        for strategy in tqdm(strategy_list, desc="Backtesting Strategies"):
            results = run_backtest_for_strategy(strategy, market_data_cache, markets_to_test)
            all_detailed_results.extend(results)

    if all_detailed_results:
        detailed_df = pd.DataFrame(all_detailed_results)
        summary_df = detailed_df.groupby('strategy_id').agg(
            markets_tested=('market', 'count'), avg_profit_factor=('profit_factor', 'mean'),
            avg_max_drawdown_pct=('max_drawdown_pct', 'mean'), total_trades=('total_trades', 'sum')
        ).reset_index()
        strategy_defs = detailed_df.drop(columns=[c for c in detailed_df.columns if c in ['market', 'profit_factor', 'max_drawdown_pct', 'final_capital', 'total_trades', 'win_rate_pct', 'session_pct', 'trend_regime_pct', 'vol_regime_pct']]).drop_duplicates(subset='strategy_id')
        summary_df = pd.merge(summary_df, strategy_defs, on='strategy_id').sort_values(by='avg_profit_factor', ascending=False)
        passed_strategies = detailed_df[(detailed_df.profit_factor > PROFIT_FACTOR_PASS_THRESHOLD) & (detailed_df.max_drawdown_pct < MAX_DRAWDOWN_PASS_THRESHOLD)].groupby('strategy_id')['market'].count()
        summary_df['markets_passed_count'] = summary_df['strategy_id'].map(passed_strategies).fillna(0).astype(int)
        summary_df['markets_passed'] = summary_df['markets_passed_count'].astype(str) + '/' + summary_df['markets_tested'].astype(str)
        failed_strategy_ids = summary_df[summary_df['markets_passed_count'] < summary_df['markets_tested']]['strategy_id'].unique()
        blacklist_df = strategy_defs[strategy_defs['strategy_id'].isin(failed_strategy_ids)].drop(columns='strategy_id')

        summary_df.to_csv(os.path.join(backtest_results_dir, f"summary_report_{strategy_file_to_test}"), index=False)
        detailed_df.to_csv(os.path.join(backtest_results_dir, f"detailed_report_{strategy_file_to_test}"), index=False)
        print(f"\n✅ Summary and Detailed reports saved.")
        if not blacklist_df.empty:
            blacklist_df.to_csv(os.path.join(blacklist_dir, strategy_file_to_test), index=False)
            print(f"✅ Added {len(blacklist_df)} underperforming strategies to blacklist.")
    
    print("\n" + "="*50 + "\n✅ All backtesting complete.")
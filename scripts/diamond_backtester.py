# diamond_backtester.py (Professional Upgrade)

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import re
import hashlib
from functools import partial
from multiprocessing import Pool, cpu_count

# --- PROFESSIONAL BACKTESTING CONFIGURATION ---
INITIAL_CAPITAL = 10000.0
RISK_PER_TRADE_PCT = 0.02 # Risk 2% of current capital on each trade
ANNUAL_RISK_FREE_RATE = 0.04 # For Sharpe Ratio calculation (e.g., 4% T-bill rate)
TRADES_PER_YEAR_ESTIMATE = 252 # Assumed trading days for annualizing Sharpe Ratio

# --- Market-Specific Transaction Cost Configuration ---
MARKET_CONFIG = {
    "DEFAULT":      {"spread_pips": 2.5, "commission_per_lot": 7.0, "pip_value": 0.0001, "slippage_pips": 0.5},
    "EURUSD":       {"spread_pips": 1.5, "commission_per_lot": 7.0, "pip_value": 0.0001, "slippage_pips": 0.3},
    "GBPUSD":       {"spread_pips": 2.0, "commission_per_lot": 7.0, "pip_value": 0.0001, "slippage_pips": 0.4},
    "AUDUSD":       {"spread_pips": 2.5, "commission_per_lot": 7.0, "pip_value": 0.0001, "slippage_pips": 0.5},
    "USDJPY":       {"spread_pips": 2.0, "commission_per_lot": 7.0, "pip_value": 0.01,   "slippage_pips": 0.4},
    "USDCAD":       {"spread_pips": 2.5, "commission_per_lot": 7.0, "pip_value": 0.0001, "slippage_pips": 0.5}
}

# --- General Script Configuration (Unchanged) ---
PROFIT_FACTOR_PASS_THRESHOLD = 1.2
MAX_DRAWDOWN_PASS_THRESHOLD = 20.0
TOP_N_PER_RULE = 5
MAX_CPU_USAGE = max(1, cpu_count() - 2)

# --- HELPER FUNCTIONS ---
def get_dynamic_price(entry_price, level_price, placement_bin):
    placement_ratio = (placement_bin + 0.5) / 10.0
    return entry_price + (level_price - entry_price) * placement_ratio

def calculate_sharpe_ratio(pnl_series, risk_free_rate, trades_per_year):
    """Calculates the annualized Sharpe Ratio."""
    if pnl_series.std() == 0 or len(pnl_series) < 2: return 0.0
    daily_returns = pnl_series / INITIAL_CAPITAL # Simplified return calculation
    excess_returns = daily_returns - (risk_free_rate / trades_per_year)
    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(trades_per_year)

def simulate_trades(strategy, entries, full_silver_data, market_config):
    """Core backtesting engine with costs, slippage, and advanced metrics."""
    trades, capital = [], INITIAL_CAPITAL
    
    # Pre-calculate mapping for speed
    time_to_idx = {time: i for i, time in enumerate(full_silver_data['time'])}
    
    # Check if SL/TP are dynamic (based on indicators) or static (ratio-based)
    is_sl_dynamic = isinstance(strategy['sl_def'], str)
    is_tp_dynamic = isinstance(strategy['tp_def'], str)
    
    # Unpack market-specific costs
    spread_cost_pips = market_config['spread_pips'] * market_config['pip_value']
    commission_per_unit = market_config['commission_per_lot'] / 100000
    slippage_cost_pips = market_config['slippage_pips'] * market_config['pip_value']

    for entry in entries.itertuples():
        entry_price = entry.close
        
        # --- SLIPPAGE SIMULATION: Make the entry price slightly worse ---
        # For a buy, we assume we buy slightly higher; for a sell, slightly lower.
        # This is a fixed slippage model; a more complex one could use ATR.
        adjusted_entry_price = entry_price + slippage_cost_pips if strategy['trade_type'] == 'buy' else entry_price - slippage_cost_pips

        # Determine SL/TP prices based on the original entry candle's context
        sl_price = get_dynamic_price(entry_price, getattr(entry, strategy['sl_def']), strategy['sl_bin']) if is_sl_dynamic else entry_price * (1 - strategy['sl_def'])
        tp_price = get_dynamic_price(entry_price, getattr(entry, strategy['tp_def']), strategy['tp_bin']) if is_tp_dynamic else entry_price * (1 + strategy['tp_def'])
        
        # Basic sanity check
        if sl_price >= adjusted_entry_price or tp_price <= adjusted_entry_price: continue
            
        # --- RISK MANAGEMENT: Fixed Fractional Sizing ---
        risk_per_unit = abs(adjusted_entry_price - sl_price)
        if risk_per_unit == 0: continue
        position_size = (capital * RISK_PER_TRADE_PCT) / risk_per_unit
        if position_size <= 0: continue
            
        # --- FORWARD LOOK: Find trade outcome ---
        entry_idx = time_to_idx.get(entry.time)
        if entry_idx is None: continue
        
        future_highs = full_silver_data['high'].values[entry_idx + 1:]
        future_lows = full_silver_data['low'].values[entry_idx + 1:]
        
        # Determine TP/SL hit indices
        tp_hits = np.where(future_highs >= tp_price)[0]
        sl_hits = np.where(future_lows <= sl_price)[0]
        first_tp_idx = tp_hits[0] if len(tp_hits) > 0 else np.inf
        first_sl_idx = sl_hits[0] if len(sl_hits) > 0 else np.inf

        # --- P&L CALCULATION (NET OF COSTS) ---
        pnl = 0.0
        if first_tp_idx < first_sl_idx: # WIN
            pnl = (tp_price - adjusted_entry_price) * position_size
        elif first_sl_idx < first_tp_idx: # LOSS
            pnl = (sl_price - adjusted_entry_price) * position_size

        if pnl != 0.0:
            # Deduct transaction costs from the gross P&L
            transaction_costs = (spread_cost_pips * position_size) + (commission_per_unit * position_size)
            net_pnl = pnl - transaction_costs
            capital += net_pnl
            trades.append({'pnl': net_pnl, 'capital': capital})

    if not trades: return None
    
    # --- PERFORMANCE METRIC CALCULATION ---
    log = pd.DataFrame(trades)
    wins, losses = log[log['pnl'] > 0], log[log['pnl'] <= 0]
    
    # Handle the "zero losses" case gracefully for Profit Factor
    total_gross_wins = wins['pnl'].sum()
    total_gross_losses = abs(losses['pnl'].sum())
    pf = total_gross_wins / total_gross_losses if total_gross_losses > 0 else np.inf

    # Drawdown calculation
    log['peak'] = log['capital'].cummax()
    log['drawdown'] = log['peak'] - log['capital']
    max_dd = log['drawdown'].max()
    max_dd_pct = (max_dd / log['peak'].max()) * 100 if log['peak'].max() > 0 else 0
    
    # Sharpe Ratio
    sharpe_ratio = calculate_sharpe_ratio(log['pnl'], ANNUAL_RISK_FREE_RATE, TRADES_PER_YEAR_ESTIMATE)
    
    return {
        'profit_factor': pf, 'sharpe_ratio': sharpe_ratio, 'max_drawdown_pct': max_dd_pct, 
        'final_capital': capital, 'total_trades': len(log), 
        'win_rate_pct': (len(wins) / len(log)) * 100 if len(log) > 0 else 0
    }

def run_backtest_for_strategy(strategy_dict, market_data_cache, markets_to_test):
    """Wrapper function for parallel processing."""
    strategy = strategy_dict
    all_market_results = []
    strategy_id = hashlib.sha256(str(strategy).encode()).hexdigest()[:10]
    
    for market_csv in markets_to_test:
        market_name = market_csv.replace('.csv', '')
        silver_df, gold_df = market_data_cache[market_csv]['silver'], market_data_cache[market_csv]['gold']
        
        try:
            entry_candles = gold_df.query(strategy['market_rule'])
            if entry_candles.empty: continue
            
            # Important: Get the full silver data for the entry candles for context
            entry_points_silver = silver_df[silver_df.time.isin(entry_candles.time)]
            
            # Determine market config for costs
            market_config = MARKET_CONFIG.get(market_name, MARKET_CONFIG["DEFAULT"])
            
            # Perform the simulation
            performance = simulate_trades(strategy, entry_points_silver, silver_df, market_config)
            
            if performance:
                # Regime Analysis (unchanged)
                regime_analysis = {f"{col}_pct": (entry_points_silver[col].value_counts(normalize=True) * 100).to_dict() for col in ['session', 'trend_regime', 'vol_regime']}
                result = {'strategy_id': strategy_id, 'market': market_csv, **strategy, **performance, **regime_analysis}
                all_market_results.append(result)

        except Exception as e:
            # print(f"Warning: Rule failed for {strategy_id} on {market_csv}. Rule: {strategy['market_rule']}. Error: {e}")
            continue
            
    return all_market_results

if __name__ == "__main__":
    # --- Setup and User Prompts (largely unchanged) ---
    core_dir = os.path.dirname(os.path.abspath(__file__)); discovered_dir, prepared_data_dir, backtest_results_dir, blacklist_dir = [os.path.abspath(os.path.join(core_dir, '..', d)) for d in ['platinum_data/discovered_strategy', 'diamond_data/prepared_data', 'diamond_data/backtesting_results', 'platinum_data/blacklists']]; os.makedirs(backtest_results_dir, exist_ok=True); os.makedirs(blacklist_dir, exist_ok=True)
    strategy_files = [f for f in os.listdir(discovered_dir) if f.endswith('.csv')]; 
    if not strategy_files: print("❌ No discovered strategy files found."); exit()
    print("--- Select a Strategy File to Backtest ---")
    for i, f in enumerate(strategy_files): print(f"  [{i+1}] {f}")
    try: choice = int(input(f"Enter number (1-{len(strategy_files)}): ")) - 1; strategy_file_to_test = strategy_files[choice]
    except (ValueError, IndexError): print("❌ Invalid selection. Exiting."); exit()
    discovered_path = os.path.join(discovered_dir, strategy_file_to_test)
    prepared_markets = sorted([f.replace('_silver.parquet', '') for f in os.listdir(prepared_data_dir) if f.endswith('_silver.parquet')])
    if not prepared_markets: print("❌ No prepared market data found. Run diamond_data_prepper.py first."); exit()
    print("\n--- Select Markets to Test On ---")
    for i, f in enumerate(prepared_markets): print(f"  [{i+1}] {f}.csv")
    try:
        selections = input(f"Enter numbers separated by commas (e.g., 1,3,5), or 'all': "); markets_to_test = [f'{m}.csv' for m in prepared_markets] if 'all' in selections.lower() else [f'{prepared_markets[int(i.strip()) - 1]}.csv' for i in selections.split(',')]
    except (ValueError, IndexError): print("❌ Invalid selection. Exiting."); exit()
    
    # --- Loading and Data Type Handling ---
    def to_numeric_or_str(x):
        try: return float(x)
        except (ValueError, TypeError): return x
    strategies_df_full = pd.read_csv(discovered_path, dtype={'sl_def': object, 'tp_def': object})
    for col in ['sl_def', 'tp_def']:
        if col in strategies_df_full.columns: strategies_df_full[col] = strategies_df_full[col].apply(to_numeric_or_str)
    
    # Infer trade_type from SL/TP definitions (needed for slippage)
    def infer_trade_type(row):
        is_sl_ratio = isinstance(row['sl_def'], float)
        return 'buy' if is_sl_ratio else 'sell' # Simple assumption based on our Bronze Layer logic
    strategies_df_full['trade_type'] = strategies_df_full.apply(infer_trade_type, axis=1)

    print(f"\nFound {len(strategies_df_full)} total strategies in the selected file.")
    try: top_n_input = input(f"How many top strategies per rule to test? (Default: {TOP_N_PER_RULE}): ").strip(); TOP_N_PER_RULE = int(top_n_input) if top_n_input else TOP_N_PER_RULE
    except ValueError: print(f"Invalid number. Defaulting to {TOP_N_PER_RULE}.")
    strategies_df = strategies_df_full.groupby('market_rule').head(TOP_N_PER_RULE).reset_index(drop=True)
    print(f"\nSelected {len(strategies_df)} strategies for backtesting.")

    # --- Data Loading and Multiprocessing ---
    print("\nLoading all prepared market data into memory...")
    market_data_cache = {m_csv: {'silver': pd.read_parquet(os.path.join(prepared_data_dir, f"{m_csv.replace('.csv','')}_silver.parquet")), 'gold': pd.read_parquet(os.path.join(prepared_data_dir, f"{m_csv.replace('.csv','')}_gold.parquet"))} for m_csv in tqdm(markets_to_test)}
    print("✅ All data loaded.")
    
    use_multiprocessing = input("Use multiprocessing? (y/n): ").strip().lower() == 'y'
    num_processes = MAX_CPU_USAGE if use_multiprocessing else 1
    
    strategy_list = strategies_df.to_dict('records'); effective_num_processes = min(num_processes, len(strategy_list))
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
        
        # --- NEW & IMPROVED REPORT GENERATION ---
        summary_df = detailed_df.groupby('strategy_id').agg(
            markets_tested=('market', 'count'),
            avg_profit_factor=('profit_factor', 'mean'),
            avg_sharpe_ratio=('sharpe_ratio', 'mean'), # New metric
            avg_max_drawdown_pct=('max_drawdown_pct', 'mean'),
            total_trades=('total_trades', 'sum')
        ).reset_index()
        
        strategy_definition_cols = [col for col in strategies_df.columns if col in detailed_df.columns]
        strategy_defs = detailed_df[['strategy_id'] + strategy_definition_cols].drop_duplicates(subset='strategy_id')
        summary_df = pd.merge(summary_df, strategy_defs, on='strategy_id').sort_values(by='avg_profit_factor', ascending=False)
        
        passed_strategies = detailed_df[(detailed_df.profit_factor > PROFIT_FACTOR_PASS_THRESHOLD) & (detailed_df.max_drawdown_pct < MAX_DRAWDOWN_PASS_THRESHOLD)].groupby('strategy_id')['market'].count()
        summary_df['markets_passed_count'] = summary_df['strategy_id'].map(passed_strategies).fillna(0).astype(int)
        summary_df['markets_passed'] = summary_df['markets_passed_count'].astype(str) + '/' + summary_df['markets_tested'].astype(str)
        
        # Blacklist logic now considers if ANY market failed
        failed_strategy_ids = summary_df[summary_df['markets_passed_count'] < summary_df['markets_tested']]['strategy_id'].unique()
        blacklist_df = strategy_defs[strategy_defs['strategy_id'].isin(failed_strategy_ids)].drop(columns='strategy_id')

        # Save reports
        summary_df.to_csv(os.path.join(backtest_results_dir, f"summary_report_{strategy_file_to_test}"), index=False)
        detailed_df.to_csv(os.path.join(backtest_results_dir, f"detailed_report_{strategy_file_to_test}"), index=False)
        print(f"\n✅ Summary and Detailed reports saved.")
        if not blacklist_df.empty:
            blacklist_df.to_csv(os.path.join(blacklist_dir, strategy_file_to_test), index=False)
            print(f"✅ Added {len(blacklist_df)} underperforming strategies to blacklist.")
    
    print("\n" + "="*50 + "\n✅ All backtesting complete.")
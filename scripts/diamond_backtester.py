# diamond_backtester.py (Final Version with Correct Report Generation)

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

# --- CONFIGURATION & ALL HELPER FUNCTIONS ARE CORRECT ---
# ... (The first ~200 lines of the script are unchanged and correct) ...
INITIAL_CAPITAL = 10000
RISK_PER_TRADE_PCT = 0.02; PROFIT_FACTOR_PASS_THRESHOLD = 1.2; MAX_DRAWDOWN_PASS_THRESHOLD = 20.0; TOP_N_PER_RULE = 5; MAX_CPU_USAGE = max(1, cpu_count() - 2)
SMA_PERIODS, EMA_PERIODS = [20, 50, 100, 200], [8, 13, 21, 50]; BBANDS_PERIOD, BBANDS_STD_DEV, RSI_PERIOD = 20, 2.0, 14; MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9; ATR_PERIOD, ADX_PERIOD, PIVOT_WINDOW = 14, 14, 10; INDICATOR_WARMUP_PERIOD = 200
def robust_read_csv(filepath):
    df = pd.read_csv(filepath, header=None, sep=None, engine='python')
    if df.shape[1] > 5: df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
    elif df.shape[1] == 5: df.columns = ['time', 'open', 'high', 'low', 'close']; df['volume'] = 1
    else: raise ValueError(f"File '{filepath}' does not have 5 or 6 columns.")
    df['time'] = pd.to_datetime(df['time']); df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric)
    return df.sort_values('time').reset_index(drop=True)
@numba.njit
def _calculate_s_r_numba(lows, highs, window):
    n = len(lows); support, resistance = np.full(n, np.nan, dtype=np.float32), np.full(n, np.nan, dtype=np.float32)
    for i in range(window, n - window):
        if lows[i] == np.min(lows[i - window : i + window + 1]): support[i] = lows[i]
        if highs[i] == np.max(highs[i - window : i + window + 1]): resistance[i] = highs[i]
    return support, resistance
def add_support_resistance(df, window=PIVOT_WINDOW):
    s_pts, r_pts = _calculate_s_r_numba(df["low"].values.astype(np.float32), df["high"].values.astype(np.float32), window)
    sr_df = pd.DataFrame({'s_pts': s_pts, 'r_pts': r_pts}, index=df.index); df["support"], df["resistance"] = sr_df["s_pts"].ffill(), sr_df["r_pts"].ffill()
    return df
def add_all_market_features(df):
    indicator_df = pd.DataFrame(index=df.index)
    for p in SMA_PERIODS: indicator_df[f"SMA_{p}"] = ta.trend.SMAIndicator(df["close"], p).sma_indicator()
    for p in EMA_PERIODS: indicator_df[f"EMA_{p}"] = ta.trend.EMAIndicator(df["close"], p).ema_indicator()
    bb = ta.volatility.BollingerBands(df["close"], BBANDS_PERIOD, BBANDS_STD_DEV); indicator_df["BB_upper"], indicator_df["BB_lower"], indicator_df["BB_width"] = bb.bollinger_hband(), bb.bollinger_lband(), bb.bollinger_wband()
    indicator_df[f"RSI_{RSI_PERIOD}"] = ta.momentum.RSIIndicator(df["close"], RSI_PERIOD).rsi(); indicator_df["MACD_hist"] = ta.trend.MACD(df["close"], MACD_SLOW, MACD_FAST, MACD_SIGNAL).macd_diff()
    indicator_df[f"ATR_{ATR_PERIOD}"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], ATR_PERIOD).average_true_range(); indicator_df["ADX"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], ADX_PERIOD).adx()
    indicator_df["OBV"] = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume(); atr = indicator_df[f"ATR_{ATR_PERIOD}"]
    indicator_df["ATR_level_up_1x"], indicator_df["ATR_level_down_1x"] = df["close"] + atr, df["close"] - atr
    price_action_df = pd.DataFrame(index=df.index)
    price_action_df['session'] = df['time'].dt.hour.map(lambda h: 'London' if 7<=h<12 else 'London_NY_Overlap' if 12<=h<16 else 'New_York' if 16<=h<21 else 'Asian')
    price_action_df['trend_regime'] = np.where(indicator_df['ADX'] > 25, 'trend', 'range'); price_action_df['vol_regime'] = np.where(atr > atr.rolling(50).mean(), 'high_vol', 'low_vol')
    patterns_df = pd.DataFrame({p: getattr(talib, p)(df["open"], df["high"], df["low"], df["close"]) for p in talib.get_function_groups().get("Pattern Recognition", [])}, index=df.index)
    combined = pd.concat([df, indicator_df, price_action_df, patterns_df], axis=1)
    return add_support_resistance(combined)
def create_gold_features(features_df, scaler=None):
    df = features_df.copy()
    abs_price_cols = [col for col in df.columns if re.match(r'^(open|high|low|close|SMA_\d+|EMA_\d+|BB_(upper|lower)|support|resistance|ATR_level_.+)$', col)]
    for col in abs_price_cols:
        if col != 'close': df[f'{col}_dist_norm'] = (df['close'] - df[col]) / df['close']
    df.drop(columns=abs_price_cols + ['volume'], inplace=True, errors='ignore')
    df = pd.get_dummies(df, columns=['session', 'trend_regime', 'vol_regime'], drop_first=True)
    candle_cols = [col for col in df.columns if col.startswith("CDL")]
    for col in candle_cols: df[col] = df[col].fillna(0).apply(lambda v: 1.0 if v >= 80 else 0.5 if v > 0 else -1.0 if v <= -80 else -0.5 if v < 0 else 0.0)
    non_scalable = set(candle_cols + [c for c in df.columns if any(s in c for s in ['session_', 'trend_regime_', 'vol_regime_'])] + ['time'])
    to_scale = [c for c in df.columns if c not in non_scalable and df[c].dtype in ['float32', 'float64', 'int32', 'int64']]
    if scaler is None:
        scaler = StandardScaler(); df[to_scale] = scaler.fit_transform(df[to_scale].fillna(0)); return df, scaler
    else:
        df[to_scale] = scaler.transform(df[to_scale].fillna(0)); return df, scaler
def get_dynamic_price(entry_price, level_price, placement_bin):
    placement_ratio = (placement_bin + 0.5) / 10.0
    return entry_price + (level_price - entry_price) * placement_ratio
def simulate_trades(strategy, entries, full_silver_data):
    trades, capital = [], INITIAL_CAPITAL
    time_to_idx = {time: i for i, time in enumerate(full_silver_data['time'])}
    is_sl_dynamic = isinstance(strategy['sl_def'], str)
    is_tp_dynamic = isinstance(strategy['tp_def'], str)
    for entry in entries.itertuples():
        entry_price = entry.close
        sl_price = get_dynamic_price(entry_price, getattr(entry, strategy['sl_def']), strategy['sl_bin']) if is_sl_dynamic else entry_price * (1 - strategy['sl_def'])
        tp_price = get_dynamic_price(entry_price, getattr(entry, strategy['tp_def']), strategy['tp_bin']) if is_tp_dynamic else entry_price * (1 + strategy['tp_def'])
        if sl_price >= entry_price or tp_price <= entry_price: continue
        position_size = (capital * RISK_PER_TRADE_PCT) / abs(entry_price - sl_price) if entry_price != sl_price else 0
        if position_size <= 0: continue
        entry_idx = time_to_idx.get(entry.time)
        if entry_idx is None: continue
        future_highs, future_lows = full_silver_data['high'].values[entry_idx + 1:], full_silver_data['low'].values[entry_idx + 1:]
        tp_hits, sl_hits = np.where(future_highs >= tp_price)[0], np.where(future_lows <= sl_price)[0]
        first_tp_idx = tp_hits[0] if len(tp_hits) > 0 else np.inf
        first_sl_idx = sl_hits[0] if len(sl_hits) > 0 else np.inf
        if first_tp_idx < first_sl_idx:
            pnl = (tp_price - entry_price) * position_size; capital += pnl; trades.append({'pnl': pnl, 'capital': capital})
        elif first_sl_idx < first_tp_idx:
            pnl = (sl_price - entry_price) * position_size; capital += pnl; trades.append({'pnl': pnl, 'capital': capital})
    if not trades: return None
    log = pd.DataFrame(trades); wins, losses = log[log['pnl'] > 0], log[log['pnl'] <= 0]
    pf = wins['pnl'].sum() / abs(losses['pnl'].sum()) if abs(losses['pnl'].sum()) > 0 else np.inf
    log['peak'] = log['capital'].cummax(); log['drawdown'] = log['peak'] - log['capital']; max_dd = log['drawdown'].max()
    return {'profit_factor': pf, 'max_drawdown_pct': (max_dd / log['peak'].max()) * 100 if log['peak'].max() > 0 else 0, 'final_capital': capital, 'total_trades': len(log), 'win_rate_pct': (len(wins) / len(log)) * 100}
def run_backtest_for_strategy(strategy_dict, market_data_cache, markets_to_test):
    strategy = strategy_dict
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
    # ... (Setup and interactive prompts are correct) ...
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
    def to_numeric_or_str(x):
        try: return float(x)
        except (ValueError, TypeError): return x
    strategies_df_full = pd.read_csv(discovered_path, dtype={'sl_def': object, 'tp_def': object})
    for col in ['sl_def', 'tp_def']:
        if col in strategies_df_full.columns: strategies_df_full[col] = strategies_df_full[col].apply(to_numeric_or_str)
    total_strategies = len(strategies_df_full)
    print(f"\nFound {total_strategies} total strategies in the selected file.")
    try:
        top_n_input = input(f"How many top strategies per rule to test? (Default: {TOP_N_PER_RULE}): ").strip(); TOP_N_PER_RULE = int(top_n_input) if top_n_input else TOP_N_PER_RULE
    except ValueError: print(f"Invalid number. Defaulting to {TOP_N_PER_RULE}.");
    strategies_df = strategies_df_full.groupby('market_rule').head(TOP_N_PER_RULE).reset_index(drop=True)
    print(f"\nSelected {len(strategies_df)} strategies for backtesting ({TOP_N_PER_RULE} best variations per unique market rule).")
    print("\nLoading all prepared market data into memory...")
    market_data_cache = {}
    for market_csv in tqdm(markets_to_test, desc="Loading Data"):
        market_name = market_csv.replace('.csv', ''); silver_path = os.path.join(prepared_data_dir, f"{market_name}_silver.parquet"); gold_path = os.path.join(prepared_data_dir, f"{market_name}_gold.parquet")
        market_data_cache[market_csv] = {'silver': pd.read_parquet(silver_path), 'gold': pd.read_parquet(gold_path)}
    print("✅ All data loaded.")
    use_multiprocessing = input("Use multiprocessing? (y/n): ").strip().lower() == 'y'
    if use_multiprocessing: num_processes = MAX_CPU_USAGE
    else:
        try:
            num_processes = int(input(f"Enter number of processes (1-{cpu_count()}): ").strip())
            if num_processes < 1 or num_processes > cpu_count(): raise ValueError
        except ValueError: print("Invalid input. Defaulting to 1 process."); num_processes = 1
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
        
        # --- THE FIX IS HERE ---
        # 1. Create the aggregated summary
        summary_df = detailed_df.groupby('strategy_id').agg(
            markets_tested=('market', 'count'),
            avg_profit_factor=('profit_factor', 'mean'),
            avg_max_drawdown_pct=('max_drawdown_pct', 'mean'),
            total_trades=('total_trades', 'sum') # Sum trades across all markets
        ).reset_index()

        # 2. Get the unique strategy definitions from the detailed report itself
        strategy_definition_cols = [col for col in strategies_df.columns if col in detailed_df.columns]
        strategy_defs = detailed_df[['strategy_id'] + strategy_definition_cols].drop_duplicates(subset='strategy_id')
        
        # 3. Merge them back together
        summary_df = pd.merge(summary_df, strategy_defs, on='strategy_id').sort_values(by='avg_profit_factor', ascending=False)
        
        # 4. Calculate passing markets and blacklist
        passed_strategies = detailed_df[(detailed_df.profit_factor > PROFIT_FACTOR_PASS_THRESHOLD) & (detailed_df.max_drawdown_pct < MAX_DRAWDOWN_PASS_THRESHOLD)].groupby('strategy_id')['market'].count()
        summary_df['markets_passed_count'] = summary_df['strategy_id'].map(passed_strategies).fillna(0).astype(int)
        summary_df['markets_passed'] = summary_df['markets_passed_count'].astype(str) + '/' + summary_df['markets_tested'].astype(str)
        
        failed_strategy_ids = summary_df[summary_df['markets_passed_count'] < summary_df['markets_tested']]['strategy_id'].unique()
        # Use the same strategy_defs to create the blacklist
        blacklist_df = strategy_defs[strategy_defs['strategy_id'].isin(failed_strategy_ids)].drop(columns='strategy_id')

        # Save reports
        summary_df.to_csv(os.path.join(backtest_results_dir, f"summary_report_{strategy_file_to_test}"), index=False)
        detailed_df.to_csv(os.path.join(backtest_results_dir, f"detailed_report_{strategy_file_to_test}"), index=False)
        print(f"\n✅ Summary and Detailed reports saved.")
        if not blacklist_df.empty:
            blacklist_df.to_csv(os.path.join(blacklist_dir, strategy_file_to_test), index=False)
            print(f"✅ Added {len(blacklist_df)} underperforming strategies to blacklist.")
    
    print("\n" + "="*50 + "\n✅ All backtesting complete.")
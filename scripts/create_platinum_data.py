import os
import pandas as pd
import ta
from backtesting import Backtest, Strategy
import ast
from tqdm import tqdm

# --- CONFIGURATION ---

# --- CHOOSE THE FILES TO PROCESS ---
# The GOLD file containing the strategy candidates you want to validate.
GOLD_DATA_FILE = 'EURUSD1.csv' # e.g., The strategies discovered on EURUSD 15-min data

# --- ROBUSTNESS & RANKING ---
# A strategy must be profitable (Return > 0%) on at least this many test files.
# For the ultimate test, set this to the total number of files in your test_data folder.
MIN_PROFITABLE_TESTS = 3
# Which backtest metric should be used to rank the final robust strategies?
RANKING_METRIC = 'Sharpe Ratio'
# A strategy must have at least this many trades on average across all tests.
MIN_AVG_TRADES = 10

# --- BACKTEST SETTINGS ---
BROKERAGE_COMMISSION = 0.0002
INITIAL_CASH = 10000

# --- HELPER FUNCTIONS ---
def get_market_session(hour):
    if (hour >= 7 and hour < 12): return 'London'
    if (hour >= 12 and hour < 16): return 'London_NY_Overlap'
    if (hour >= 16 and hour < 21): return 'New_York'
    if (hour >= 21 or hour < 7): return 'Asian'
    return 'Off-Session'

_data_cache = {} # Use a dictionary to cache multiple data files
def get_prepared_data(data_path):
    """ Loads and prepares a test data file, caching the result for speed. """
    global _data_cache
    if data_path in _data_cache:
        return _data_cache[data_path]

    print(f"  > Preparing and caching new test data from: {os.path.basename(data_path)}")
    df = pd.read_csv(data_path, sep=None, engine="python")
    if df.shape[1] > 5: df = df.iloc[:, :5]
    df.columns = ["time", "open", "high", "low", "close"]
    
    # (The data preparation logic is identical to the previous script)
    df.rename(columns={"time": "Time", "open": "Open", "high": "High", "low": "Low", "close": "Close"}, inplace=True)
    df['Time'] = pd.to_datetime(df['Time']); df.set_index('Time', inplace=True)
    for p in [20, 50, 100, 200]: df[f'SMA_{p}'] = ta.trend.SMAIndicator(df['Close'], window=p).sma_indicator()
    for p in [8, 13, 21, 50]: df[f'EMA_{p}'] = ta.trend.EMAIndicator(df['Close'], window=p).ema_indicator()
    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2.0)
    df['BB_lower'] = bb.bollinger_lband(); df['BB_width'] = bb.bollinger_wband()
    df['RSI_14'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14).adx()
    df['ATR_14'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
    df['MACD_hist'] = ta.trend.MACD(df['Close']).macd_diff()
    df['support'] = df['Low'].rolling(200, min_periods=1).min()
    df['resistance'] = df['High'].rolling(200, min_periods=1).max()
    for p in [20, 50, 100, 200]: df[f'price_vs_sma{p}_ratio'] = (df['Close'] - df[f'SMA_{p}']) / df[f'SMA_{p}']
    for p in [8, 13, 21, 50]: df[f'price_vs_ema{p}_ratio'] = (df['Close'] - df[f'EMA_{p}']) / df[f'EMA_{p}']
    df['ema_short_vs_long_ratio'] = (df['EMA_8'] - df['EMA_50']) / df['Close']
    df['bb_position'] = (df['Close'] - df['BB_lower']) / (df['BB_width'] + 1e-9)
    df['atr_ratio'] = df['ATR_14'] / df['Close']
    df['price_vs_support_ratio'] = (df['Close'] - df['support']) / df['Close']
    df['price_vs_resistance_ratio'] = (df['resistance'] - df['Close']) / df['Close']
    df['RSI_delta'] = df['RSI_14'].diff()
    df['MACD_hist_delta'] = df['MACD_hist'].diff()
    is_bullish = (df['Close'] > df['Open']).astype(int)
    for p in [5, 10, 20, 50]: df[f'bullish_ratio_last_{p}'] = is_bullish.rolling(window=p).mean()
    N_QUANTILES = 5; quantile_labels = [f'q_{i+1}' for i in range(N_QUANTILES)]
    cols_to_bin = [col for col in df.columns if 'ratio' in col or 'delta' in col or 'position' in col or col in ['RSI_14', 'ADX']]
    cols_to_bin += [col for col in df.columns if col.startswith('bullish_ratio_last_')]
    for col in cols_to_bin:
        try: df[f'{col}_qbin'] = pd.qcut(df[col], N_QUANTILES, labels=quantile_labels, duplicates='drop')
        except: continue
    df['session'] = df.index.hour.map(get_market_session)
    df['trade_type'] = 'N/A'; df['candlestick_type'] = 'None'
    df.dropna(inplace=True)
    _data_cache[data_path] = df
    return df

# --- UNIVERSAL STRATEGY CLASS (Unchanged) ---
class UniversalStrategy(Strategy):
    rules, tp_ratio, sl_ratio, trade_direction = None, None, None, None
    def init(self): pass
    def next(self):
        if self.position: return
        all_conditions_met = True
        for feature, required_value in self.rules.items():
            current_value = self.data[feature][-1]
            if current_value != required_value: all_conditions_met = False; break
        if all_conditions_met:
            tp = self.data.Close[-1] * (1 + self.tp_ratio); sl = self.data.Close[-1] * (1 - self.sl_ratio)
            if self.trade_direction == 'buy': self.buy(sl=sl, tp=tp)
            elif self.trade_direction == 'sell': self.sell(sl=sl, tp=tp)

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    core_dir = os.path.dirname(os.path.abspath(__file__))
    gold_data_dir = os.path.abspath(os.path.join(core_dir, '..', 'gold_data'))
    test_data_dir = os.path.abspath(os.path.join(core_dir, '..', 'test_data'))
    platinum_data_dir = os.path.abspath(os.path.join(core_dir, '..', 'platinum_data'))
    os.makedirs(platinum_data_dir, exist_ok=True)

    # --- Load Strategy Candidates ---
    gold_file_path = os.path.join(gold_data_dir, GOLD_DATA_FILE)
    if not os.path.exists(gold_file_path):
        print(f"❌ FATAL: Gold file not found: '{gold_file_path}'"); exit()
    gold_df = pd.read_csv(gold_file_path)
    print(f"Loaded {len(gold_df)} strategy candidates from {GOLD_DATA_FILE}.")

    # --- Load All Test Files ---
    try:
        test_files = [os.path.join(test_data_dir, f) for f in os.listdir(test_data_dir) if f.endswith('.csv')]
    except FileNotFoundError:
        print(f"❌ Error: The directory '{test_data_dir}' was not found."); test_files = []
    if len(test_files) < MIN_PROFITABLE_TESTS:
        print(f"❌ FATAL: Not enough test files ({len(test_files)}) to meet MIN_PROFITABLE_TESTS ({MIN_PROFITABLE_TESTS})."); exit()
    print(f"Found {len(test_files)} test files for robustness validation.")

    # --- Iterate, Backtest, and Validate All Strategies ---
    robust_strategies = []
    print(f"\n--- Starting robustness validation for {len(gold_df)} strategies... ---")
    
    for index, strategy_candidate in tqdm(gold_df.iterrows(), total=len(gold_df), desc="Validating Strategies"):
        all_backtests_for_strategy = []
        is_robust = True

        for test_file in test_files:
            data = get_prepared_data(test_file)
            rules = ast.literal_eval(strategy_candidate['strategy_definition'])
            trade_direction = rules.pop('trade_type', None)

            if any(rule not in data.columns for rule in rules.keys()): continue

            bt = Backtest(data, UniversalStrategy, cash=INITIAL_CASH, commission=BROKERAGE_COMMISSION, finalize_trades=True)
            
            try:
                td = trade_direction if trade_direction else ('buy' if 'buy' in strategy_candidate['strategy_definition'] else 'sell')
                stats = bt.run(rules=rules, tp_ratio=strategy_candidate['suggested_tp_ratio'], sl_ratio=strategy_candidate['suggested_sl_ratio'], trade_direction=td)
                all_backtests_for_strategy.append(stats)
            except Exception:
                is_robust = False; break
        
        # --- The Robustness Check ---
        if not is_robust or len(all_backtests_for_strategy) < len(test_files):
            continue # Strategy failed to run on one or more test files

        num_profitable = sum(1 for stats in all_backtests_for_strategy if stats['Return [%]'] > 0)
        avg_trades = sum(stats['# Trades'] for stats in all_backtests_for_strategy) / len(all_backtests_for_strategy)

        if num_profitable >= MIN_PROFITABLE_TESTS and avg_trades >= MIN_AVG_TRADES:
            # This strategy is robust! Now, aggregate its performance.
            agg_results = pd.DataFrame(all_backtests_for_strategy)
            avg_stats = agg_results.mean(numeric_only=True)
            
            robust_info = strategy_candidate.copy()
            for metric, value in avg_stats.items():
                robust_info[f"Avg. {metric}"] = value
            robust_strategies.append(robust_info)

    if not robust_strategies:
        print("\n❌ No robust strategies found that were profitable across the specified number of test files."); exit()

    # --- Rank and Save Platinum Data ---
    print(f"\n--- Found {len(robust_strategies)} robust strategies. Ranking... ---")
    platinum_df = pd.DataFrame(robust_strategies)
    
    # Use the averaged metric for ranking
    platinum_df.sort_values(by=f"Avg. {RANKING_METRIC}", ascending=False, inplace=True)
    
    output_filename = os.path.join(platinum_data_dir, f"platinum_{os.path.splitext(GOLD_DATA_FILE)[0]}.csv")
    platinum_df.to_csv(output_filename, index=False)

    print(f"\n✅ Success! Saved {len(platinum_df)} validated and robust strategies to:\n{output_filename}")
    print("\n" + "="*50 + "\nPlatinum data generation complete.")
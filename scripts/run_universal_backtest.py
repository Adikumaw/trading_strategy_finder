import os
import pandas as pd
import ta
from backtesting import Backtest, Strategy
import ast

# --- CONFIGURATION ---
GOLD_DATA_FILE = 'EURUSD1.csv'
STRATEGY_ID_TO_TEST = 0
BROKERAGE_COMMISSION = 0.0002
INITIAL_CASH = 10000

# --- HELPER FUNCTION ---
def get_market_session(hour):
    if (hour >= 7 and hour < 12): return 'London'
    if (hour >= 12 and hour < 16): return 'London_NY_Overlap'
    if (hour >= 16 and hour < 21): return 'New_York'
    if (hour >= 21 or hour < 7): return 'Asian'
    return 'Off-Session'

def prepare_data(df):
    print("  Preparing data: Calculating all possible indicators and features...")
    df.rename(columns={"time": "Time", "open": "Open", "high": "High", "low": "Low", "close": "Close"}, inplace=True)
    df['Time'] = pd.to_datetime(df['Time'])
    df.set_index('Time', inplace=True)

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

    N_QUANTILES = 5
    quantile_labels = [f'q_{i+1}' for i in range(N_QUANTILES)]
    cols_to_bin = [col for col in df.columns if 'ratio' in col or 'delta' in col or 'position' in col or col in ['RSI_14', 'ADX']]
    cols_to_bin += [col for col in df.columns if col.startswith('bullish_ratio_last_')]
    for col in cols_to_bin:
        try:
            df[f'{col}_qbin'] = pd.qcut(df[col], N_QUANTILES, labels=quantile_labels, duplicates='drop')
        except: continue
            
    df['session'] = df.index.hour.map(get_market_session)
    df['trade_type'] = 'N/A'
    df['candlestick_type'] = 'None'
    
    df.dropna(inplace=True)
    return df

# --- UNIVERSAL STRATEGY CLASS ---
class UniversalStrategy(Strategy):
    rules, tp_ratio, sl_ratio, trade_direction = None, None, None, None
    def init(self):
        print("\n--- Initializing Strategy ---")
        print(f"Rules: {self.rules}"); print(f"TP Ratio: {self.tp_ratio}, SL Ratio: {self.sl_ratio}")
        print(f"Direction: {self.trade_direction}"); print("----------------------------\n")
    def next(self):
        if self.position: return
        all_conditions_met = True
        for feature, required_value in self.rules.items():
            current_value = self.data[feature][-1]
            if current_value != required_value:
                all_conditions_met = False; break
        if all_conditions_met:
            tp = self.data.Close[-1] * (1 + self.tp_ratio); sl = self.data.Close[-1] * (1 - self.sl_ratio)
            if self.trade_direction == 'buy': self.buy(sl=sl, tp=tp)
            elif self.trade_direction == 'sell': self.sell(sl=sl, tp=tp)

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    core_dir = os.path.dirname(os.path.abspath(__file__))
    gold_data_dir = os.path.abspath(os.path.join(core_dir, '..', 'gold_data'))
    test_data_dir = os.path.abspath(os.path.join(core_dir, '..', 'test_data'))
    reports_dir = os.path.abspath(os.path.join(core_dir, '..', 'reports'))
    os.makedirs(test_data_dir, exist_ok=True); os.makedirs(reports_dir, exist_ok=True)

    gold_file_path = os.path.join(gold_data_dir, GOLD_DATA_FILE)
    if not os.path.exists(gold_file_path):
        print(f"❌ FATAL: Gold data file not found at '{gold_file_path}'"); exit()
        
    gold_df = pd.read_csv(gold_file_path)
    if STRATEGY_ID_TO_TEST >= len(gold_df):
        print(f"❌ FATAL: Strategy ID {STRATEGY_ID_TO_TEST} out of bounds. File has {len(gold_df)} strategies."); exit()

    strategy_to_test = gold_df.iloc[STRATEGY_ID_TO_TEST]
    rules = ast.literal_eval(strategy_to_test['strategy_definition'])
    trade_direction = rules.pop('trade_type', 'buy')
    
    try:
        test_files = [f for f in os.listdir(test_data_dir) if f.endswith('.csv')]
    except FileNotFoundError:
        print(f"❌ Error: The directory '{test_data_dir}' was not found."); test_files = []

    if not test_files:
        print("❌ No CSV files found in 'test_data' folder.")
    else:
        print(f"Found {len(test_files)} files to backtest...")

    for filename in test_files:
        print("\n" + "="*50 + f"\nBacktesting Strategy #{STRATEGY_ID_TO_TEST} from {GOLD_DATA_FILE} on: {filename}")
        input_path = os.path.join(test_data_dir, filename)
        
        df = pd.read_csv(input_path, sep=None, engine="python")
        if df.shape[1] > 5: df = df.iloc[:, :5]
        df.columns = ["time", "open", "high", "low", "close"]

        data = prepare_data(df)
        if data.empty:
            print("  ⚠️ Not enough data to run backtest after preparing indicators. Skipping."); continue

        # --- FIX: Validate that all strategy rules can be checked ---
        missing_features = [rule for rule in rules.keys() if rule not in data.columns]
        if missing_features:
            print(f"  ❌ SKIPPING STRATEGY: The following required features were not found in the prepared data: {missing_features}")
            continue
        # --- END FIX ---
            
        bt = Backtest(data, UniversalStrategy, cash=INITIAL_CASH, commission=BROKERAGE_COMMISSION, trade_on_close=True)
        stats = bt.run(
            rules=rules, tp_ratio=strategy_to_test['suggested_tp_ratio'],
            sl_ratio=strategy_to_test['suggested_sl_ratio'], trade_direction=trade_direction
        )
        
        print("\n--- Backtest Results ---"); print(stats)
        
        report_filename = os.path.join(reports_dir, f"report_strat{STRATEGY_ID_TO_TEST}_{os.path.splitext(filename)[0]}.html")
        bt.plot(filename=report_filename, open_browser=False)
        print(f"\n✅ Detailed interactive report saved to: {report_filename}")

    print("\n" + "="*50 + "\nAll backtests complete.")
import os
import pandas as pd
import ta
from backtesting import Backtest, Strategy

# --- CONFIGURATION ---

# 1. Strategy Parameters (from your gold_data analysis)
TAKE_PROFIT_RATIO = 0.013  # 1.3%
STOP_LOSS_RATIO = 0.012   # 1.2%

# 2. Indicator and Filter Settings
RSI_PERIOD = 14
ADX_PERIOD = 14
SMA_PERIOD = 200
PAST_MOMENTUM_WINDOW = 10
RSI_NEUTRAL_LOWER = 30
RSI_NEUTRAL_UPPER = 70
ADX_STRONG_TREND = 50
MOMENTUM_MIXED_LOWER = 0.3
MOMENTUM_MIXED_UPPER = 0.7

# 3. Backtest Settings
BROKERAGE_COMMISSION = 0.0002 # 0.02%
INITIAL_CASH = 10000

# --- HELPER FUNCTION TO PREPARE DATA ---

def get_market_session(hour):
    """Determines the market session based on the hour of the day (UTC)."""
    if (hour >= 7 and hour < 12): return 'London'
    if (hour >= 12 and hour < 16): return 'London_NY_Overlap'
    if (hour >= 16 and hour < 21): return 'New_York'
    if (hour >= 21 or hour < 7): return 'Asian'
    return 'Off-Session'

def prepare_data(df):
    """Calculates all necessary indicators and features for the strategy."""
    print("  Preparing data: Calculating indicators and features...")
    # Rename columns to what backtesting.py expects
    df.rename(columns={
        "time": "Time", "open": "Open", "high": "High", "low": "Low", "close": "Close"
    }, inplace=True)
    df['Time'] = pd.to_datetime(df['Time'])
    df.set_index('Time', inplace=True)

    # Calculate Indicators
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=RSI_PERIOD).rsi()
    df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=ADX_PERIOD).adx()
    df['SMA200'] = ta.trend.SMAIndicator(df['Close'], window=SMA_PERIOD).sma_indicator()
    
    # Calculate Past Momentum
    is_bullish = (df['Close'] > df['Open']).astype(int)
    df['bullish_ratio_last_10'] = is_bullish.rolling(window=PAST_MOMENTUM_WINDOW).mean()

    # Calculate Session
    df['session'] = df.index.hour.map(get_market_session)
    
    df.dropna(inplace=True)
    return df

# --- THE STRATEGY CLASS ---

class GoldStrategy(Strategy):
    def init(self):
        pass

    def next(self):
        if self.position:
            return

        is_session = self.data.session[-1] == 'London_NY_Overlap'
        is_below_sma = self.data.Close[-1] < self.data.SMA200[-1]
        is_strong_trend = self.data.ADX[-1] > ADX_STRONG_TREND
        is_rsi_neutral = (self.data.RSI[-1] > RSI_NEUTRAL_LOWER and 
                          self.data.RSI[-1] < RSI_NEUTRAL_UPPER)
        is_momentum_mixed = (self.data.bullish_ratio_last_10[-1] > MOMENTUM_MIXED_LOWER and
                             self.data.bullish_ratio_last_10[-1] < MOMENTUM_MIXED_UPPER)

        if is_session and is_below_sma and is_strong_trend and is_rsi_neutral and is_momentum_mixed:
            tp_price = self.data.Close[-1] * (1 + TAKE_PROFIT_RATIO)
            sl_price = self.data.Close[-1] * (1 - STOP_LOSS_RATIO)
            self.buy(sl=sl_price, tp=tp_price)

# --- MAIN EXECUTION BLOCK ---

if __name__ == "__main__":
    core_dir = os.path.dirname(os.path.abspath(__file__))
    test_data_dir = os.path.abspath(os.path.join(core_dir, '..', 'test_data'))
    reports_dir = os.path.abspath(os.path.join(core_dir, '..', 'reports'))
    os.makedirs(test_data_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    try:
        test_files = [f for f in os.listdir(test_data_dir) if f.endswith('.csv')]
    except FileNotFoundError:
        print(f"❌ Error: The directory '{test_data_dir}' was not found. Please create it.")
        test_files = []

    if not test_files:
        print("❌ No CSV files found in 'test_data' folder. Please add a CSV file to test the strategy on.")
    else:
        print(f"Found {len(test_files)} files to backtest...")

    for filename in test_files:
        print("\n" + "="*50 + f"\nBacktesting on: {filename}")
        input_path = os.path.join(test_data_dir, filename)
        
        # 1. Load Data
        df = pd.read_csv(input_path, sep=None, engine="python")
        
        # --- FIX: Standardize column names immediately after loading ---
        if df.shape[1] > 5: 
            df = df.iloc[:, :5]
        df.columns = ["time", "open", "high", "low", "close"]
        # --- END FIX ---

        # 2. Prepare Data
        data = prepare_data(df)
        
        if data.empty:
            print("  ⚠️ Not enough data to run backtest after preparing indicators. Skipping.")
            continue

        # 3. Run Backtest
        print("  Starting backtest...")
        bt = Backtest(
            data,
            GoldStrategy,
            cash=INITIAL_CASH,
            commission=BROKERAGE_COMMISSION,
            trade_on_close=True
        )
        stats = bt.run()
        
        # 4. Print and Save Results
        print("\n--- Backtest Results ---")
        print(stats)
        
        report_filename = os.path.join(reports_dir, f"report_{os.path.splitext(filename)[0]}.html")
        bt.plot(filename=report_filename, open_browser=False)
        print(f"\n✅ Detailed interactive report saved to: {report_filename}")

    print("\n" + "="*50 + "\nAll backtests complete.")
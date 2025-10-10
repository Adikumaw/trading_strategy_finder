# diamond_data_prepper.py (Fully Automated)

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

# --- CONFIGURATION & COPIED FUNCTIONS ---
# (All helper functions and configurations are unchanged)
SMA_PERIODS, EMA_PERIODS = [20, 50, 100, 200], [8, 13, 21, 50]
# ... (All helper functions are identical to the previous version) ...
BBANDS_PERIOD, BBANDS_STD_DEV, RSI_PERIOD = 20, 2.0, 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
ATR_PERIOD, ADX_PERIOD, PIVOT_WINDOW = 14, 14, 10
INDICATOR_WARMUP_PERIOD = 200
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

if __name__ == "__main__":
    core_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.abspath(os.path.join(core_dir, '..', 'raw_data'))
    prepared_data_dir = os.path.abspath(os.path.join(core_dir, '..', 'diamond_data', 'prepared_data'))
    os.makedirs(prepared_data_dir, exist_ok=True)
    
    try:
        raw_files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
    except FileNotFoundError:
        print(f"❌ Raw data directory not found. Exiting."); exit()
    
    if not raw_files:
        print("❌ No raw data files found to process. Exiting."); exit()

    markets_to_process = []
    for f in raw_files:
        market_name = f.replace('.csv', '')
        output_silver_path = os.path.join(prepared_data_dir, f"{market_name}_silver.parquet")
        if not os.path.exists(output_silver_path):
            markets_to_process.append(f)
    
    if not markets_to_process:
        print("✅ All raw data is already prepared. Nothing to do."); exit()

    print(f"Found {len(markets_to_process)} new market(s) to prepare.")
    
    print("\nFitting scaler on the first market to ensure consistent scaling...")
    base_raw_df = robust_read_csv(os.path.join(raw_dir, markets_to_process[0]))
    base_silver_df = add_all_market_features(base_raw_df.copy()).iloc[INDICATOR_WARMUP_PERIOD:].reset_index(drop=True)
    _, scaler = create_gold_features(base_silver_df.copy())
    print("✅ Scaler fitted.")
    
    for market in tqdm(markets_to_process, desc="Preparing Market Data"):
        market_name = market.replace('.csv', '')
        output_silver_path = os.path.join(prepared_data_dir, f"{market_name}_silver.parquet")
        output_gold_path = os.path.join(prepared_data_dir, f"{market_name}_gold.parquet")
            
        print(f"\nProcessing {market}...")
        raw_df = robust_read_csv(os.path.join(raw_dir, market))
        silver_df = add_all_market_features(raw_df.copy()).iloc[INDICATOR_WARMUP_PERIOD:].reset_index(drop=True)
        gold_df, _ = create_gold_features(silver_df.copy(), scaler=scaler)
        
        silver_df.to_parquet(output_silver_path)
        gold_df.to_parquet(output_gold_path)
        print(f"✅ Prepared data for {market} saved.")
        
    print("\n" + "="*50 + "\n✅ All market data preparation complete.")
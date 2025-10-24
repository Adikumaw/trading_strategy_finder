# diamond_data_prepper.py (Fully Automated)

"""
Diamond Layer - Stage 0: The Data Prepper

This script is a critical, one-time utility that must be run before any
backtesting can occur in the Diamond or Zircon layers. Its purpose is to
"prepare once, test many."

It automates the entire feature generation pipeline (Silver -> Gold) for all
available raw market data files. The output is a cache of analysis-ready
market data saved in the efficient Parquet format. This pre-computation makes
the subsequent backtesting stages incredibly fast because they no longer need
to calculate indicators and features on the fly.

A key feature of this prepper is its use of a **consistent scaler**. It fits a
`StandardScaler` on the first market's data and then uses that *same fitted
scaler* to `transform` all other markets. This ensures that a feature value
(e.g., RSI=70) is represented by the same scaled number across all datasets,
making the discovered rules more robust and comparable across different
instruments.
"""

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
import sys # <-- IMPORT SYS MODULE

# --- CONFIGURATION & COPIED FUNCTIONS ---
# These configurations and helper functions are identical to those in the
# Silver and Gold layers, ensuring that the feature generation process is
# perfectly replicated for the backtesting environment.

SMA_PERIODS, EMA_PERIODS = [20, 50, 100, 200], [8, 13, 21, 50]
BBANDS_PERIOD, BBANDS_STD_DEV, RSI_PERIOD = 20, 2.0, 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
ATR_PERIOD, ADX_PERIOD, PIVOT_WINDOW = 14, 14, 10
INDICATOR_WARMUP_PERIOD = 200

def robust_read_csv(filepath):
    """
    Reads a raw OHLC CSV file reliably, handling common formatting inconsistencies.

    This function is designed to be resilient to different CSV delimiters and
    column counts. It automatically assigns standard column names, adds a dummy
    volume column if one is not present, converts columns to their proper
    numeric and datetime formats, and sorts the data by time.

    Args:
        filepath (str): The full path to the raw input CSV file.

    Returns:
        pd.DataFrame: A clean, sorted DataFrame with standardized column names.
    """
    df = pd.read_csv(filepath, header=None, sep=None, engine='python')
    if df.shape[1] > 5:
        df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
    elif df.shape[1] == 5:
        df.columns = ['time', 'open', 'high', 'low', 'close']
        df['volume'] = 1
    else:
        raise ValueError(f"File '{filepath}' does not have 5 or 6 columns.")
    df['time'] = pd.to_datetime(df['time'])
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric)
    return df.sort_values('time').reset_index(drop=True)

@numba.njit
def _calculate_s_r_numba(lows, highs, window):
    """
    Identifies fractal support and resistance points using a high-speed Numba kernel.

    This function iterates through price data and identifies local minima and maxima
    within a specified rolling window. A low is marked as a support point if it is
    the lowest value in the surrounding window, and a high is marked as a resistance
    point if it is the highest.

    Args:
        lows (np.array): A NumPy array of low prices.
        highs (np.array): A NumPy array of high prices.
        window (int): The number of candles to look forward and backward to define the local window.

    Returns:
        tuple: A tuple of two NumPy arrays (support_points, resistance_points).
    """
    n = len(lows)
    support, resistance = np.full(n, np.nan, dtype=np.float32), np.full(n, np.nan, dtype=np.float32)
    for i in range(window, n - window):
        if lows[i] == np.min(lows[i - window : i + window + 1]):
            support[i] = lows[i]
        if highs[i] == np.max(highs[i - window : i + window + 1]):
            resistance[i] = highs[i]
    return support, resistance

def add_support_resistance(df, window=PIVOT_WINDOW):
    """
    Calculates and adds forward-filled support and resistance levels to the DataFrame.

    This function acts as a wrapper for the core Numba S/R calculation. It calls
    the high-speed Numba function and then forward-fills the results to ensure
    every candle has a value for the "last known" support and resistance level.

    Args:
        df (pd.DataFrame): The input market data DataFrame.
        window (int): The window size for the fractal calculation.

    Returns:
        pd.DataFrame: The DataFrame with 'support' and 'resistance' columns added.
    """
    s_pts, r_pts = _calculate_s_r_numba(df["low"].values.astype(np.float32), df["high"].values.astype(np.float32), window)
    sr_df = pd.DataFrame({'s_pts': s_pts, 'r_pts': r_pts}, index=df.index)
    df["support"], df["resistance"] = sr_df["s_pts"].ffill(), sr_df["r_pts"].ffill()
    return df

def add_all_market_features(df):
    """
    Generates a comprehensive suite of Silver-layer market features for raw OHLC data.

    This function replicates the complete feature engineering process of the
    Silver layer. It calculates technical indicators, candlestick patterns, market
    regimes, and support/resistance levels to create a rich, contextual dataset
    for each candle.

    Args:
        df (pd.DataFrame): The raw OHLC DataFrame.

    Returns:
        pd.DataFrame: The original DataFrame enriched with over 200 feature columns.
    """
    indicator_df = pd.DataFrame(index=df.index)
    for p in SMA_PERIODS: indicator_df[f"SMA_{p}"] = ta.trend.SMAIndicator(df["close"], p).sma_indicator()
    for p in EMA_PERIODS: indicator_df[f"EMA_{p}"] = ta.trend.EMAIndicator(df["close"], p).ema_indicator()
    bb = ta.volatility.BollingerBands(df["close"], BBANDS_PERIOD, BBANDS_STD_DEV)
    indicator_df["BB_upper"], indicator_df["BB_lower"], indicator_df["BB_width"] = bb.bollinger_hband(), bb.bollinger_lband(), bb.bollinger_wband()
    indicator_df[f"RSI_{RSI_PERIOD}"] = ta.momentum.RSIIndicator(df["close"], RSI_PERIOD).rsi()
    indicator_df["MACD_hist"] = ta.trend.MACD(df["close"], MACD_SLOW, MACD_FAST, MACD_SIGNAL).macd_diff()
    indicator_df[f"ATR_{ATR_PERIOD}"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], ATR_PERIOD).average_true_range()
    indicator_df["ADX"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], ADX_PERIOD).adx()
    indicator_df["OBV"] = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
    atr = indicator_df[f"ATR_{ATR_PERIOD}"]
    indicator_df["ATR_level_up_1x"], indicator_df["ATR_level_down_1x"] = df["close"] + atr, df["close"] - atr
    price_action_df = pd.DataFrame(index=df.index)
    price_action_df['session'] = df['time'].dt.hour.map(lambda h: 'London' if 7<=h<12 else 'London_NY_Overlap' if 12<=h<16 else 'New_York' if 16<=h<21 else 'Asian')
    price_action_df['trend_regime'] = np.where(indicator_df['ADX'] > 25, 'trend', 'range')
    price_action_df['vol_regime'] = np.where(atr > atr.rolling(50).mean(), 'high_vol', 'low_vol')
    patterns_df = pd.DataFrame({p: getattr(talib, p)(df["open"], df["high"], df["low"], df["close"]) for p in talib.get_function_groups().get("Pattern Recognition", [])}, index=df.index)
    combined = pd.concat([df, indicator_df, price_action_df, patterns_df], axis=1)
    return add_support_resistance(combined)

def create_gold_features(features_df, scaler=None):
    """
    Transforms a Silver-layer DataFrame into a Gold-layer, ML-ready dataset.

    This function replicates the Gold-layer preprocessing pipeline. It normalizes
    absolute price levels, one-hot encodes categorical features, discretizes
    candlestick pattern scores, and scales all numeric features. Crucially, it
    can operate in two modes:
    1. If `scaler` is None, it will create and fit a new `StandardScaler`.
    2. If a pre-fitted `scaler` is provided, it will use that scaler to `transform`
       the data, ensuring consistency across different datasets.

    Args:
        features_df (pd.DataFrame): The Silver-layer DataFrame with all features.
        scaler (sklearn.preprocessing.StandardScaler, optional): A pre-fitted
            scaler object. If None, a new one will be fitted. Defaults to None.

    Returns:
        tuple: A tuple containing:
               - The transformed, ML-ready Gold DataFrame.
               - The scaler object used for the transformation (either the one
                 that was passed in or the newly fitted one).
    """
    df = features_df.copy()
    abs_price_cols = [col for col in df.columns if re.match(r'^(open|high|low|close|SMA_\d+|EMA_\d+|BB_(upper|lower)|support|resistance|ATR_level_.+)$', col)]
    for col in abs_price_cols:
        if col != 'close': df[f'{col}_dist_norm'] = (df['close'] - df[col]) / df['close']
    df.drop(columns=abs_price_cols + ['volume'], inplace=True, errors='ignore')
    df = pd.get_dummies(df, columns=['session', 'trend_regime', 'vol_regime'], drop_first=True)
    candle_cols = [col for col in df.columns if col.startswith("CDL")]
    for col in candle_cols:
        df[col] = df[col].fillna(0).apply(lambda v: 1.0 if v >= 80 else 0.5 if v > 0 else -1.0 if v <= -80 else -0.5 if v < 0 else 0.0)
    non_scalable = set(candle_cols + [c for c in df.columns if any(s in c for s in ['session_', 'trend_regime_', 'vol_regime_'])] + ['time'])
    to_scale = [c for c in df.columns if c not in non_scalable and df[c].dtype in ['float32', 'float64', 'int32', 'int64']]
    
    if scaler is None:
        # If no scaler is provided, create and fit a new one.
        scaler = StandardScaler()
        df[to_scale] = scaler.fit_transform(df[to_scale].fillna(0))
        return df, scaler
    else:
        # If a scaler is provided, use it to transform the data.
        df[to_scale] = scaler.transform(df[to_scale].fillna(0))
        return df, scaler

if __name__ == "__main__":
    """
    Main execution block.
    
    This script can be run in two modes:
    1. Discovery Mode (no arguments): Scans for all raw files and prepares data for any that are missing.
       Example: `python scripts/diamond_data_prepper.py`
       
    2. Targeted Mode (one argument): Identifies the timeframe from the target file and prepares
       data for ALL raw files that share that same timeframe.
       Example: `python scripts/diamond_data_prepper.py XAUUSD15.csv`
    """
    # --- Define Project Directory Structure ---
    core_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.abspath(os.path.join(core_dir, '..', 'raw_data'))
    prepared_data_dir = os.path.abspath(os.path.join(core_dir, '..', 'diamond_data', 'prepared_data'))
    os.makedirs(prepared_data_dir, exist_ok=True)
    
    # --- Discover Raw Data Files ---
    try:
        all_raw_files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
    except FileNotFoundError:
        print(f"❌ Raw data directory not found. Exiting."); sys.exit(1)
    
    if not all_raw_files:
        print("❌ No raw data files found to process. Exiting."); sys.exit(1)

    # --- NEW: DUAL-MODE FILE DISCOVERY LOGIC ---
    target_file_arg = sys.argv[1] if len(sys.argv) > 1 else None
    files_to_consider = []

    if target_file_arg:
        # --- Targeted Mode ---
        print(f"🎯 Targeted Mode: Preparing data for timeframe of '{target_file_arg}'")
        match = re.search(r'(\d+)\.csv$', target_file_arg)
        if not match:
            print(f"❌ Error: Could not extract timeframe from '{target_file_arg}'. Exiting.")
            sys.exit(1)
        
        target_timeframe_num = match.group(1)
        # Find all raw files that share this timeframe
        files_to_consider = [f for f in all_raw_files if f.endswith(f"{target_timeframe_num}.csv")]
        print(f"Found {len(files_to_consider)} files matching the '{target_timeframe_num}m' timeframe.")
    else:
        # --- Discovery Mode (Default) ---
        print("🔍 Discovery Mode: Scanning for all new raw files to prepare...")
        files_to_consider = all_raw_files

    # --- Identify Markets That Need Processing ---
    # This makes the script resumable by only processing new or missing files.
    markets_to_process = []
    for f in files_to_consider:
        market_name = f.replace('.csv', '')
        # Check if the FINAL output already exists. This makes the script resumable.
        output_gold_path = os.path.join(prepared_data_dir, f"{market_name}_gold.parquet")
        if not os.path.exists(output_gold_path):
            markets_to_process.append(f)
    
    if not markets_to_process:
        print("✅ All necessary market data is already prepared. Nothing to do.")
        sys.exit(0)

    print(f"Found {len(markets_to_process)} new market(s) to prepare.")
    
    # --- Fit a Consistent Scaler ---
    # The first market in the list is used as the "base" to fit the StandardScaler.
    # This ensures consistency across all datasets.
    print("\nFitting scaler on the first market to ensure consistent scaling...")
    base_raw_df = robust_read_csv(os.path.join(raw_dir, markets_to_process[0]))
    base_silver_df = add_all_market_features(base_raw_df.copy()).iloc[INDICATOR_WARMUP_PERIOD:].reset_index(drop=True)
    _, scaler = create_gold_features(base_silver_df.copy())
    print("✅ Scaler fitted.")
    
    # --- Main Processing Loop ---
    # Iterate through each new market and apply the full Silver -> Gold pipeline.
    for market in tqdm(markets_to_process, desc="Preparing Market Data"):
        market_name = market.replace('.csv', '')
        output_silver_path = os.path.join(prepared_data_dir, f"{market_name}_silver.parquet")
        output_gold_path = os.path.join(prepared_data_dir, f"{market_name}_gold.parquet")
            
        print(f"\nProcessing {market}...")
        try:
            # 1. Load raw data.
            raw_df = robust_read_csv(os.path.join(raw_dir, market))
            # 2. Generate Silver features.
            silver_df = add_all_market_features(raw_df.copy()).iloc[INDICATOR_WARMUP_PERIOD:].reset_index(drop=True)
            # 3. Generate Gold features using the pre-fitted scaler.
            gold_df, _ = create_gold_features(silver_df.copy(), scaler=scaler)
            
            # 4. Save the final data to efficient Parquet files.
            silver_df.to_parquet(output_silver_path)
            gold_df.to_parquet(output_gold_path)
            print(f"✅ Prepared data for {market} saved.")
        except Exception as e:
            print(f"❌ FAILED to process {market}. Error: {e}")
            # In a full pipeline, we might want to stop here.
            # For robustness, we'll just print the error and continue.
            
    print("\n" + "="*50 + "\n✅ All market data preparation complete.")
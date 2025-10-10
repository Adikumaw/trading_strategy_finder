# silver_data_generator_nolimits.py

import os
import gc
import pandas as pd
import ta
import talib
import numba
import numpy as np
from tqdm import tqdm

# --- CONFIGURATION (Unchanged) ---
SMA_PERIODS, EMA_PERIODS = [20, 50, 100, 200], [8, 13, 21, 50]
BBANDS_PERIOD, BBANDS_STD_DEV, RSI_PERIOD = 20, 2.0, 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
ATR_PERIOD, ADX_PERIOD, PIVOT_WINDOW, PAST_LOOKBACKS = 14, 14, 10, [3, 5, 10, 20, 50]
INDICATOR_WARMUP_PERIOD = 200

# --- UTILITY & FEATURE FUNCTIONS (Unchanged) ---
def downcast_dtypes(df):
    for col in df.select_dtypes(include=['float64']).columns: df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns: df[col] = df[col].astype('int32')
    return df
def robust_read_csv(filepath):
    df = pd.read_csv(filepath, sep=None, engine="python", header=None)
    if df.shape[1] > 5: df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
    elif df.shape[1] == 5: df.columns = ['time', 'open', 'high', 'low', 'close']; df['volume'] = 1
    else: raise ValueError(f"File '{filepath}' has fewer than 5 columns.")
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=['time'] + numeric_cols, inplace=True)
    return df.sort_values('time').reset_index(drop=True)
@numba.njit
def _calculate_s_r_numba(lows, highs, window):
    n = len(lows); support_points, resistance_points = np.full(n, np.nan, dtype=np.float32), np.full(n, np.nan, dtype=np.float32)
    for i in range(window, n - window):
        if lows[i] == np.min(lows[i - window : i + window + 1]): support_points[i] = lows[i]
        if highs[i] == np.max(highs[i - window : i + window + 1]): resistance_points[i] = highs[i]
    return support_points, resistance_points
def add_support_resistance(df, window=PIVOT_WINDOW):
    lows, highs = df["low"].values.astype(np.float32), df["high"].values.astype(np.float32)
    support_points, resistance_points = _calculate_s_r_numba(lows, highs, window); sr_df = pd.DataFrame({'support_points': support_points, 'resistance_points': resistance_points}, index=df.index)
    sr_df["support"], sr_df["resistance"] = sr_df["support_points"].ffill(), sr_df["resistance_points"].ffill()
    return pd.concat([df, sr_df[['support', 'resistance']]], axis=1)
def add_all_market_features(df):
    new_features_list = []
    print("Calculating standard indicators..."); indicator_df = pd.DataFrame(index=df.index)
    for p in SMA_PERIODS: indicator_df[f"SMA_{p}"] = ta.trend.SMAIndicator(df["close"], p).sma_indicator()
    for p in EMA_PERIODS: indicator_df[f"EMA_{p}"] = ta.trend.EMAIndicator(df["close"], p).ema_indicator()
    bb = ta.volatility.BollingerBands(df["close"], BBANDS_PERIOD, BBANDS_STD_DEV)
    indicator_df["BB_upper"], indicator_df["BB_lower"], indicator_df["BB_width"] = bb.bollinger_hband(), bb.bollinger_lband(), bb.bollinger_wband()
    indicator_df[f"RSI_{RSI_PERIOD}"] = ta.momentum.RSIIndicator(df["close"], RSI_PERIOD).rsi(); indicator_df["MACD_hist"] = ta.trend.MACD(df["close"], MACD_SLOW, MACD_FAST, MACD_SIGNAL).macd_diff()
    indicator_df[f"ATR_{ATR_PERIOD}"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], ATR_PERIOD).average_true_range(); indicator_df["ADX"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], ADX_PERIOD).adx()
    indicator_df["MOM_10"] = ta.momentum.ROCIndicator(df["close"], window=10).roc(); indicator_df["CCI_20"] = ta.trend.CCIIndicator(df["high"], df["low"], df["close"], window=20).cci()
    if "volume" in df.columns and df['volume'].nunique() > 1: indicator_df["OBV"] = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
    atr_series = indicator_df[f"ATR_{ATR_PERIOD}"]; indicator_df["ATR_level_up_1x"] = df["close"] + atr_series; indicator_df["ATR_level_down_1x"] = df["close"] - atr_series
    new_features_list.append(indicator_df)
    print("Calculating candlestick patterns..."); patterns_data = {p: getattr(talib, p)(df["open"], df["high"], df["low"], df["close"]) for p in talib.get_function_groups().get("Pattern Recognition", [])}
    new_features_list.append(pd.DataFrame(patterns_data, index=df.index))
    print("Calculating support and resistance..."); df_with_sr = add_support_resistance(df.copy()); new_features_list.append(df_with_sr[['support', 'resistance']])
    print("Calculating time-based and price-action features..."); price_action_df = pd.DataFrame(index=df.index)
    price_action_df['session'] = df['time'].dt.hour.map(lambda h: 'London' if 7<=h<12 else 'London_NY_Overlap' if 12<=h<16 else 'New_York' if 16<=h<21 else 'Asian')
    price_action_df['hour'], price_action_df['weekday'] = df['time'].dt.hour, df['time'].dt.weekday
    is_bullish, body_size = (df['close'] > df['open']).astype(int), (df['close'] - df['open']).abs()
    for n in PAST_LOOKBACKS:
        price_action_df[f'bullish_ratio_last_{n}'] = is_bullish.rolling(n).mean(); price_action_df[f'avg_body_last_{n}'] = body_size.rolling(n).mean()
        price_action_df[f'avg_range_last_{n}'] = (df['high'] - df['low']).rolling(n).mean()
        sma_col = f'SMA_{n}' if f'SMA_{n}' in indicator_df else 'SMA_20'; price_action_df[f'close_{sma_col}_ratio'] = df['close'] / indicator_df[sma_col]
    price_action_df['trend_regime'] = np.where(indicator_df['ADX'] > 25, 'trend', 'range')
    if f'ATR_{ATR_PERIOD}' in indicator_df: price_action_df['vol_regime'] = np.where(indicator_df[f'ATR_{ATR_PERIOD}'] > indicator_df[f'ATR_{ATR_PERIOD}'].rolling(50).mean(), 'high_vol', 'low_vol')
    new_features_list.append(price_action_df)
    return pd.concat([df] + new_features_list, axis=1)
def add_positioning_features(merged_df):
    level_cols = ['open', 'high', 'low', 'support', 'resistance', 'BB_upper', 'BB_lower', 'ATR_level_up_1x', 'ATR_level_down_1x']
    level_cols.extend([col for col in merged_df.columns if 'SMA_' in col or 'EMA_' in col])
    entry, sl, tp, close = merged_df['entry_price'], merged_df['sl_price'], merged_df['tp_price'], merged_df['close']
    for level_name in level_cols:
        if level_name in merged_df.columns and not merged_df[level_name].isnull().all():
            level_price = merged_df[level_name]
            merged_df[f'sl_dist_to_{level_name}_bps'] = ((sl - level_price) / close) * 10000
            merged_df[f'tp_dist_to_{level_name}_bps'] = ((tp - level_price) / close) * 10000
            total_dist_to_level = level_price - entry; sl_dist_from_entry = sl - entry; tp_dist_from_entry = tp - entry
            merged_df[f'sl_placement_pct_to_{level_name}'] = (sl_dist_from_entry / total_dist_to_level).replace([np.inf, -np.inf], np.nan)
            merged_df[f'tp_placement_pct_to_{level_name}'] = (tp_dist_from_entry / total_dist_to_level).replace([np.inf, -np.inf], np.nan)
    return merged_df

def create_enriched_silver_data_nolimits(bronze_path, raw_path, features_path, outcomes_path):
    print(f"\n{'='*25}\nProcessing: {os.path.basename(raw_path)}\n{'='*25}")
    
    # --- STEP 1: Create Silver Features (Unchanged) ---
    print("STEP 1: Creating Silver Features dataset...")
    raw_df = robust_read_csv(raw_path)
    if len(raw_df) < INDICATOR_WARMUP_PERIOD + 1:
        print(f"❌ SKIPPING: Not enough data for indicator warmup."); return
    features_df = add_all_market_features(raw_df); del raw_df; gc.collect()
    features_df = features_df.iloc[INDICATOR_WARMUP_PERIOD:].reset_index(drop=True)
    features_df = downcast_dtypes(features_df)
    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    features_df.to_csv(features_path, index=False)
    print(f"✅ Silver Features saved to: {features_path}")
    
    # --- STEP 2: SPEED OPTIMIZATION - Load all Bronze data at once ---
    print("\nSTEP 2: Creating ENRICHED Silver Outcomes dataset (in-memory)...")
    try:
        bronze_df = pd.read_csv(bronze_path, parse_dates=['entry_time'])
        print(f"Loaded {len(bronze_df)} trades from bronze data.")
    except FileNotFoundError:
        print(f"❌ Bronze file not found at {bronze_path}. Skipping outcomes generation."); return

    # Filter trades to align with the start of the feature data
    bronze_df = bronze_df[bronze_df['entry_time'] >= features_df['time'].min()].copy()
    if bronze_df.empty:
        print("ℹ️ No trades in the bronze file occur after the feature warmup period."); return
    
    # Merge the full bronze DataFrame with the necessary feature columns
    indicator_levels = ['support', 'resistance', 'BB_upper', 'BB_lower', 'ATR_level_up_1x', 'ATR_level_down_1x'] + [f"SMA_{p}" for p in SMA_PERIODS] + [f"EMA_{p}" for p in EMA_PERIODS]
    cols_to_keep = ['time', 'open', 'high', 'low', 'close'] + indicator_levels
    
    merged_df = pd.merge_asof(
        bronze_df.sort_values('entry_time'), 
        features_df[cols_to_keep], 
        left_on='entry_time', 
        right_on='time', 
        direction='backward'
    )
    del bronze_df, features_df; gc.collect()

    # --- Process the entire merged DataFrame in memory ---
    print("Enriching trades with positioning features...")
    enriched_df = add_positioning_features(merged_df)
    
    # Drop intermediate columns
    enriched_df.drop(columns=indicator_levels + ['time'], inplace=True, errors='ignore')
    
    # --- SPEED OPTIMIZATION: Write the final result in one go ---
    enriched_df = downcast_dtypes(enriched_df)
    enriched_df.to_csv(outcomes_path, index=False)
    print(f"✅ Enriched Silver Outcomes saved to: {outcomes_path}")

if __name__ == "__main__":
    core_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir, bronze_dir = [os.path.abspath(os.path.join(core_dir, '..', d)) for d in ['raw_data', 'bronze_data']]
    features_dir = os.path.abspath(os.path.join(core_dir, '..', 'silver_data', 'features'))
    outcomes_dir = os.path.abspath(os.path.join(core_dir, '..', 'silver_data', 'outcomes'))
    os.makedirs(features_dir, exist_ok=True); os.makedirs(outcomes_dir, exist_ok=True)

    bronze_files = [f for f in os.listdir(bronze_dir) if f.endswith('.csv')]
    if not bronze_files: print("❌ No bronze files found to process.")
    else:
        for fname in bronze_files:
            bronze_path, raw_path = [os.path.join(d, fname) for d in [bronze_dir, raw_dir]]
            features_path, outcomes_path = [os.path.join(d, fname) for d in [features_dir, outcomes_dir]]
            
            if not os.path.exists(bronze_path):
                print(f"⚠️ SKIPPING: Bronze file for {fname} not found."); continue
            if not os.path.exists(raw_path):
                print(f"⚠️ SKIPPING: Raw file for {fname} not found."); continue
            
            if os.path.exists(outcomes_path) and os.path.exists(features_path):
                print(f"✅ SKIPPING: Silver data already exists for {fname}."); continue
            
            try:
                create_enriched_silver_data_nolimits(bronze_path, raw_path, features_path, outcomes_path)
            except Exception as e:
                print(f"❌ FAILED to process {fname}. Error: {e}"); import traceback; traceback.print_exc()

    print("\n" + "="*50 + "\n✅ All silver data generation complete.")
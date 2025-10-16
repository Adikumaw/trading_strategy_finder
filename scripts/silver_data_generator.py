# silver_data_generator.py (Upgraded to create chunked_outcomes directly)

"""
Silver Layer: The Enrichment Engine

This script is the central feature engineering hub of the entire pipeline. It
takes the raw, high-volume trade simulations from the Bronze Layer and transforms
them into an intelligent, context-rich dataset ready for machine learning.

It operates in two distinct stages for each instrument:
1.  Market Feature Generation: It first consumes the raw OHLC price data and
    calculates a massive suite of over 200 technical indicators, candlestick
    patterns, and custom market context features for every single candle. This
    creates a complete, candle-by-candle "fingerprint" of the market's state.
2.  Trade Enrichment & Chunking: It then reads the enormous Bronze Dataset in
    memory-safe chunks. For each winning trade, it merges the pre-calculated
    market features and calculates a powerful set of "relational positioning"
    features, which describe where a trade's SL/TP were relative to market
    structures.
"""

import os
import gc
import pandas as pd
import ta
import talib
import numba
import numpy as np
from tqdm import tqdm

# --- CONFIGURATION ---
# Defines the periods for various technical indicators to be calculated.
SMA_PERIODS = [20, 50, 100, 200]
EMA_PERIODS = [8, 13, 21, 50]
BBANDS_PERIOD, BBANDS_STD_DEV = 20, 2.0
RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
ATR_PERIOD = 14
ADX_PERIOD = 14
PIVOT_WINDOW = 10  # Window for calculating fractal-based Support/Resistance
PAST_LOOKBACKS = [3, 5, 10, 20, 50] # Lookback periods for price-action features
# The number of initial candles to discard to allow indicators to generate stable values.
INDICATOR_WARMUP_PERIOD = 200

# --- UTILITY & FEATURE FUNCTIONS ---

def downcast_dtypes(df):
    """
    Reduces the memory footprint of a DataFrame by downcasting numeric types
    to more efficient formats (e.g., float64 to float32).
    """
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
    return df

def robust_read_csv(filepath):
    """
    Reads a raw OHLC CSV file reliably, handling potential delimiter issues
    and varying column counts. Ensures data is clean and sorted by time.
    """
    df = pd.read_csv(filepath, sep=None, engine="python", header=None)
    if df.shape[1] > 5:
        df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
    elif df.shape[1] == 5:
        df.columns = ['time', 'open', 'high', 'low', 'close']
        df['volume'] = 1 # Add a dummy volume column if not present
    else:
        raise ValueError(f"File '{filepath}' has fewer than 5 columns.")
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=['time'] + numeric_cols, inplace=True)
    return df.sort_values('time').reset_index(drop=True)

@numba.njit
def _calculate_s_r_numba(lows, highs, window):
    """
    A high-performance Numba function to identify fractal-based support and
    resistance points. A point is support if it's the lowest low in its
    surrounding window, and resistance if it's the highest high.
    """
    n = len(lows)
    support_points, resistance_points = np.full(n, np.nan, dtype=np.float32), np.full(n, np.nan, dtype=np.float32)
    for i in range(window, n - window):
        window_lows = lows[i - window : i + window + 1]
        window_highs = highs[i - window : i + window + 1]
        if lows[i] == np.min(window_lows):
            support_points[i] = lows[i]
        if highs[i] == np.max(window_highs):
            resistance_points[i] = highs[i]
    return support_points, resistance_points

def add_support_resistance(df, window=PIVOT_WINDOW):
    """
    Wrapper function to calculate and forward-fill support/resistance levels.
    """
    lows, highs = df["low"].values.astype(np.float32), df["high"].values.astype(np.float32)
    support_points, resistance_points = _calculate_s_r_numba(lows, highs, window)
    sr_df = pd.DataFrame({'support_points': support_points, 'resistance_points': resistance_points}, index=df.index)
    # Forward-fill to carry the last known S/R level forward in time
    sr_df["support"], sr_df["resistance"] = sr_df["support_points"].ffill(), sr_df["resistance_points"].ffill()
    return pd.concat([df, sr_df[['support', 'resistance']]], axis=1)

def add_all_market_features(df):
    """
    Calculates a comprehensive suite of market features for each candle
    in the input DataFrame, organized into efficient batches.
    """
    new_features_list = []
    
    # --- Batch 1: Standard Technical Indicators ---
    print("Calculating standard indicators...")
    indicator_df = pd.DataFrame(index=df.index)
    for p in SMA_PERIODS: indicator_df[f"SMA_{p}"] = ta.trend.SMAIndicator(df["close"], p).sma_indicator()
    for p in EMA_PERIODS: indicator_df[f"EMA_{p}"] = ta.trend.EMAIndicator(df["close"], p).ema_indicator()
    bb = ta.volatility.BollingerBands(df["close"], BBANDS_PERIOD, BBANDS_STD_DEV)
    indicator_df["BB_upper"], indicator_df["BB_lower"], indicator_df["BB_width"] = bb.bollinger_hband(), bb.bollinger_lband(), bb.bollinger_wband()
    indicator_df[f"RSI_{RSI_PERIOD}"] = ta.momentum.RSIIndicator(df["close"], RSI_PERIOD).rsi()
    indicator_df["MACD_hist"] = ta.trend.MACD(df["close"], MACD_SLOW, MACD_FAST, MACD_SIGNAL).macd_diff()
    indicator_df[f"ATR_{ATR_PERIOD}"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], ATR_PERIOD).average_true_range()
    indicator_df["ADX"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], ADX_PERIOD).adx()
    indicator_df["MOM_10"] = ta.momentum.ROCIndicator(df["close"], window=10).roc()
    indicator_df["CCI_20"] = ta.trend.CCIIndicator(df["high"], df["low"], df["close"], window=20).cci()
    if "volume" in df.columns and df['volume'].nunique() > 1:
        indicator_df["OBV"] = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
    
    # Add ATR-based dynamic price levels to the indicator batch
    atr_series = indicator_df[f"ATR_{ATR_PERIOD}"]
    indicator_df["ATR_level_up_1x"] = df["close"] + atr_series
    indicator_df["ATR_level_down_1x"] = df["close"] - atr_series
    new_features_list.append(indicator_df)

    # --- Batch 2: Candlestick Patterns ---
    print("Calculating candlestick patterns...")
    # Use TA-Lib to efficiently generate scores for all classic candlestick patterns
    patterns_data = {p: getattr(talib, p)(df["open"], df["high"], df["low"], df["close"]) for p in talib.get_function_groups().get("Pattern Recognition", [])}
    new_features_list.append(pd.DataFrame(patterns_data, index=df.index))
    
    # --- Batch 3: Support & Resistance ---
    print("Calculating support and resistance...")
    df_with_sr = add_support_resistance(df.copy())
    new_features_list.append(df_with_sr[['support', 'resistance']])

    # --- Batch 4: Time-based and Price-Action Features ---
    print("Calculating time-based and price-action features...")
    price_action_df = pd.DataFrame(index=df.index)
    # Market session based on UTC hour
    price_action_df['session'] = df['time'].dt.hour.map(lambda h: 'London' if 7<=h<12 else 'London_NY_Overlap' if 12<=h<16 else 'New_York' if 16<=h<21 else 'Asian')
    price_action_df['hour'], price_action_df['weekday'] = df['time'].dt.hour, df['time'].dt.weekday
    is_bullish, body_size = (df['close'] > df['open']).astype(int), (df['close'] - df['open']).abs()
    for n in PAST_LOOKBACKS:
        price_action_df[f'bullish_ratio_last_{n}'] = is_bullish.rolling(n).mean()
        price_action_df[f'avg_body_last_{n}'] = body_size.rolling(n).mean()
        price_action_df[f'avg_range_last_{n}'] = (df['high'] - df['low']).rolling(n).mean()
        sma_col = f'SMA_{n}' if f'SMA_{n}' in indicator_df else 'SMA_20'
        price_action_df[f'close_{sma_col}_ratio'] = df['close'] / indicator_df[sma_col]
    # Market regime classification
    price_action_df['trend_regime'] = np.where(indicator_df['ADX'] > 25, 'trend', 'range')
    if f'ATR_{ATR_PERIOD}' in indicator_df:
        price_action_df['vol_regime'] = np.where(indicator_df[f'ATR_{ATR_PERIOD}'] > indicator_df[f'ATR_{ATR_PERIOD}'].rolling(50).mean(), 'high_vol', 'low_vol')
    new_features_list.append(price_action_df)

    # Combine the original data with all new feature batches
    return pd.concat([df] + new_features_list, axis=1)

def add_positioning_features(merged_chunk):
    """
    Calculates deep relational positioning features for each trade.

    This is a critical function that transforms simple SL/TP prices into
    intelligent features describing *how* they were placed relative to the
    market structure at the time of entry.

    It calculates two types of features for each indicator level:
    1.  `_dist_to_{level}_bps`: The raw distance from the SL/TP to an indicator
        level, measured in basis points (1/100th of 1%). This is useful for
        finding strategies like "place SL 50bps behind the SMA".
    2.  `_placement_pct_to_{level}`: Where the SL/TP was placed on a scale from
        the entry price (0%) to the indicator level (100%). This is powerful
        for finding strategies like "place TP 80% of the way to resistance".
    """
    level_cols = [
        'open', 'high', 'low', 'support', 'resistance',
        'BB_upper', 'BB_lower', 'ATR_level_up_1x', 'ATR_level_down_1x'
    ]
    level_cols.extend([col for col in merged_chunk.columns if 'SMA_' in col or 'EMA_' in col])
    
    entry, sl, tp, close = merged_chunk['entry_price'], merged_chunk['sl_price'], merged_chunk['tp_price'], merged_chunk['close']
    
    for level_name in level_cols:
        if level_name in merged_chunk.columns and not merged_chunk[level_name].isnull().all():
            level_price = merged_chunk[level_name]
            
            # Calculation 1: Distance in Basis Points
            merged_chunk[f'sl_dist_to_{level_name}_bps'] = ((sl - level_price) / close) * 10000
            merged_chunk[f'tp_dist_to_{level_name}_bps'] = ((tp - level_price) / close) * 10000

            # Calculation 2: Placement as a Percentage
            total_dist_to_level = level_price - entry
            sl_dist_from_entry = sl - entry
            tp_dist_from_entry = tp - entry

            # Calculate the percentage, handling division by zero safely
            merged_chunk[f'sl_placement_pct_to_{level_name}'] = (sl_dist_from_entry / total_dist_to_level).replace([np.inf, -np.inf], np.nan)
            merged_chunk[f'tp_placement_pct_to_{level_name}'] = (tp_dist_from_entry / total_dist_to_level).replace([np.inf, -np.inf], np.nan)

    return merged_chunk

def create_silver_data(bronze_path, raw_path, features_path, chunked_outcomes_dir):
    """
    Orchestrates the entire Silver Layer process for a single instrument.
    It performs the two main steps: feature generation and trade enrichment.
    """
    print(f"\n{'='*25}\nProcessing: {os.path.basename(raw_path)}\n{'='*25}")

    # --- STEP 1: Create and Save the Silver Features Dataset ---
    print("STEP 1: Creating Silver Features dataset...")
    raw_df = robust_read_csv(raw_path)
    if len(raw_df) < INDICATOR_WARMUP_PERIOD + 1:
        print(f"❌ SKIPPING: Not enough data for indicator warmup."); return

    features_df = add_all_market_features(raw_df)
    del raw_df; gc.collect() # Free up memory
    
    # Remove the initial warmup period rows where indicators are unreliable
    features_df = features_df.iloc[INDICATOR_WARMUP_PERIOD:].reset_index(drop=True)
    features_df = downcast_dtypes(features_df)
    
    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    features_df.to_csv(features_path, index=False)
    print(f"✅ Silver Features saved to: {features_path}")
    
    # --- STEP 2: Create ENRICHED and CHUNKED Silver Outcomes ---
    print("\nSTEP 2: Creating ENRICHED and CHUNKED Silver Outcomes...")
    os.makedirs(chunked_outcomes_dir, exist_ok=True)
    
    # Read the massive bronze file in chunks to manage memory usage
    bronze_iterator = pd.read_csv(bronze_path, chunksize=500_000, parse_dates=['entry_time'])
    
    # Define which features we need to merge for positioning calculations
    indicator_levels = ['support', 'resistance', 'BB_upper', 'BB_lower', 'ATR_level_up_1x', 'ATR_level_down_1x'] + [f"SMA_{p}" for p in SMA_PERIODS] + [f"EMA_{p}" for p in EMA_PERIODS]
    cols_to_keep = ['time', 'open', 'high', 'low', 'close'] + indicator_levels
    
    chunk_counter = 1
    for chunk in tqdm(bronze_iterator, desc="  Enriching Bronze Chunks"):
        # Discard trades that occurred during the indicator warmup period
        chunk = chunk[chunk['entry_time'] >= features_df['time'].min()]
        if chunk.empty: continue
        
        # Use merge_asof for a point-in-time correct join. This is crucial
        # to prevent lookahead bias by ensuring we only use features that
        # were known at the time of the trade's entry.
        merged_chunk = pd.merge_asof(
            chunk.sort_values('entry_time'), 
            features_df[cols_to_keep], 
            left_on='entry_time', 
            right_on='time', 
            direction='backward'
        )
        
        enriched_chunk = add_positioning_features(merged_chunk)
        
        # Drop the original indicator level columns after enrichment, as their
        # information is now encoded in the new '_bps' and '_pct' features.
        enriched_chunk.drop(columns=indicator_levels + ['time'], inplace=True, errors='ignore')
        
        if not enriched_chunk.empty:
            enriched_chunk = downcast_dtypes(enriched_chunk)
            
            # Save the enriched chunk directly to its own file. This avoids
            # creating another single giant file and prepares the data for
            # parallel processing in the Platinum layer.
            chunk_output_path = os.path.join(chunked_outcomes_dir, f"chunk_{chunk_counter}.csv")
            enriched_chunk.to_csv(chunk_output_path, index=False)
            chunk_counter += 1
            
    del features_df; gc.collect() # Free up memory
    print(f"✅ Enriched and chunked Silver Outcomes saved to: {chunked_outcomes_dir}")

if __name__ == "__main__":
    # --- Define Project Directory Structure ---
    core_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir, bronze_dir = [os.path.abspath(os.path.join(core_dir, '..', d)) for d in ['raw_data', 'bronze_data']]
    features_dir = os.path.abspath(os.path.join(core_dir, '..', 'silver_data', 'features'))
    chunked_outcomes_dir = os.path.abspath(os.path.join(core_dir, '..', 'silver_data', 'chunked_outcomes'))
    
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(chunked_outcomes_dir, exist_ok=True)

    # --- Find Bronze files to process ---
    bronze_files = [f for f in os.listdir(bronze_dir) if f.endswith('.csv')]
    if not bronze_files:
        print("❌ No bronze files found to process.")
    else:
        # --- Main Loop: Iterate through each instrument ---
        for fname in bronze_files:
            instrument_name = fname.replace('.csv', '')
            bronze_path = os.path.join(bronze_dir, fname)
            raw_path = os.path.join(raw_dir, fname)
            features_path = os.path.join(features_dir, fname)
            # The output directory for chunks is specific to each instrument
            instrument_chunked_outcomes_dir = os.path.join(chunked_outcomes_dir, instrument_name)
            
            # --- Pre-computation Checks ---
            if not os.path.exists(raw_path):
                print(f"⚠️ SKIPPING {fname}: Corresponding raw file not found."); continue
            
            # Skip if the final output directory already exists to make the script resumable
            if os.path.exists(instrument_chunked_outcomes_dir):
                print(f"✅ SKIPPING {fname}: Chunked outcomes directory already exists."); continue
            
            # --- Execute Processing ---
            try:
                create_silver_data(bronze_path, raw_path, features_path, instrument_chunked_outcomes_dir)
            except Exception as e:
                print(f"❌ FAILED to process {fname}. Error: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "="*50 + "\n✅ All silver data generation complete.")
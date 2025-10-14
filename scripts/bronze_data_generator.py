# bronze_data_generator.py (V3 - Fast, Memory-Efficient, and Correct)

import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import time
from numba import njit

# --- GLOBAL CONFIGURATION ---
MAX_CPU_USAGE = max(1, cpu_count() - 2)
# The number of trades to accumulate in memory before saving to disk.
OUTPUT_CHUNK_SIZE = 1_000_000
# The number of candles to process in each Numba batch. A tuning parameter.
INPUT_CHUNK_SIZE = 10000

# --- SPREAD CONFIGURATION (in Pips) ---
SPREAD_Pips = {
    "DEFAULT": 3.0, "EURUSD": 1.5, "GBPUSD": 2.0, "AUDUSD": 2.5, "USDJPY": 2.0, "USDCAD": 2.5, "XAUUSD": 20.0,
}

# --- Timeframe Presets (Unchanged) ---
TIMEFRAME_PRESETS = {
    "1m": {"SL_RATIOS": np.arange(0.0005, 0.0105, 0.0005), "TP_RATIOS": np.arange(0.0005, 0.0205, 0.0005), "MAX_LOOKFORWARD": 200},
    "5m": {"SL_RATIOS": np.arange(0.001, 0.0155, 0.0005), "TP_RATIOS": np.arange(0.001, 0.0305, 0.0005), "MAX_LOOKFORWARD": 300},
    "15m": {"SL_RATIOS": np.arange(0.002, 0.0255, 0.001), "TP_RATIOS": np.arange(0.002, 0.0505, 0.001), "MAX_LOOKFORWARD": 400},
    "30m": {"SL_RATIOS": np.arange(0.003, 0.0355, 0.001), "TP_RATIOS": np.arange(0.003, 0.0705, 0.001), "MAX_LOOKFORWARD": 500},
    "60m": {"SL_RATIOS": np.arange(0.005, 0.0505, 0.001), "TP_RATIOS": np.arange(0.005, 0.1005, 0.001), "MAX_LOOKFORWARD": 600},
    "240m": {"SL_RATIOS": np.arange(0.010, 0.1005, 0.001), "TP_RATIOS": np.arange(0.010, 0.2005, 0.001), "MAX_LOOKFORWARD": 800}
}

# --- Numba-Accelerated Core Logic ---
@njit
def find_winning_trades_numba(
    close_prices, high_prices, low_prices, timestamps,
    sl_ratios, tp_ratios, max_lookforward, spread_cost, processing_limit
):
    all_profitable_trades = []
    # Loop only up to the processing_limit to avoid re-processing the overlapping candles
    for i in range(processing_limit):
        entry_price, entry_time = close_prices[i], timestamps[i]
        
        # BUY trades
        for sl_r in sl_ratios:
            for tp_r in tp_ratios:
                sl_price, tp_price = entry_price * (1 - sl_r), entry_price * (1 + tp_r)
                limit = min(i + 1 + max_lookforward, len(close_prices))
                for j in range(i + 1, limit):
                    if high_prices[j] >= (tp_price + spread_cost):
                        all_profitable_trades.append((entry_time, 1, entry_price, sl_price, tp_price, sl_r, tp_r, timestamps[j]))
                        break
                    if low_prices[j] <= sl_price: break
        # SELL trades
        for sl_r in sl_ratios:
            for tp_r in tp_ratios:
                sl_price, tp_price = entry_price * (1 + sl_r), entry_price * (1 - tp_r)
                limit = min(i + 1 + max_lookforward, len(close_prices))
                for j in range(i + 1, limit):
                    if low_prices[j] <= (tp_price - spread_cost):
                        all_profitable_trades.append((entry_time, -1, entry_price, sl_price, tp_price, sl_r, tp_r, timestamps[j]))
                        break
                    if high_prices[j] >= sl_price: break
    return all_profitable_trades

def get_config_from_filename(filename):
    match = re.search(r'([A-Z0-9]+?)(\d+)\.csv$', filename)
    if not match:
        print(f"⚠️ Could not determine timeframe or instrument for {filename}. Skipping."); return None, None
    instrument, timeframe_num = match.group(1), match.group(2)
    timeframe_key = f"{timeframe_num}m"
    if timeframe_key not in TIMEFRAME_PRESETS:
        print(f"⚠️ No preset for timeframe '{timeframe_key}' in {filename}. Skipping."); return None, None
    print(f"✅ Config '{timeframe_key}' detected for {filename}.")
    if "JPY" in instrument.upper(): pip_size = 0.01
    elif "XAU" in instrument.upper() or "XAG" in instrument.upper(): pip_size = 0.01
    elif len(instrument) > 6 or any(char.isdigit() for char in instrument): pip_size = 0.1
    else: pip_size = 0.0001
    spread_in_pips = SPREAD_Pips.get(instrument, SPREAD_Pips["DEFAULT"])
    spread_cost = spread_in_pips * pip_size
    print(f"   -> Instrument: {instrument} | Pip Size: {pip_size:.4f} | Spread: {spread_in_pips} pips ({spread_cost:.4f})")
    return TIMEFRAME_PRESETS[timeframe_key], spread_cost

def process_file(task_id, input_file, output_file, config, spread_cost):
    filename = os.path.basename(input_file)
    try:
        df = pd.read_csv(input_file, sep=None, engine="python", header=None)
        df.columns = ["time", "open", "high", "low", "close", "volume"][:df.shape[1]]
        df["time"] = pd.to_datetime(df["time"])
        numeric_cols = ["open", "high", "low", "close"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
    except Exception as e:
        print(f"❌ Error loading or parsing {filename}: {e}"); return f"Error: {filename}"

    # --- RE-IMPLEMENTED MEMORY-SAVING LOGIC ---
    profitable_trades_accumulator = []
    total_trades_found = 0
    if os.path.exists(output_file): os.remove(output_file)
    is_first_chunk = True

    max_lookforward = config["MAX_LOOKFORWARD"]
    
    # Loop through the INPUT dataframe in chunks
    for i in tqdm(range(0, len(df), INPUT_CHUNK_SIZE), desc=f"Processing {filename}", position=task_id, leave=True):
        # Create a slice of the input data
        start_index = i
        end_index = i + INPUT_CHUNK_SIZE
        
        # Create an OVERLAPPING slice to provide lookahead data for Numba
        overlap_end_index = end_index + max_lookforward
        df_slice_with_overlap = df.iloc[start_index:overlap_end_index]
        
        # Convert this smaller slice to NumPy arrays
        close = df_slice_with_overlap["close"].values.astype(np.float64)
        high = df_slice_with_overlap["high"].values.astype(np.float64)
        low = df_slice_with_overlap["low"].values.astype(np.float64)
        timestamps = df_slice_with_overlap["time"].values.astype('datetime64[ns]').astype(np.int64)
        sl_ratios = config["SL_RATIOS"].astype(np.float64)
        tp_ratios = config["TP_RATIOS"].astype(np.float64)

        # Run the fast Numba function on the small chunk
        results_list = find_winning_trades_numba(
            close, high, low, timestamps, sl_ratios, tp_ratios, max_lookforward, spread_cost,
            processing_limit=len(df.iloc[start_index:end_index]) # Tell Numba how many candles to actually process
        )

        if results_list:
            profitable_trades_accumulator.extend(results_list)

        # Check if the accumulator has enough trades to be saved to disk
        if len(profitable_trades_accumulator) >= OUTPUT_CHUNK_SIZE:
            chunk_df = pd.DataFrame(profitable_trades_accumulator, columns=["entry_time", "trade_type", "entry_price", "sl_price", "tp_price", "sl_ratio", "tp_ratio", "exit_time"])
            chunk_df.to_csv(output_file, mode='a', header=is_first_chunk, index=False)
            total_trades_found += len(chunk_df)
            profitable_trades_accumulator.clear() # Clear memory
            is_first_chunk = False

    # Save any remaining trades after the loop finishes
    if profitable_trades_accumulator:
        final_chunk_df = pd.DataFrame(profitable_trades_accumulator, columns=["entry_time", "trade_type", "entry_price", "sl_price", "tp_price", "sl_ratio", "tp_ratio", "exit_time"])
        final_chunk_df.to_csv(output_file, mode='a', header=is_first_chunk, index=False)
        total_trades_found += len(final_chunk_df)

    if total_trades_found == 0:
        return f"No trades found for {filename}."

    # Final post-processing on the saved file to format timestamps and types
    final_df = pd.read_csv(output_file)
    final_df['entry_time'] = pd.to_datetime(final_df['entry_time'], unit='ns')
    final_df['exit_time'] = pd.to_datetime(final_df['exit_time'], unit='ns')
    final_df['trade_type'] = final_df['trade_type'].map({1: 'buy', -1: 'sell'})
    final_df['outcome'] = 'win'
    final_df.to_csv(output_file, index=False)

    return f"SUCCESS: {total_trades_found} trades found in {filename}."

if __name__ == "__main__":
    start_time = time.time()
    core_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.abspath(os.path.join(core_dir, '..', 'raw_data'))
    bronze_data_dir = os.path.abspath(os.path.join(core_dir, '..', 'bronze_data'))
    os.makedirs(bronze_data_dir, exist_ok=True)
    
    print("Warming up Numba JIT compiler... (this may take a moment on first run)")
    find_winning_trades_numba(np.random.rand(10), np.random.rand(10), np.random.rand(10), np.random.randint(0, 10, 10, dtype=np.int64), np.random.rand(2), np.random.rand(2), 1, 0.0001, 10)
    print("✅ Numba is ready.")

    try:
        raw_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.csv')]
    except FileNotFoundError:
        print(f"❌ Error: The directory '{raw_data_dir}' was not found."); raw_files = []

    if not raw_files: 
        print("❌ No CSV files found in 'raw_data'.")
    else:
        raw_files = [f for f in raw_files if not os.path.exists(os.path.join(bronze_data_dir, f))]
        print(f"Found {len(raw_files)} new files to process...")
        
        use_multiprocessing = input("Use multiprocessing? (y/n): ").strip().lower() == 'y'
        num_processes = MAX_CPU_USAGE if use_multiprocessing else 1
        
        tasks = []
        for task_id, filename in enumerate(raw_files):
            config, spread_cost = get_config_from_filename(filename)
            if config:
                input_path = os.path.join(raw_data_dir, filename)
                output_path = os.path.join(bronze_data_dir, filename)
                tasks.append((task_id, input_path, output_path, config, spread_cost))
        
        if not tasks:
            print("❌ No valid files to process.")
        else:
            effective_workers = min(num_processes, len(tasks))
            print(f"\n🚀 Starting processing with {effective_workers} workers.")
            if effective_workers > 1:
                with Pool(processes=effective_workers) as pool:
                    results = pool.starmap(process_file, tasks)
            else:
                results = [process_file(*task) for task in tasks]

            print("\n" + "="*50 + "\nProcessing Summary:")
            for res in results: print(res)

    end_time = time.time()
    print(f"\nBronze data generation complete. Total time: {end_time - start_time:.2f} seconds.")
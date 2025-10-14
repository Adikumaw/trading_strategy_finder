import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import time
from numba import njit, prange

# --- GLOBAL CONFIGURATION ---
MAX_CPU_USAGE = max(1, cpu_count() - 2)  # Leave 2 cores free
CHUNK_SIZE = 1_000_000  # Save data in chunks of 1 million rows

# --- NEW: SPREAD CONFIGURATION (in Pips) ---
# A pip is typically 0.0001 for most pairs, 0.01 for JPY pairs.
# We'll define it based on the instrument's quote currency later.
SPREAD_Pips = {
    "DEFAULT": 2.0,
    "EURUSD": 1.5,
    "GBPUSD": 2.0,
    "AUDUSD": 2.5,
    "USDJPY": 2.0,
    "USDCAD": 2.5,
}

# --- Timeframe Presets for SL/TP (Unchanged) ---
TIMEFRAME_PRESETS = {
    "1m": {"SL_RATIOS": np.arange(0.0005, 0.0105, 0.0005), "TP_RATIOS": np.arange(0.0005, 0.0205, 0.0005), "MAX_LOOKFORWARD": 200},
    "5m": {"SL_RATIOS": np.arange(0.001, 0.0155, 0.0005), "TP_RATIOS": np.arange(0.001, 0.0305, 0.0005), "MAX_LOOKFORWARD": 300},
    "15m": {"SL_RATIOS": np.arange(0.002, 0.0255, 0.001), "TP_RATIOS": np.arange(0.002, 0.0505, 0.001), "MAX_LOOKFORWARD": 400},
    "30m": {"SL_RATIOS": np.arange(0.003, 0.0355, 0.001), "TP_RATIOS": np.arange(0.003, 0.0705, 0.001), "MAX_LOOKFORWARD": 500},
    "60m": {"SL_RATIOS": np.arange(0.005, 0.0505, 0.001), "TP_RATIOS": np.arange(0.005, 0.1005, 0.001), "MAX_LOOKFORWARD": 600},
    "240m": {"SL_RATIOS": np.arange(0.010, 0.1005, 0.001), "TP_RATIOS": np.arange(0.010, 0.2005, 0.001), "MAX_LOOKFORWARD": 800}
}

# --- NEW: Numba-Accelerated Core Logic ---
@njit(parallel=True)
def find_winning_trades_numba(
    close_prices, high_prices, low_prices, timestamps,
    sl_ratios, tp_ratios, max_lookforward, spread_cost
):
    """
    Core trade-finding logic, JIT-compiled with Numba for extreme speed.
    This function processes NumPy arrays, not Pandas DataFrames.
    """
    n_candles = len(close_prices)
    all_profitable_trades = []

    # prange enables parallel execution of this outer loop
    for i in prange(n_candles - 1):
        entry_price = close_prices[i]
        entry_time = timestamps[i]
        
        # Local list for each parallel thread
        local_profitable_trades = []

        # --- Simulate BUY trades ---
        for sl_r in sl_ratios:
            for tp_r in tp_ratios:
                sl_price = entry_price * (1 - sl_r)
                tp_price = entry_price * (1 + tp_r)

                # Look forward for an outcome
                limit = min(i + 1 + max_lookforward, n_candles)
                for j in range(i + 1, limit):
                    # TP is only hit if the high overcomes the spread
                    if high_prices[j] >= (tp_price + spread_cost):
                        exit_time = timestamps[j]
                        # Append tuple: (entry_time, type, entry, sl, tp, sl_r, tp_r, exit_time)
                        local_profitable_trades.append((entry_time, 1, entry_price, sl_price, tp_price, sl_r, tp_r, exit_time))
                        break  # Outcome found, stop looking forward for this trade
                    if low_prices[j] <= sl_price:
                        break # SL hit, stop looking forward

        # --- Simulate SELL trades ---
        for sl_r in sl_ratios:
            for tp_r in tp_ratios:
                sl_price = entry_price * (1 + sl_r)
                tp_price = entry_price * (1 - tp_r)

                # Look forward for an outcome
                limit = min(i + 1 + max_lookforward, n_candles)
                for j in range(i + 1, limit):
                    # TP is only hit if the low overcomes the spread
                    if low_prices[j] <= (tp_price - spread_cost):
                        exit_time = timestamps[j]
                        local_profitable_trades.append((entry_time, -1, entry_price, sl_price, tp_price, sl_r, tp_r, exit_time))
                        break # Outcome found
                    if high_prices[j] >= sl_price:
                        break # SL hit
        
        # Combine results from this thread
        if local_profitable_trades:
            all_profitable_trades.extend(local_profitable_trades)
            
    return all_profitable_trades

def get_config_from_filename(filename):
    """Parses filename for timeframe and returns corresponding config and spread."""
    match = re.search(r'([A-Z]{6})(\d+)\.csv$', filename)
    if not match:
        print(f"⚠️ Could not determine timeframe or instrument for {filename}. Skipping.")
        return None, None

    instrument, timeframe_num = match.group(1), match.group(2)
    timeframe_key = f"{timeframe_num}m"
    
    if timeframe_key not in TIMEFRAME_PRESETS:
        print(f"⚠️ No preset found for timeframe '{timeframe_key}' in {filename}. Skipping.")
        return None, None

    print(f"✅ Config '{timeframe_key}' detected for {filename}.")
    
    # Determine pip size and spread cost
    pip_size = 0.01 if "JPY" in instrument else 0.0001
    spread_in_pips = SPREAD_Pips.get(instrument, SPREAD_Pips["DEFAULT"])
    spread_cost = spread_in_pips * pip_size
    print(f"   -> Applying spread: {spread_in_pips} pips ({spread_cost:.5f})")
    
    return TIMEFRAME_PRESETS[timeframe_key], spread_cost

def process_file(task_id, input_file, output_file, config, spread_cost):
    """
    Worker function. Prepares data for Numba, runs the simulation,
    and saves the results in memory-efficient chunks.
    """
    filename = os.path.basename(input_file)
    
    try:
        df = pd.read_csv(input_file, sep=None, engine="python", header=None)
        df.columns = ["time", "open", "high", "low", "close", "volume"][:df.shape[1]]
        df["time"] = pd.to_datetime(df["time"])
        numeric_cols = ["open", "high", "low", "close"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
    except Exception as e:
        print(f"❌ Error loading or parsing {filename}: {e}"); return f"Error: {filename}"

    # --- Convert to NumPy arrays for Numba ---
    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    timestamps = df["time"].values.astype('datetime64[ns]').astype(np.int64)
    sl_ratios = config["SL_RATIOS"].astype(np.float64)
    tp_ratios = config["TP_RATIOS"].astype(np.float64)
    max_lookforward = config["MAX_LOOKFORWARD"]

    print(f"🚀 [{filename}] Starting Numba calculation...")
    start_numba = time.time()
    
    results_list = find_winning_trades_numba(
        close, high, low, timestamps, sl_ratios, tp_ratios, max_lookforward, spread_cost
    )
    
    end_numba = time.time()
    print(f"🏁 [{filename}] Numba finished in {end_numba - start_numba:.2f}s. Found {len(results_list)} trades.")

    if not results_list:
        return f"No trades found for {filename}."

    # --- Convert results back to DataFrame and save in chunks ---
    results_df = pd.DataFrame(results_list, columns=[
        "entry_time", "trade_type", "entry_price", "sl_price", "tp_price",
        "sl_ratio", "tp_ratio", "exit_time"
    ])
    results_df['entry_time'] = pd.to_datetime(results_df['entry_time'])
    results_df['exit_time'] = pd.to_datetime(results_df['exit_time'])
    results_df['trade_type'] = results_df['trade_type'].map({1: 'buy', -1: 'sell'})
    results_df['outcome'] = 'win'

    if os.path.exists(output_file): os.remove(output_file)
    
    header = True
    for i in tqdm(range(0, len(results_df), CHUNK_SIZE), desc=f"Saving {filename}", position=task_id, leave=True):
        chunk = results_df.iloc[i:i + CHUNK_SIZE]
        chunk.to_csv(output_file, mode='a', header=header, index=False)
        header = False
    
    return f"SUCCESS: {len(results_df)} trades found in {filename}."

if __name__ == "__main__":
    start_time = time.time()
    core_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.abspath(os.path.join(core_dir, '..', 'raw_data'))
    bronze_data_dir = os.path.abspath(os.path.join(core_dir, '..', 'bronze_data'))
    os.makedirs(bronze_data_dir, exist_ok=True)
    
    # --- One-time Numba compilation warmup ---
    print("Warming up Numba JIT compiler... (this may take a moment on first run)")
    find_winning_trades_numba(
        np.random.rand(10), np.random.rand(10), np.random.rand(10),
        np.random.randint(0, 10, 10), np.random.rand(2), np.random.rand(2), 1, 0.0001
    )
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
        if use_multiprocessing:
            num_processes = MAX_CPU_USAGE
        else:
            try: num_processes = int(input(f"Enter number of processes (1 for single process): ").strip())
            except ValueError: num_processes = 1
        
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
            print(f"\n🚀 Starting processing with {min(num_processes, len(tasks))} workers.")
            if min(num_processes, len(tasks)) > 1:
                with Pool(processes=min(num_processes, len(tasks))) as pool:
                    results = pool.starmap(process_file, tasks)
            else:
                results = [process_file(*task) for task in tasks]

            print("\n" + "="*50 + "\nProcessing Summary:")
            for res in results: print(res)

    end_time = time.time()
    print(f"\nBronze data generation complete. Total time: {end_time - start_time:.2f} seconds.")
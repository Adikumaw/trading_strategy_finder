# bronze_data_generator_nolimits.py

import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import time

# --- GLOBAL CONFIGURATION ---
# No CHUNK_SIZE needed in the 'nolimits' version.
MAX_CPU_USAGE = max(1, cpu_count() - 2)

# --- Timeframe Presets (Unchanged) ---
TIMEFRAME_PRESETS = {
    "1m": {"SL_RATIOS": np.arange(0.0005, 0.0105, 0.0005), "TP_RATIOS": np.arange(0.0005, 0.0205, 0.0005), "MAX_LOOKFORWARD": 200},
    "5m": {"SL_RATIOS": np.arange(0.001, 0.0155, 0.0005), "TP_RATIOS": np.arange(0.001, 0.0305, 0.0005), "MAX_LOOKFORWARD": 300},
    "15m": {"SL_RATIOS": np.arange(0.002, 0.0255, 0.001), "TP_RATIOS": np.arange(0.002, 0.0505, 0.001), "MAX_LOOKFORWARD": 400},
    "30m": {"SL_RATIOS": np.arange(0.003, 0.0355, 0.001), "TP_RATIOS": np.arange(0.003, 0.0705, 0.001), "MAX_LOOKFORWARD": 500},
    "60m": {"SL_RATIOS": np.arange(0.005, 0.0505, 0.001), "TP_RATIOS": np.arange(0.005, 10.0005, 0.001), "MAX_LOOKFORWARD": 600}, # Corrected typo in TP_RATIOS
    "240m": {"SL_RATIOS": np.arange(0.010, 0.1005, 0.001), "TP_RATIOS": np.arange(0.010, 0.2005, 0.001), "MAX_LOOKFORWARD": 800}
}

def get_config_from_filename(filename):
    match = re.search(r'(\d+)\.csv$', filename)
    if match:
        timeframe_key = f"{match.group(1)}m"
        if timeframe_key in TIMEFRAME_PRESETS:
            print(f"✅ Timeframe '{timeframe_key}' detected for {filename}.")
            return TIMEFRAME_PRESETS[timeframe_key]
    print(f"⚠️ Could not determine a valid timeframe preset for {filename}. Skipping file.")
    return None

def process_file_nolimits(task_id, input_file, output_file, config):
    """
    'No Limits' worker function. It collects all results in memory and writes once at the end.
    """
    filename = os.path.basename(input_file)
    
    SL_RATIOS, TP_RATIOS, MAX_LOOKFORWARD = config["SL_RATIOS"], config["TP_RATIOS"], config["MAX_LOOKFORWARD"]

    try:
        df = pd.read_csv(input_file, sep=None, engine="python")
        if df.shape[1] > 5: df = df.iloc[:, :5]
        df.columns = ["time", "open", "high", "low", "close"]
        df["time"] = pd.to_datetime(df["time"])
        df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].apply(pd.to_numeric)
    except Exception as e:
        print(f"❌ Error loading or parsing {filename}: {e}")
        return f"Error: {filename}"

    profitable_trades = []
    
    # The main loop remains the same, but it no longer checks for chunk size.
    for i in tqdm(range(len(df) - 1), desc=f"{filename}", position=task_id, leave=True):
        entry_time, entry_price = df.loc[i, "time"], df.loc[i, "close"]

        # Using lists and direct checks for a potential micro-optimization over dicts for this specific logic
        buy_tps = [[entry_price * (1 + r), r] for r in TP_RATIOS]
        buy_sls = [[entry_price * (1 - r), r] for r in SL_RATIOS]
        sell_tps = [[entry_price * (1 - r), r] for r in TP_RATIOS]
        sell_sls = [[entry_price * (1 + r), r] for r in SL_RATIOS]

        lookforward_limit = min(i + 1 + MAX_LOOKFORWARD, len(df))
        
        # Vectorizing the inner loop is complex, so we keep the candle-by-candle simulation
        for j in range(i + 1, lookforward_limit):
            future_high, future_low, exit_time = df.loc[j, "high"], df.loc[j, "low"], df.loc[j, "time"]

            # Buy side check
            if buy_sls:
                # Find TPs that hit
                hit_tps = [tp for tp in buy_tps if tp[0] <= future_high]
                if hit_tps:
                    for tp_p, tp_r in hit_tps:
                        for sl_p, sl_r in buy_sls: # For every TP hit, pair it with all *currently active* SLs
                            profitable_trades.append({"entry_time": entry_time, "trade_type": "buy", "entry_price": entry_price, "sl_price": sl_p, "tp_price": tp_p, "sl_ratio": sl_r, "tp_ratio": tp_r, "exit_time": exit_time, "outcome": "win"})
                    # Remove the TPs that have been hit
                    buy_tps = [tp for tp in buy_tps if tp not in hit_tps]
                
                # Invalidate SLs that would have been hit on this candle
                buy_sls = [sl for sl in buy_sls if sl[0] < future_low]

            # Sell side check
            if sell_sls:
                hit_tps = [tp for tp in sell_tps if tp[0] >= future_low]
                if hit_tps:
                    for tp_p, tp_r in hit_tps:
                        for sl_p, sl_r in sell_sls:
                            profitable_trades.append({"entry_time": entry_time, "trade_type": "sell", "entry_price": entry_price, "sl_price": sl_p, "tp_price": tp_p, "sl_ratio": sl_r, "tp_ratio": tp_r, "exit_time": exit_time, "outcome": "win"})
                    sell_tps = [tp for tp in sell_tps if tp not in hit_tps]
                
                sell_sls = [sl for sl in sell_sls if sl[0] > future_high]
            
            # If all possible trades have been resolved, break early
            if not buy_sls and not sell_sls:
                break
    
    if not profitable_trades:
        return f"No trades found for {filename}."

    # --- SPEED OPTIMIZATION: Write the entire result to disk in one operation ---
    pd.DataFrame(profitable_trades).to_csv(output_file, index=False)
    
    return f"SUCCESS: {len(profitable_trades)} trades found in {filename}."

if __name__ == "__main__":
    start_time = time.time()
    core_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.abspath(os.path.join(core_dir, '..', 'raw_data'))
    bronze_data_dir = os.path.abspath(os.path.join(core_dir, '..', 'bronze_data'))
    os.makedirs(bronze_data_dir, exist_ok=True)

    try:
        raw_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.csv')]
    except FileNotFoundError:
        print(f"❌ Error: The directory '{raw_data_dir}' was not found."); raw_files = []

    if not raw_files: 
        print("❌ No CSV files found in 'raw_data'.")
    else:
        print(f"Found {len(raw_files)} files to process...")
        raw_files = [f for f in raw_files if not os.path.exists(os.path.join(bronze_data_dir, f))]
        print(f"{len(raw_files)} files remaining after filtering.")

        # --- User interaction for multiprocessing (Unchanged) ---
        use_multiprocessing = input("Use multiprocessing? (y/n): ").strip().lower() == 'y'
        if use_multiprocessing:
            num_processes = MAX_CPU_USAGE
        else:
            try:
                num_processes = int(input("Enter number of processes to use (1 for single process): ").strip())
                if num_processes < 1 or num_processes > cpu_count(): raise ValueError("Invalid process count.")
            except ValueError:
                print("Invalid input. Defaulting to 1 process."); num_processes = 1

        tasks = []
        for task_id, filename in enumerate(raw_files):
            config = get_config_from_filename(filename)
            if config:
                tasks.append((task_id, os.path.join(raw_data_dir, filename), os.path.join(bronze_data_dir, filename), config))
        
        if not tasks:
            print("❌ No valid files to process.")
        else:
            print(f"\n🚀 Starting processing with {num_processes} workers for {len(tasks)} files.")
            
            if num_processes > 1:
                with Pool(processes=num_processes) as pool:
                    results = pool.starmap(process_file_nolimits, tasks)
            else:
                results = [process_file_nolimits(*task) for task in tasks]

            print("\n--- Processing Summary ---")
            for res in results: print(res)

    end_time = time.time()
    print(f"\nBronze data generation complete. Total time: {end_time - start_time:.2f} seconds.")
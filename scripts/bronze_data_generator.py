import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import time

# --- GLOBAL CONFIGURATION ---
USE_MULTIPROCESSING = False  # Set to False for running on custom process count for limited testing
MAX_CPU_USAGE = max(1, cpu_count() - 2)  # Leave 2 cores free for system responsiveness

# --- Timeframe Presets for SL/TP ---
# This dictionary remains unchanged.
TIMEFRAME_PRESETS = {
    "1m": {
        "SL_RATIOS": np.arange(0.0005, 0.0105, 0.0005),  # 0.05% - 1.0%
        "TP_RATIOS": np.arange(0.0005, 0.0205, 0.0005),  # 0.05% - 2.0%
        "MAX_LOOKFORWARD": 200
    },
    "5m": {
        "SL_RATIOS": np.arange(0.001, 0.0155, 0.0005),  # 0.1% - 1.5%
        "TP_RATIOS": np.arange(0.001, 0.0305, 0.0005),  # 0.1% - 3.0%
        "MAX_LOOKFORWARD": 300
    },
    "15m": {
        "SL_RATIOS": np.arange(0.002, 0.0255, 0.001),   # 0.2% - 2.5%
        "TP_RATIOS": np.arange(0.002, 0.0505, 0.001),   # 0.2% - 5.0%
        "MAX_LOOKFORWARD": 400
    },
    "30m": {
        "SL_RATIOS": np.arange(0.003, 0.0355, 0.001),   # 0.3% - 3.5%
        "TP_RATIOS": np.arange(0.003, 0.0705, 0.001),   # 0.3% - 7.0%
        "MAX_LOOKFORWARD": 500
    },
    "60m": {
        "SL_RATIOS": np.arange(0.005, 0.0505, 0.001),   # 0.5% - 5.0%
        "TP_RATIOS": np.arange(0.005, 0.1005, 0.001),   # 0.5% - 10.0%
        "MAX_LOOKFORWARD": 600
    },
    "240m": {
        "SL_RATIOS": np.arange(0.010, 0.1005, 0.001),   # 1.0% - 10.0%
        "TP_RATIOS": np.arange(0.010, 0.2005, 0.001),   # 1.0% - 20.0%
        "MAX_LOOKFORWARD": 800
    }
}

def get_config_from_filename(filename):
    """
    Parses the filename to find a timeframe number and returns the corresponding config.
    Example: "EURUSD15.csv" -> extracts "15" -> returns config for "15m".
    """
    # This regex finds any numbers that come right before the '.csv' extension
    match = re.search(r'(\d+)\.csv$', filename)
    if match:
        timeframe_key = f"{match.group(1)}m"
        if timeframe_key in TIMEFRAME_PRESETS:
            print(f"✅ Timeframe '{timeframe_key}' detected for {filename}.")
            return TIMEFRAME_PRESETS[timeframe_key]

    print(f"⚠️ Could not determine a valid timeframe preset for {filename}. Skipping file.")
    return None

def process_file(task_id, input_file, output_file, config):
    """
    Worker function for multiprocessing. It now uses a task_id to position its progress bar.
    """
    filename = os.path.basename(input_file)
    
    # Unpack the config for this specific task
    SL_RATIOS = config["SL_RATIOS"]
    TP_RATIOS = config["TP_RATIOS"]
    MAX_LOOKFORWARD = config["MAX_LOOKFORWARD"]

    try:
        df = pd.read_csv(input_file, sep=None, engine="python")
        if df.shape[1] > 5: df = df.iloc[:, :5]
        df.columns = ["time", "open", "high", "low", "close"]
        df["time"] = pd.to_datetime(df["time"])
        df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].apply(pd.to_numeric)
    except Exception as e:
        # Using a lock or some other method to print safely would be ideal, but for errors it's often okay.
        print(f"❌ Error loading or parsing {filename}: {e}")
        return f"Error: {filename}"

    profitable_trades = []
    
    # MODIFICATION 2: The `position=task_id` argument tells tqdm which line to draw on.
    # `leave=True` ensures the bar stays on screen after completion.
    for i in tqdm(range(len(df) - 1), desc=f"{filename}", position=task_id, leave=True):
        entry_time, entry_price = df.loc[i, "time"], df.loc[i, "close"]

        buy_tps = {round(entry_price * (1 + r), 5): r for r in TP_RATIOS}
        buy_sls = {round(entry_price * (1 - r), 5): r for r in SL_RATIOS}
        sell_tps = {round(entry_price * (1 - r), 5): r for r in TP_RATIOS}
        sell_sls = {round(entry_price * (1 + r), 5): r for r in SL_RATIOS}

        lookforward_limit = min(i + 1 + MAX_LOOKFORWARD, len(df))
        for j in range(i + 1, lookforward_limit):
            future_high, future_low, exit_time = df.loc[j, "high"], df.loc[j, "low"], df.loc[j, "time"]

            # The core logic for finding trades remains the same
            if buy_sls:
                hit_tps = {p: r for p, r in buy_tps.items() if p <= future_high}
                if hit_tps:
                    for tp_p, tp_r in hit_tps.items():
                        for sl_p, sl_r in buy_sls.items():
                            profitable_trades.append({"entry_time": entry_time, "trade_type": "buy", "entry_price": entry_price, "sl_price": sl_p, "tp_price": tp_p, "sl_ratio": sl_r, "tp_ratio": tp_r, "exit_time": exit_time, "outcome": "win"})
                    buy_tps = {p: r for p, r in buy_tps.items() if p not in hit_tps}
                hit_sls = {p for p in buy_sls if p >= future_low}
                if hit_sls: buy_sls = {p: r for p, r in buy_sls.items() if p not in hit_sls}
            
            if sell_sls:
                hit_tps = {p: r for p, r in sell_tps.items() if p >= future_low}
                if hit_tps:
                    for tp_p, tp_r in hit_tps.items():
                        for sl_p, sl_r in sell_sls.items():
                            profitable_trades.append({"entry_time": entry_time, "trade_type": "sell", "entry_price": entry_price, "sl_price": sl_p, "tp_price": tp_p, "sl_ratio": sl_r, "tp_ratio": tp_r, "exit_time": exit_time, "outcome": "win"})
                    sell_tps = {p: r for p, r in sell_tps.items() if p not in hit_tps}
                hit_sls = {p for p in sell_sls if p <= future_high}
                if hit_sls: sell_sls = {p: r for p, r in sell_sls.items() if p not in hit_sls}

            if not buy_sls and not sell_sls: break
    
    if not profitable_trades:
        return f"No trades found for {filename}."

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
        print(f"❌ Error: The directory '{raw_data_dir}' was not found.")
        raw_files = []

    if not raw_files: 
        print("❌ No CSV files found in 'raw_data'.")
    else:
        # --- 1. Prepare the list of tasks for the multiprocessing pool ---
        print(f"Found {len(raw_files)} files to process...")
        
        # remove files that are already processed and present in bronze_data
        raw_files = [f for f in raw_files if not os.path.exists(os.path.join(bronze_data_dir, f))]
        print(f"{len(raw_files)} files remaining after filtering already processed files.")

        # Ask user for which processing mode to use
        use_multiprocessing = input("Use multiprocessing? (y/n): ").strip().lower() == 'y'
        if use_multiprocessing:
            num_processes = MAX_CPU_USAGE
        else:
            num_processes = int(input("Enter number of processes to use (1 for single process): ").strip())
            try:
                if num_processes < 1 or num_processes > MAX_CPU_USAGE:
                    raise ValueError
            except ValueError:
                print("Invalid input Or out of range. Defaulting to 1 process.")
                num_processes = 1

        tasks = []
        # MODIFICATION 3: Use enumerate to add a unique task_id to each task.
        for task_id, filename in enumerate(raw_files):
            config = get_config_from_filename(filename)
            if config:
                input_path = os.path.join(raw_data_dir, filename)
                output_path = os.path.join(bronze_data_dir, filename)
                tasks.append((task_id, input_path, output_path, config))
            else:
                print(f"⚠️ Skipping {filename}: No valid timeframe config found.")
        
        if not tasks:
            print("❌ No valid files to process after checking configurations.")
        else:
            print("\n" + "="*50)
            print(f"🚀 Starting multiprocessing pool with {num_processes} workers for {len(tasks)} files.")
            print("="*50 + "\n")

            with Pool(processes=num_processes) as pool:
                # MODIFICATION 4: Use `starmap` to pass the multiple arguments from each task tuple.
                results = pool.starmap(process_file, tasks)

            # Print results after all processes are finished
            print("\n" + "="*50 + "\nProcessing Summary:")
            for res in results:
                print(res)

    end_time = time.time()
    print("\n" + "="*50)
    print(f"Bronze data generation complete. Total time: {end_time - start_time:.2f} seconds.")
    print("="*50)
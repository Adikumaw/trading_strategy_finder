import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import time

# --- Timeframe Presets for SL/TP ---
# This dictionary remains unchanged.
TIMEFRAME_PRESETS = {
    "1m": {
        "SL_RATIOS": np.arange(0.001, 0.0105, 0.001), "TP_RATIOS": np.arange(0.001, 0.0205, 0.001), "MAX_LOOKFORWARD": 200
    },
    "5m": {
        "SL_RATIOS": np.arange(0.002, 0.0155, 0.001), "TP_RATIOS": np.arange(0.002, 0.0305, 0.001), "MAX_LOOKFORWARD": 300
    },
    "15m": {
        "SL_RATIOS": np.arange(0.003, 0.0205, 0.001), "TP_RATIOS": np.arange(0.003, 0.0405, 0.001), "MAX_LOOKFORWARD": 400
    },
    "30m": {
        "SL_RATIOS": np.arange(0.005, 0.0255, 0.001), "TP_RATIOS": np.arange(0.005, 0.0505, 0.001), "MAX_LOOKFORWARD": 500
    },
    "60m": {
        "SL_RATIOS": np.arange(0.010, 0.0305, 0.001), "TP_RATIOS": np.arange(0.010, 0.0705, 0.001), "MAX_LOOKFORWARD": 600
    },
    "240m": {
        "SL_RATIOS": np.arange(0.020, 0.0505, 0.001), "TP_RATIOS": np.arange(0.020, 0.1005, 0.001), "MAX_LOOKFORWARD": 800
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

def process_file(task_args):
    """
    Wrapper function for multiprocessing. Unpacks arguments and calls the main processing logic.
    """
    input_file, output_file, config = task_args
    filename = os.path.basename(input_file)
    print(f"Starting processing for: {filename}")

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
        print(f"❌ Error loading or parsing {filename}: {e}")
        return

    profitable_trades = []
    
    # tqdm is used here to show progress for each individual file process
    for i in tqdm(range(len(df) - 1), desc=f"Processing {filename}", position=0, leave=True):
        entry_time, entry_price = df.loc[i, "time"], df.loc[i, "close"]

        buy_tps = {round(entry_price * (1 + r), 5): r for r in TP_RATIOS}
        buy_sls = {round(entry_price * (1 - r), 5): r for r in SL_RATIOS}
        sell_tps = {round(entry_price * (1 - r), 5): r for r in TP_RATIOS}
        sell_sls = {round(entry_price * (1 + r), 5): r for r in SL_RATIOS}

        lookforward_limit = min(i + 1 + MAX_LOOKFORWARD, len(df))
        for j in range(i + 1, lookforward_limit):
            future_high, future_low, exit_time = df.loc[j, "high"], df.loc[j, "low"], df.loc[j, "time"]

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
        print(f"⚠️ No profitable trades found for {filename}.")
        return

    print(f"\n✅ Found {len(profitable_trades)} trades in {filename}. Saving...")
    pd.DataFrame(profitable_trades).to_csv(output_file, index=False)
    print(f"SUCCESS: Data for {filename} saved to {output_file}")


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
        tasks = []
        for filename in raw_files:
            config = get_config_from_filename(filename)
            if config:
                input_path = os.path.join(raw_data_dir, filename)
                output_path = os.path.join(bronze_data_dir, filename)
                tasks.append((input_path, output_path, config))
            else:
                print(f"⚠️ Skipping {filename}: No valid timeframe config found.")
        
        if not tasks:
            print("❌ No valid files to process after checking configurations.")
        else:
            # --- 2. Set up and run the processing pool ---
            # Use one less than the total number of CPUs to keep the system responsive
            num_processes = max(1, cpu_count() - 1)
            print("\n" + "="*50)
            print(f"🚀 Starting multiprocessing pool with {num_processes} workers to process {len(tasks)} files.")
            print("="*50 + "\n")

            with Pool(processes=num_processes) as pool:
                # The map function will distribute the 'tasks' list to the 'process_file' function
                pool.map(process_file, tasks)

    end_time = time.time()
    print("\n" + "="*50)
    print(f"Bronze data generation complete. Total time: {end_time - start_time:.2f} seconds.")
    print("="*50)
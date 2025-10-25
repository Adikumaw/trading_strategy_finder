# bronze_data_generator.py (V11 - Final Architecture)

"""
Bronze Layer: The Possibility Engine

This script is the foundational data generation layer of the entire strategy
discovery pipeline. Its purpose is to systematically scan historical price data
and generate a vast, high-quality dataset of every conceivable winning trade
based on a predefined set of rules.

This "universe of possibilities" forms the bedrock upon which all subsequent
analysis is built. It operates by performing a brute-force simulation for every
candle, testing thousands of Stop-Loss (SL) and Take-Profit (TP) combinations
and recording only those that would have resulted in a win.

This version uses a sophisticated intra-file parallelism model (Producer-Consumer)
with a worker initializer for maximum speed, memory safety, ordered output,
and cross-platform stability (especially on Windows).
"""

import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import time
from numba import njit
import sys

# --- GLOBAL CONFIGURATION ---

# Sets the maximum number of CPU cores to use for multiprocessing.
# Leaves 2 cores free to ensure system responsiveness.
MAX_CPU_USAGE = max(1, cpu_count() - 2)

# The number of discovered winning trades to accumulate in memory before
# flushing them to the output CSV file. This is a key memory-saving feature.
OUTPUT_CHUNK_SIZE = 1_000_000

# The number of candles from the input file to process in each batch passed
# to the Numba JIT-compiled function. This is a performance tuning parameter.
INPUT_CHUNK_SIZE = 7000

# --- SPREAD CONFIGURATION (in Pips) ---

# Defines the estimated trading spread for various instruments in pips.
# A default value is used if a specific instrument is not listed.
# This ensures backtests are more realistic by accounting for transaction costs.
SPREAD_Pips = {
    "DEFAULT": 3.0, "EURUSD": 1.5, "GBPUSD": 2.0, "AUDUSD": 2.5, "USDJPY": 2.0, "USDCAD": 2.5, "XAUUSD": 20.0,
}

# --- Timeframe Presets ---

# Defines the simulation parameters for different chart timeframes.
# SL_RATIOS: A NumPy array of stop-loss percentages to test (e.g., 0.001 = 0.1%).
# TP_RATIOS: A NumPy array of take-profit percentages to test.
# MAX_LOOKFORWARD: The maximum number of future candles to check for a trade's outcome.
#                  This prevents simulations from running indefinitely.
TIMEFRAME_PRESETS = {
    "1m": {"SL_RATIOS": np.arange(0.0005, 0.0105, 0.0005), "TP_RATIOS": np.arange(0.0005, 0.0205, 0.0005), "MAX_LOOKFORWARD": 200},
    "5m": {"SL_RATIOS": np.arange(0.001, 0.0155, 0.0005), "TP_RATIOS": np.arange(0.001, 0.0305, 0.0005), "MAX_LOOKFORWARD": 300},
    "15m": {"SL_RATIOS": np.arange(0.002, 0.0255, 0.001), "TP_RATIOS": np.arange(0.002, 0.0505, 0.001), "MAX_LOOKFORWARD": 400},
    "30m": {"SL_RATIOS": np.arange(0.003, 0.0355, 0.001), "TP_RATIOS": np.arange(0.003, 0.0705, 0.001), "MAX_LOOKFORWARD": 500},
    "60m": {"SL_RATIOS": np.arange(0.005, 0.0505, 0.001), "TP_RATIOS": np.arange(0.005, 0.1005, 0.001), "MAX_LOOKFORWARD": 600},
    "240m": {"SL_RATIOS": np.arange(0.010, 0.1005, 0.001), "TP_RATIOS": np.arange(0.010, 0.2005, 0.001), "MAX_LOOKFORWARD": 800}
}


# --- GLOBAL VARIABLES FOR WORKER INITIALIZATION ---
# These variables will be populated in each worker process upon creation.
# This is a robust pattern that avoids passing large objects through inter-process
# communication, which is slow and can be unstable on Windows.
worker_df = None
worker_config = None
worker_spread_cost = None
worker_max_lookforward = None

def init_worker(df, config, spread_cost, max_lookforward):
    """
    Initializer function for each worker process in the multiprocessing Pool.
    
    This function is called once per worker when the pool is spawned. It receives the
    large, shared, read-only data objects (like the main DataFrame) and stores them
    in global variables within that specific worker's memory space. This prevents
    the need to pickle and transfer this large data for every single task,
    improving performance and stability.
    """
    global worker_df, worker_config, worker_spread_cost, worker_max_lookforward
    worker_df = df
    worker_config = config
    worker_spread_cost = spread_cost
    worker_max_lookforward = max_lookforward


# --- Numba-Accelerated Core Logic ---

@njit
def find_winning_trades_numba(
    close_prices, high_prices, low_prices, timestamps,
    sl_ratios, tp_ratios, max_lookforward, spread_cost, processing_limit
):
    """
    Executes the core trade simulation logic at high speed using Numba.

    This function is the performance-critical engine of the script. It iterates
    through a pre-loaded chunk of candlestick data and, for each candle, simulates
    every possible buy and sell trade defined by the grid of Stop-Loss (SL) and
    Take-Profit (TP) ratios. It then looks forward in the data to determine if
    a trade would have hit its TP before its SL, accounting for spread costs.
    Only profitable trades are recorded.

    Args:
        close_prices (np.array): A NumPy array of closing prices for the data chunk.
        high_prices (np.array): A NumPy array of high prices for the data chunk.
        low_prices (np.array): A NumPy array of low prices for the data chunk.
        timestamps (np.array): A NumPy array of timestamps (as int64) for each candle.
        sl_ratios (np.array): A NumPy array of stop-loss ratios to be tested.
        tp_ratios (np.array): A NumPy array of take-profit ratios to be tested.
        max_lookforward (int): The maximum number of future candles to check for an outcome.
        spread_cost (float): The calculated spread cost for the instrument, applied to TP levels.
        processing_limit (int): The number of candles within the chunk to process,
                                 which excludes the overlapping lookahead data at the end.

    Returns:
        list: A list of tuples, where each tuple contains the details of a
              single profitable trade found during the simulation.
    """
    all_profitable_trades = []
    # Loop only up to the processing_limit to avoid re-processing the overlapping candles
    for i in range(processing_limit):
        entry_price, entry_time = close_prices[i], timestamps[i]
        
        # --- Simulate BUY trades ---
        for sl_r in sl_ratios:
            for tp_r in tp_ratios:
                sl_price = entry_price * (1 - sl_r)
                tp_price = entry_price * (1 + tp_r)
                # Define the look-forward window for this specific trade
                limit = min(i + 1 + max_lookforward, len(close_prices))
                for j in range(i + 1, limit):
                    # Check for a win (TP hit), accounting for the spread cost
                    if high_prices[j] >= (tp_price + spread_cost):
                        # If a win is found, record it and stop checking for this SL/TP combo
                        all_profitable_trades.append((entry_time, 1, entry_price, sl_price, tp_price, sl_r, tp_r, timestamps[j]))
                        break
                    # Check for a loss (SL hit)
                    if low_prices[j] <= sl_price:
                        # If a loss is found, stop checking for this SL/TP combo
                        break
                        
        # --- Simulate SELL trades ---
        for sl_r in sl_ratios:
            for tp_r in tp_ratios:
                sl_price = entry_price * (1 + sl_r)
                tp_price = entry_price * (1 - tp_r)
                limit = min(i + 1 + max_lookforward, len(close_prices))
                for j in range(i + 1, limit):
                    # Check for a win (TP hit), accounting for the spread cost
                    if low_prices[j] <= (tp_price - spread_cost):
                        all_profitable_trades.append((entry_time, -1, entry_price, sl_price, tp_price, sl_r, tp_r, timestamps[j]))
                        break
                    # Check for a loss (SL hit)
                    if high_prices[j] >= sl_price:
                        break
                        
    return all_profitable_trades


# --- HELPER & WORKER FUNCTIONS ---
def get_config_from_filename(filename):
    """
    Parses a filename to extract the instrument and timeframe, then retrieves the correct preset.

    This function uses regular expressions to identify the trading instrument
    (e.g., "EURUSD", "XAUUSD") and the chart timeframe (e.g., "15", "60") from
    the input CSV filename. It then calculates the appropriate spread cost in
    the instrument's quote currency and retrieves the corresponding simulation
    parameters from the TIMEFRAME_PRESETS global dictionary.

    Args:
        filename (str): The name of the raw data file (e.g., "XAUUSD15.csv").

    Returns:
        tuple: A tuple containing two elements:
               - The configuration dictionary from TIMEFRAME_PRESETS.
               - The calculated spread cost as a float.
               Returns (None, None) if the filename cannot be parsed or if no
               matching preset is found.
    """
    # Use regex to extract the instrument name and timeframe number from the filename
    match = re.search(r'([A-Z0-9]+?)(\d+)\.csv$', filename)
    if not match:
        print(f"[WARNING] Could not determine timeframe or instrument for {filename}. Skipping.")
        return None, None
        
    instrument, timeframe_num = match.group(1), match.group(2)
    timeframe_key = f"{timeframe_num}m"
    
    # Check if a preset exists for the detected timeframe
    if timeframe_key not in TIMEFRAME_PRESETS:
        print(f"[WARNING] No preset for timeframe '{timeframe_key}' in {filename}. Skipping.")
        return None, None
        
    print(f"[SUCCESS] Config '{timeframe_key}' detected for {filename}.")
    
    # --- Determine Pip Size for Accurate Spread Calculation ---
    if "JPY" in instrument.upper():
        pip_size = 0.01
    elif "XAU" in instrument.upper() or "XAG" in instrument.upper():
        pip_size = 0.01
    elif len(instrument) > 6 or any(char.isdigit() for char in instrument): # For indices/crypto
        pip_size = 0.1
    else: # Standard Forex pairs
        pip_size = 0.0001
        
    # Calculate the final spread cost in the instrument's quote currency
    spread_in_pips = SPREAD_Pips.get(instrument, SPREAD_Pips["DEFAULT"])
    spread_cost = spread_in_pips * pip_size
    print(f"   -> Instrument: {instrument} | Pip Size: {pip_size:.4f} | Spread: {spread_in_pips} pips ({spread_cost:.4f})")
    
    return TIMEFRAME_PRESETS[timeframe_key], spread_cost

# --- WORKER FUNCTION (WITH THE CRITICAL FIX) ---
def process_chunk_task(task_indices):
    """
    The "Producer" worker function, executed in parallel by the Pool.
    
    It receives the start and end indices for its assigned data slice. It accesses the
    large, shared DataFrame via a process-global variable, calculates the correct
    processing limit for its chunk (handling the final, shorter chunk), and then
    calls the high-speed Numba simulation.
    """
    start_index, end_index = task_indices
    
    global worker_df, worker_config, worker_spread_cost, worker_max_lookforward
    
    # Create the overlapping data slice this worker needs.
    df_slice = worker_df.iloc[start_index:end_index]

    # --- THE CRITICAL FIX IS HERE ---
    # The processing limit must be calculated dynamically to handle the final chunk,
    # which may be shorter than INPUT_CHUNK_SIZE. It's the smaller of the chunk size
    # or the number of candles remaining in the *entire* dataframe from this chunk's start.
    processing_limit = min(INPUT_CHUNK_SIZE, len(worker_df) - start_index)
    
    # We must also ensure our processing_limit isn't larger than the non-overlapping
    # part of the slice we've been given. This is a safety check.
    processing_limit = min(processing_limit, len(df_slice) - worker_max_lookforward)
    if processing_limit <= 0:
        return [] # This chunk is too small to process (it's all lookahead data).

    # Convert the slice to NumPy arrays for Numba.
    close = df_slice["close"].values.astype(np.float64)
    high = df_slice["high"].values.astype(np.float64)
    low = df_slice["low"].values.astype(np.float64)
    timestamps = df_slice["time"].values.astype('datetime64[ns]').astype(np.int64)
    sl_ratios = worker_config["SL_RATIOS"].astype(np.float64)
    tp_ratios = worker_config["TP_RATIOS"].astype(np.float64)
    
    return find_winning_trades_numba(
        close, high, low, timestamps, sl_ratios, tp_ratios, worker_max_lookforward, worker_spread_cost,
        processing_limit  # Pass the correctly calculated, dynamic limit.
    )

# --- MAIN PROCESSING ORCHESTRATOR ---
def process_file_pipelined(input_file, output_file, config, spread_cost):
    """
    Orchestrates data generation for a single file using a pipelined, ordered,
    producer-consumer model for maximum speed, memory safety, and stability.

    This function acts as the "Manager" and "Consumer." It loads the data, prepares
    a list of tasks (chunk indices), and then creates a Pool of "Producer" workers.
    It uses the `pool.imap()` method to ensure that while workers process chunks in
    parallel, the results are consumed sequentially in the correct chronological order.
    This guarantees both high CPU utilization and an ordered output file, while the
    single consumer loop prevents memory overloads.

    Args:
        input_file (str): Full path to the raw input CSV.
        output_file (str): Full path where the Bronze data should be saved.
        config (dict): Configuration dictionary from TIMEFRAME_PRESETS.
        spread_cost (float): The calculated spread cost.

    Returns:
        str: A status message indicating success or failure.
    """
    filename = os.path.basename(input_file)
    print(f"\n[INFO] Starting pipelined processing for {filename}...")
    try:
        df = pd.read_csv(input_file, sep=None, engine="python", header=None)
        df.columns = ["time", "open", "high", "low", "close", "volume"][:df.shape[1]]
        df["time"] = pd.to_datetime(df["time"])
        df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].apply(pd.to_numeric)
    except Exception as e:
        return f"[ERROR] Failed to load or parse {filename}: {e}"

    if os.path.exists(output_file):
        os.remove(output_file)

    max_lookforward = config["MAX_LOOKFORWARD"]
    
    # 1. Prepare tasks (now just start/end indices)
    tasks = []
    for i in range(0, len(df), INPUT_CHUNK_SIZE):
        start_index = i
        end_index = i + INPUT_CHUNK_SIZE + max_lookforward
        if start_index < len(df):
             tasks.append((start_index, end_index))

    if not tasks:
        return "[INFO] No processable chunks found."

    profitable_trades_accumulator = []
    total_trades_found = 0
    is_first_write = True
    
    # 2. Define arguments for the worker initializer
    pool_init_args = (df, config, spread_cost, max_lookforward)
    
    # 3. Create the Pool, passing the initializer and its arguments
    with Pool(processes=MAX_CPU_USAGE, initializer=init_worker, initargs=pool_init_args) as pool:
        
        # 4. Use imap() for ordered results
        results_iterator = pool.imap(process_chunk_task, tasks)
        
        # 5. Consume results sequentially in the main process
        for results_list in tqdm(results_iterator, total=len(tasks), desc=f"Simulating Chunks for {filename}"):
            if results_list:
                profitable_trades_accumulator.extend(results_list)
            
            # 6. Memory-safe write to disk
            if len(profitable_trades_accumulator) >= OUTPUT_CHUNK_SIZE:
                chunk_df = pd.DataFrame(profitable_trades_accumulator, columns=["entry_time", "trade_type", "entry_price", "sl_price", "tp_price", "sl_ratio", "tp_ratio", "exit_time"])
                chunk_df['entry_time'] = pd.to_datetime(chunk_df['entry_time'], unit='ns')
                chunk_df['exit_time'] = pd.to_datetime(chunk_df['exit_time'], unit='ns')
                chunk_df['trade_type'] = chunk_df['trade_type'].map({1: 'buy', -1: 'sell'})
                chunk_df['outcome'] = 'win'
                
                chunk_df.to_csv(output_file, mode='a', header=is_first_write, index=False)
                total_trades_found += len(chunk_df)
                profitable_trades_accumulator.clear()
                is_first_write = False

    # Final save for any remaining trades
    if profitable_trades_accumulator:
        final_chunk_df = pd.DataFrame(profitable_trades_accumulator, columns=["entry_time", "trade_type", "entry_price", "sl_price", "tp_price", "sl_ratio", "tp_ratio", "exit_time"])
        final_chunk_df['entry_time'] = pd.to_datetime(final_chunk_df['entry_time'], unit='ns')
        final_chunk_df['exit_time'] = pd.to_datetime(final_chunk_df['exit_time'], unit='ns')
        final_chunk_df['trade_type'] = final_chunk_df['trade_type'].map({1: 'buy', -1: 'sell'})
        final_chunk_df['outcome'] = 'win'
        final_chunk_df.to_csv(output_file, mode='a', header=is_first_write, index=False)
        total_trades_found += len(final_chunk_df)

    if total_trades_found == 0:
        return f"No trades found for {filename}."
    return f"SUCCESS: {total_trades_found} trades found in {filename}."


if __name__ == "__main__":
    start_time = time.time()
    
    core_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.abspath(os.path.join(core_dir, '..', 'raw_data'))
    bronze_data_dir = os.path.abspath(os.path.join(core_dir, '..', 'bronze_data'))
    os.makedirs(bronze_data_dir, exist_ok=True)
    
    print("Warming up Numba JIT compiler...")
    find_winning_trades_numba(np.random.rand(10), np.random.rand(10), np.random.rand(10), np.random.randint(0, 10, 10, dtype=np.int64), np.random.rand(2), np.random.rand(2), 1, 0.0001, 10)
    print("[SUCCESS] Numba is ready.")

    files_to_process = []
    target_file_arg = sys.argv[1] if len(sys.argv) > 1 else None
    
    if target_file_arg:
        # Targeted Mode for the orchestrator
        print(f"[TARGET] Targeted Mode: Processing single file '{target_file_arg}'")
        if not os.path.exists(os.path.join(raw_data_dir, target_file_arg)):
            print(f"[ERROR] Target file not found in raw_data directory: {target_file_arg}")
        else:
            files_to_process = [target_file_arg]
    else:
        # Interactive Mode for manual runs
        print("[SCAN] Interactive Mode: Scanning for all new files...")
        try:
            all_raw_files = sorted([f for f in os.listdir(raw_data_dir) if f.endswith('.csv')])
            new_files = [f for f in all_raw_files if not os.path.exists(os.path.join(bronze_data_dir, f))]
            
            if not new_files:
                print("[INFO] No new files to process.")
            else:
                print("\n--- Select File(s) to Process ---")
                for i, f in enumerate(new_files):
                    print(f"  [{i+1}] {f}")
                print("\nYou can select multiple files by entering numbers separated by commas (e.g., 1,3,5)")
                
                user_input = input("Enter number(s) to process: ").strip()
                if user_input:
                    try:
                        selected_indices = [int(i.strip()) - 1 for i in user_input.split(',')]
                        valid_indices = sorted(list(set(idx for idx in selected_indices if 0 <= idx < len(new_files))))
                        files_to_process = [new_files[idx] for idx in valid_indices]
                    except ValueError:
                        print("[ERROR] Invalid input. Please enter numbers separated by commas.")
        except FileNotFoundError:
            print(f"[ERROR] The directory '{raw_data_dir}' was not found.")

    if not files_to_process:
        print("[INFO] No files selected or found for processing.")
    else:
        print(f"\n[INFO] Queued {len(files_to_process)} file(s) for processing: {files_to_process}")
        
        # Main Execution Loop: Processes selected files serially (one after another) for stability.
        for filename in files_to_process:
            config, spread_cost = get_config_from_filename(filename)
            if config:
                input_path = os.path.join(raw_data_dir, filename)
                output_path = os.path.join(bronze_data_dir, filename)
                
                result = process_file_pipelined(input_path, output_path, config, spread_cost)
                
                print("\n" + "="*50 + f"\nSummary for {filename}:")
                print(result)
            else:
                print(f"[ERROR] Could not generate configuration for {filename}. Skipping.")

    end_time = time.time()
    print(f"\nBronze data generation complete. Total time: {end_time - start_time:.2f} seconds.")
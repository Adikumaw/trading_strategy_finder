# bronze_data_generator.py (V4 - No Post-Processing Delay)

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
"""

import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import time
from numba import njit
import sys # <-- IMPORT SYS MODULE

# --- GLOBAL CONFIGURATION ---

# Sets the maximum number of CPU cores to use for multiprocessing.
# Leaves 2 cores free to ensure system responsiveness.
MAX_CPU_USAGE = max(1, cpu_count() - 2)

# The number of discovered winning trades to accumulate in memory before
# flushing them to the output CSV file. This is a key memory-saving feature.
OUTPUT_CHUNK_SIZE = 1_000_000

# The number of candles from the input file to process in each batch passed
# to the Numba JIT-compiled function. This is a performance tuning parameter.
INPUT_CHUNK_SIZE = 10000

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
        print(f"⚠️ Could not determine timeframe or instrument for {filename}. Skipping.")
        return None, None
        
    instrument, timeframe_num = match.group(1), match.group(2)
    timeframe_key = f"{timeframe_num}m"
    
    # Check if a preset exists for the detected timeframe
    if timeframe_key not in TIMEFRAME_PRESETS:
        print(f"⚠️ No preset for timeframe '{timeframe_key}' in {filename}. Skipping.")
        return None, None
        
    print(f"✅ Config '{timeframe_key}' detected for {filename}.")
    
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

def process_file(task_id, input_file, output_file, config, spread_cost):
    """
    Orchestrates the entire data generation process for a single input file.

    It reads the raw data, processes it in memory-safe chunks, calls the
    high-speed Numba function for each chunk, and saves the results to a
    CSV file intermittently to keep memory usage low.

    Args:
        task_id (int): The ID of the worker process, used for positioning the progress bar.
        input_file (str): The full path to the raw input CSV file.
        output_file (str): The full path where the Bronze data should be saved.
        config (dict): The configuration dictionary from TIMEFRAME_PRESETS.
        spread_cost (float): The calculated spread cost for this instrument.

    Returns:
        str: A status message indicating success or failure and the number of trades found.
    """
    filename = os.path.basename(input_file)
    try:
        # Load the raw data file, handling potential delimiter issues
        df = pd.read_csv(input_file, sep=None, engine="python", header=None)
        # Assign standard column names
        df.columns = ["time", "open", "high", "low", "close", "volume"][:df.shape[1]]
        df["time"] = pd.to_datetime(df["time"])
        numeric_cols = ["open", "high", "low", "close"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
    except Exception as e:
        print(f"❌ Error loading or parsing {filename}: {e}")
        return f"Error: {filename}"

    profitable_trades_accumulator = []
    total_trades_found = 0
    # Ensure a clean start by removing any previous output file
    if os.path.exists(output_file):
        os.remove(output_file)
    is_first_chunk = True

    max_lookforward = config["MAX_LOOKFORWARD"]
    
    # --- Main Processing Loop: Iterate through the input dataframe in chunks ---
    for i in tqdm(range(0, len(df), INPUT_CHUNK_SIZE), desc=f"Processing {filename}", position=task_id, leave=True):
        start_index = i
        end_index = i + INPUT_CHUNK_SIZE
        
        # Create an OVERLAPPING slice to provide lookahead data for the Numba function.
        # This ensures the simulation for the last candle in the main chunk has enough
        # future data to check against.
        overlap_end_index = end_index + max_lookforward
        df_slice_with_overlap = df.iloc[start_index:overlap_end_index]
        
        # Convert the chunk's data to NumPy arrays for Numba compatibility and speed
        close = df_slice_with_overlap["close"].values.astype(np.float64)
        high = df_slice_with_overlap["high"].values.astype(np.float64)
        low = df_slice_with_overlap["low"].values.astype(np.float64)
        timestamps = df_slice_with_overlap["time"].values.astype('datetime64[ns]').astype(np.int64)
        sl_ratios = config["SL_RATIOS"].astype(np.float64)
        tp_ratios = config["TP_RATIOS"].astype(np.float64)

        # Execute the high-performance Numba function on the prepared data chunk
        results_list = find_winning_trades_numba(
            close, high, low, timestamps, sl_ratios, tp_ratios, max_lookforward, spread_cost,
            # Tell Numba how many candles to actually process (the non-overlap part)
            processing_limit=len(df.iloc[start_index:end_index])
        )

        if results_list:
            profitable_trades_accumulator.extend(results_list)

        # --- Memory-Safe Output: Save results to disk when accumulator is full ---
        if len(profitable_trades_accumulator) >= OUTPUT_CHUNK_SIZE:
            # Convert the raw list of tuples into a formatted pandas DataFrame
            chunk_df = pd.DataFrame(profitable_trades_accumulator, columns=["entry_time", "trade_type", "entry_price", "sl_price", "tp_price", "sl_ratio", "tp_ratio", "exit_time"])
            chunk_df['entry_time'] = pd.to_datetime(chunk_df['entry_time'], unit='ns')
            chunk_df['exit_time'] = pd.to_datetime(chunk_df['exit_time'], unit='ns')
            chunk_df['trade_type'] = chunk_df['trade_type'].map({1: 'buy', -1: 'sell'})
            chunk_df['outcome'] = 'win'
            
            # Append the chunk to the output CSV file
            chunk_df.to_csv(output_file, mode='a', header=is_first_chunk, index=False)
            total_trades_found += len(chunk_df)
            profitable_trades_accumulator.clear() # Clear memory
            is_first_chunk = False # Subsequent chunks will not write a header

    # --- Final Save: Save any remaining trades after the main loop finishes ---
    if profitable_trades_accumulator:
        final_chunk_df = pd.DataFrame(profitable_trades_accumulator, columns=["entry_time", "trade_type", "entry_price", "sl_price", "tp_price", "sl_ratio", "tp_ratio", "exit_time"])
        final_chunk_df['entry_time'] = pd.to_datetime(final_chunk_df['entry_time'], unit='ns')
        final_chunk_df['exit_time'] = pd.to_datetime(final_chunk_df['exit_time'], unit='ns')
        final_chunk_df['trade_type'] = final_chunk_df['trade_type'].map({1: 'buy', -1: 'sell'})
        final_chunk_df['outcome'] = 'win'
        
        final_chunk_df.to_csv(output_file, mode='a', header=is_first_chunk, index=False)
        total_trades_found += len(final_chunk_df)

    if total_trades_found == 0:
        return f"No trades found for {filename}."

    return f"SUCCESS: {total_trades_found} trades found in {filename}."

if __name__ == "__main__":
    """
    Main execution block.
    
    This script can be run in two modes:
    1. Discovery Mode (no arguments): Scans the `raw_data` directory and processes all new files.
       Example: `python scripts/bronze_data_generator.py`
       
    2. Targeted Mode (one argument): Processes only the single file specified on the command line.
       Example: `python scripts/bronze_data_generator.py XAUUSD15.csv`
    """
    start_time = time.time()
    
    # --- Define Project Directory Structure ---
    core_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.abspath(os.path.join(core_dir, '..', 'raw_data'))
    bronze_data_dir = os.path.abspath(os.path.join(core_dir, '..', 'bronze_data'))
    os.makedirs(bronze_data_dir, exist_ok=True)
    
    # --- Numba JIT Warm-up ---
    # The first time a Numba function is called, it needs to compile.
    # We run it once with dummy data so the compilation delay doesn't affect
    # the timing of the actual first processing task.
    print("Warming up Numba JIT compiler... (this may take a moment on first run)")
    find_winning_trades_numba(np.random.rand(10), np.random.rand(10), np.random.rand(10), np.random.randint(0, 10, 10, dtype=np.int64), np.random.rand(2), np.random.rand(2), 1, 0.0001, 10)
    print("✅ Numba is ready.")

    # --- NEW: DUAL-MODE FILE DISCOVERY LOGIC ---
    target_file_arg = sys.argv[1] if len(sys.argv) > 1 else None
    
    if target_file_arg:
        # --- Targeted Mode ---
        print(f"🎯 Targeted Mode: Processing single file '{target_file_arg}'")
        if not os.path.exists(os.path.join(raw_data_dir, target_file_arg)):
            print(f"❌ Error: Target file not found in raw_data directory: {target_file_arg}")
            raw_files = []
        else:
            raw_files = [target_file_arg]
    else:
        # --- Discovery Mode (Default) ---
        print("🔍 Discovery Mode: Scanning for all new files...")
        try:
            all_raw_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.csv')]
            # Filter out files that have already been processed
            raw_files = [f for f in all_raw_files if not os.path.exists(os.path.join(bronze_data_dir, f))]
        except FileNotFoundError:
            print(f"❌ Error: The directory '{raw_data_dir}' was not found.")
            raw_files = []

    if not raw_files: 
        print("ℹ️ No new files to process.")
    else:
        print(f"Found {len(raw_files)} file(s) to process...")
        
        # --- Configure Multiprocessing ---
        # If in targeted mode, don't ask, just use multiprocessing.
        if target_file_arg:
            use_multiprocessing = True
        else:
            use_multiprocessing = input("Use multiprocessing? (y/n): ").strip().lower() == 'y'
            
        num_processes = MAX_CPU_USAGE if use_multiprocessing else 1
        
        # --- Prepare Processing Tasks ---
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
            # --- Execute Processing ---
            effective_workers = min(num_processes, len(tasks))
            print(f"\n🚀 Starting processing with {effective_workers} workers.")
            if effective_workers > 1:
                # Use a multiprocessing Pool to execute tasks in parallel
                with Pool(processes=effective_workers) as pool:
                    results = pool.starmap(process_file, tasks)
            else:
                # Execute tasks sequentially in the main process
                results = [process_file(*task) for task in tasks]

            # --- Display Summary ---
            print("\n" + "="*50 + "\nProcessing Summary:")
            for res in results:
                print(res)

    end_time = time.time()
    print(f"\nBronze data generation complete. Total time: {end_time - start_time:.2f} seconds.")
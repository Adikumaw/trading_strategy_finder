# bronze_data_generator.py (V12 - Re-documented & Formatted)

"""
Bronze Layer: The Possibility Engine

This script serves as the foundational data generation layer for a quantitative
trading strategy discovery pipeline. Its primary purpose is to systematically
scan historical price data and generate a comprehensive dataset of every
conceivable winning trade based on a predefined grid of Stop-Loss (SL) and
Take-Profit (TP) ratios.

This generated "universe of possibilities" acts as the bedrock for all
subsequent analysis in the Silver and Gold layers. The script operates by
performing a brute-force simulation on every candlestick, testing thousands of
SL/TP combinations, and recording only those that would have resulted in a
profitable outcome.

Architectural Highlights:
- Numba JIT Compilation: The core simulation logic is heavily accelerated with
  Numba for C-like performance.
- Producer-Consumer Model: Utilizes an intra-file parallelism model where
  multiple worker processes ("producers") simulate trades on data chunks. A
  single main process ("consumer") writes the results to disk.
- Ordered & Memory-Safe: Employs `multiprocessing.Pool.imap()` to ensure
  results are processed in chronological order, preventing memory overload and
  guaranteeing a sorted output file.
- Cross-Platform Stability: Uses a worker initializer (`init_worker`) to share
  large, read-only data, a robust pattern that avoids data serialization
  issues, especially on Windows.
"""

import os
import re
import sys
import time
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from numba import njit
from tqdm import tqdm

# --- GLOBAL CONFIGURATION ---

# Sets the maximum number of CPU cores to use. Leaving 2 cores free ensures
# the system remains responsive for other tasks and background processes.
MAX_CPU_USAGE: int = max(1, cpu_count() - 2)

# The number of winning trades to buffer in memory before flushing to the CSV.
# This balances I/O overhead with memory consumption.
OUTPUT_CHUNK_SIZE: int = 500_000

# The number of candles from the input file to process in each batch. This is a
# key performance-tuning parameter for the simulation.
INPUT_CHUNK_SIZE: int = 10_000


# --- SPREAD CONFIGURATION ---

# Defines the estimated trading spread for various instruments in pips.
# A default value is used if a specific instrument is not listed. This ensures
# backtests are more realistic by accounting for transaction costs.
SPREAD_PIPS: Dict[str, float] = {
    "DEFAULT": 3.0, "EURUSD": 1.5, "GBPUSD": 2.0, "AUDUSD": 2.5,
    "USDJPY": 2.0, "USDCAD": 2.5, "XAUUSD": 20.0,
}


# --- TIMEFRAME PRESETS ---

# Defines simulation parameters for different chart timeframes.
# - SL_RATIOS: A NumPy array of stop-loss percentages to test (e.g., 0.001 = 0.1%).
# - TP_RATIOS: A NumPy array of take-profit percentages to test.
# - MAX_LOOKFORWARD: The max number of future candles to check for a trade's
#   outcome. This prevents simulations from running indefinitely and defines
#   the maximum holding period for a potential trade.
TIMEFRAME_PRESETS: Dict[str, Dict[str, Any]] = {
    "1m": {
        "SL_RATIOS": np.arange(0.0005, 0.0105, 0.0005),
        "TP_RATIOS": np.arange(0.0005, 0.0205, 0.0005),
        "MAX_LOOKFORWARD": 200,
    },
    "5m": {
        "SL_RATIOS": np.arange(0.001, 0.0155, 0.0005),
        "TP_RATIOS": np.arange(0.001, 0.0305, 0.0005),
        "MAX_LOOKFORWARD": 300,
    },
    "15m": {
        "SL_RATIOS": np.arange(0.002, 0.0255, 0.001),
        "TP_RATIOS": np.arange(0.002, 0.0505, 0.001),
        "MAX_LOOKFORWARD": 400,
    },
    "30m": {
        "SL_RATIOS": np.arange(0.003, 0.0355, 0.001),
        "TP_RATIOS": np.arange(0.003, 0.0705, 0.001),
        "MAX_LOOKFORWARD": 500,
    },
    "60m": {
        "SL_RATIOS": np.arange(0.005, 0.0505, 0.001),
        "TP_RATIOS": np.arange(0.005, 0.1005, 0.001),
        "MAX_LOOKFORWARD": 600,
    },
    "240m": {
        "SL_RATIOS": np.arange(0.010, 0.1005, 0.001),
        "TP_RATIOS": np.arange(0.010, 0.2005, 0.001),
        "MAX_LOOKFORWARD": 800,
    },
}


# --- WORKER-SPECIFIC GLOBAL VARIABLES ---
# These are populated in each worker process upon creation. This design avoids
# passing large objects via inter-process communication, which is slow and can
# be unstable, particularly on Windows due to differences in process forking.
worker_df: Optional[pd.DataFrame] = None
worker_config: Optional[Dict[str, Any]] = None
worker_spread_cost: Optional[float] = None
worker_max_lookforward: Optional[int] = None


def init_worker(
    df: pd.DataFrame,
    config: Dict[str, Any],
    spread_cost: float,
    max_lookforward: int,
) -> None:
    """
    Initializer for each worker process in the multiprocessing Pool.

    This function is called once per worker when the pool is spawned. It
    receives large, shared, read-only data objects and stores them in global
    variables within that specific worker's memory space. This prevents the
    need to pickle and transfer this data for every task, dramatically
    improving performance and stability.

    Args:
        df: The complete historical price data DataFrame.
        config: The configuration dictionary for the current timeframe.
        spread_cost: The calculated spread cost for the instrument.
        max_lookforward: The maximum number of candles to look forward.
    """
    global worker_df, worker_config, worker_spread_cost, worker_max_lookforward
    worker_df = df
    worker_config = config
    worker_spread_cost = spread_cost
    worker_max_lookforward = max_lookforward


# --- NUMBA-ACCELERATED CORE LOGIC ---

@njit
def find_winning_trades_numba(
    close_prices: np.ndarray,
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    timestamps: np.ndarray,
    sl_ratios: np.ndarray,
    tp_ratios: np.ndarray,
    max_lookforward: int,
    spread_cost: float,
    processing_limit: int,
) -> List[Tuple]:
    """
    Executes the core trade simulation logic at high speed using Numba.

    This is the performance-critical engine of the script. It iterates through
    a chunk of candlestick data and, for each candle, simulates every possible
    buy and sell trade defined by the grid of SL and TP ratios. It then looks
    forward in time to determine if a trade would have hit its TP before its
    SL, accounting for spread. Only profitable trades are recorded.

    Args:
        close_prices: Array of closing prices for the data chunk.
        high_prices: Array of high prices for the data chunk.
        low_prices: Array of low prices for the data chunk.
        timestamps: Array of timestamps (as int64) for each candle.
        sl_ratios: Array of stop-loss ratios to test.
        tp_ratios: Array of take-profit ratios to test.
        max_lookforward: Max number of future candles to check for an outcome.
        spread_cost: The spread cost, applied to TP levels for realism.
        processing_limit: The number of candles within the chunk to process,
                          excluding the overlapping lookahead data at the end.

    Returns:
        A list of tuples, where each tuple represents a single profitable
        trade with its key parameters.
    """
    all_profitable_trades = []
    # Loop only up to the processing_limit to avoid re-processing overlapping
    # candles from the next chunk.
    for i in range(processing_limit):
        entry_price = close_prices[i]
        entry_time = timestamps[i]

        # --- Simulate BUY trades ---
        for sl_r in sl_ratios:
            for tp_r in tp_ratios:
                sl_price = entry_price * (1 - sl_r)
                tp_price = entry_price * (1 + tp_r)
                limit = min(i + 1 + max_lookforward, len(close_prices))

                for j in range(i + 1, limit):
                    # Check for a win (TP hit), accounting for the spread cost.
                    if high_prices[j] >= (tp_price + spread_cost):
                        all_profitable_trades.append(
                            (entry_time, 1, entry_price, sl_price, tp_price,
                             sl_r, tp_r, timestamps[j])
                        )
                        break  # Win found, move to next SL/TP combo.
                    # Check for a loss (SL hit).
                    if low_prices[j] <= sl_price:
                        break  # Loss found, move to next SL/TP combo.

        # --- Simulate SELL trades ---
        for sl_r in sl_ratios:
            for tp_r in tp_ratios:
                sl_price = entry_price * (1 + sl_r)
                tp_price = entry_price * (1 - tp_r)
                limit = min(i + 1 + max_lookforward, len(close_prices))

                for j in range(i + 1, limit):
                    # Check for a win (TP hit), accounting for the spread cost.
                    if low_prices[j] <= (tp_price - spread_cost):
                        all_profitable_trades.append(
                            (entry_time, -1, entry_price, sl_price, tp_price,
                             sl_r, tp_r, timestamps[j])
                        )
                        break
                    # Check for a loss (SL hit).
                    if high_prices[j] >= sl_price:
                        break

    return all_profitable_trades


# --- HELPER & WORKER FUNCTIONS ---

def get_config_from_filename(filename: str) -> Tuple[Optional[Dict], Optional[float]]:
    """
    Parses a filename to extract instrument and timeframe, returning the config.

    Uses regex to identify the trading instrument (e.g., "EURUSD") and the
    chart timeframe (e.g., "15m") from the input CSV filename. It then
    calculates the appropriate spread cost and retrieves the simulation
    parameters from the `TIMEFRAME_PRESETS` dictionary.

    Args:
        filename: The name of the raw data file (e.g., "XAUUSD15.csv").

    Returns:
        A tuple containing the configuration dictionary and the calculated
        spread cost. Returns (None, None) if the filename cannot be parsed
        or no matching preset is found.
    """
    match = re.search(r"([A-Z0-9]+?)(\d+)\.csv$", filename)
    if not match:
        print(f"[WARNING] Could not parse timeframe or instrument from '{filename}'. Skipping.")
        return None, None

    instrument, timeframe_num = match.group(1), match.group(2)
    timeframe_key = f"{timeframe_num}m"

    if timeframe_key not in TIMEFRAME_PRESETS:
        print(f"[WARNING] No preset for timeframe '{timeframe_key}' in '{filename}'. Skipping.")
        return None, None

    print(f"[SUCCESS] Config '{timeframe_key}' detected for {filename}.")

    # Determine pip size for accurate spread calculation based on instrument type.
    # JPY pairs have pips at the 2nd decimal place.
    if "JPY" in instrument.upper():
        pip_size = 0.01
    # Metals (Gold, Silver) often priced to 2 decimal places.
    elif "XAU" in instrument.upper() or "XAG" in instrument.upper():
        pip_size = 0.01
    # Indices or other non-standard instruments.
    elif len(instrument) > 6 or any(char.isdigit() for char in instrument):
        pip_size = 0.1
    # Standard Forex pairs (e.g., EURUSD) have pips at the 4th decimal place.
    else:
        pip_size = 0.0001

    spread_in_pips = SPREAD_PIPS.get(instrument, SPREAD_PIPS["DEFAULT"])
    spread_cost = spread_in_pips * pip_size
    print(f"   -> Instrument: {instrument} | Pip Size: {pip_size:.4f} | "
          f"Spread: {spread_in_pips} pips ({spread_cost:.5f})")

    return TIMEFRAME_PRESETS[timeframe_key], spread_cost


def process_chunk_task(task_indices: Tuple[int, int]) -> List[Tuple]:
    """
    The "Producer" worker function, executed in parallel by the Pool.

    Receives start and end indices for its assigned data slice. It accesses the
    large, shared DataFrame via a process-global variable (set by `init_worker`),
    calculates the correct processing limit for its chunk, and then calls the
    high-speed Numba simulation function.

    Args:
        task_indices: A tuple containing the start and end index for the data slice.

    Returns:
        A list of profitable trade tuples found within its assigned chunk.
    """
    start_index, end_index = task_indices

    # Access worker-specific global variables. These are guaranteed to be non-None
    # because the initializer runs before any task.
    df_slice = worker_df.iloc[start_index:end_index]

    # --- CRITICAL LOGIC FOR HANDLING CHUNKS ---
    # The processing limit must be calculated dynamically to handle the final
    # data chunk, which is likely shorter than INPUT_CHUNK_SIZE.
    # It's the smaller of the chunk size or the total number of candles
    # remaining in the *entire* dataframe from this chunk's start.
    processing_limit = min(INPUT_CHUNK_SIZE, len(worker_df) - start_index)

    # A safety check to ensure we don't process into the lookahead-only portion
    # of the data slice.
    processing_limit = min(processing_limit, len(df_slice) - worker_max_lookforward)
    if processing_limit <= 0:
        return []  # This chunk is too small (it's all lookahead data).

    # Convert the pandas slice to NumPy arrays for Numba compatibility.
    close = df_slice["close"].values.astype(np.float64)
    high = df_slice["high"].values.astype(np.float64)
    low = df_slice["low"].values.astype(np.float64)
    timestamps = df_slice["time"].values.astype("datetime64[ns]").astype(np.int64)
    sl_ratios = worker_config["SL_RATIOS"].astype(np.float64)
    tp_ratios = worker_config["TP_RATIOS"].astype(np.float64)

    return find_winning_trades_numba(
        close, high, low, timestamps, sl_ratios, tp_ratios,
        worker_max_lookforward, worker_spread_cost, processing_limit
    )


# --- MAIN PROCESSING ORCHESTRATOR ---

def process_file_pipelined(
    input_file: str, output_file: str, config: Dict, spread_cost: float
) -> str:
    """
    Orchestrates data generation for a single file using a producer-consumer model.

    This function acts as the "Manager" and "Consumer". It loads the data,
    prepares a list of tasks (chunk indices), and creates a Pool of "Producer"
    workers. It uses `pool.imap()` to ensure that while workers process chunks
    in parallel, the results are consumed sequentially in chronological order.
    This guarantees both high CPU utilization and an ordered output file, while
    the single consumer loop prevents memory overloads.

    Args:
        input_file: Full path to the raw input CSV.
        output_file: Full path where the Bronze data should be saved.
        config: Configuration dictionary from TIMEFRAME_PRESETS.
        spread_cost: The calculated spread cost.

    Returns:
        A status message indicating success or failure.
    """
    filename = os.path.basename(input_file)
    print(f"\n[INFO] Starting pipelined processing for {filename}...")
    try:
        # Using sep=None allows pandas to auto-detect the delimiter, making it
        # robust to comma or tab-separated files. engine='python' is required.
        df = pd.read_csv(input_file, sep=None, engine="python", header=None)
        df.columns = ["time", "open", "high", "low", "close", "volume"][:df.shape[1]]
        df["time"] = pd.to_datetime(df["time"])
        df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].apply(pd.to_numeric)
    except Exception as e:
        return f"[ERROR] Failed to load or parse {filename}: {e}"

    if os.path.exists(output_file):
        os.remove(output_file)

    max_lookforward = config["MAX_LOOKFORWARD"]

    # 1. Prepare tasks (start and end indices for each data slice)
    tasks = []
    for i in range(0, len(df), INPUT_CHUNK_SIZE):
        start_index = i
        # Each slice must include the lookforward data for the last candle in
        # the main processing part of the chunk.
        end_index = i + INPUT_CHUNK_SIZE + max_lookforward
        if start_index < len(df):
            tasks.append((start_index, end_index))

    if not tasks:
        return "[INFO] No processable chunks found."

    profitable_trades_accumulator = []
    total_trades_found = 0
    is_first_write = True

    # 2. Define arguments for the worker initializer.
    pool_init_args = (df, config, spread_cost, max_lookforward)

    # 3. Create the Pool, passing the initializer and its arguments.
    with Pool(processes=MAX_CPU_USAGE, initializer=init_worker, initargs=pool_init_args) as pool:
        # 4. Use imap() for ordered results. This is key for chronological output.
        results_iterator = pool.imap(process_chunk_task, tasks)

        # 5. Consume results sequentially in the main process.
        for results_list in tqdm(results_iterator, total=len(tasks), desc=f"Simulating Chunks for {filename}"):
            if results_list:
                profitable_trades_accumulator.extend(results_list)

            # 6. Memory-safe write to disk when buffer is full.
            if len(profitable_trades_accumulator) >= OUTPUT_CHUNK_SIZE:
                chunk_df = pd.DataFrame(
                    profitable_trades_accumulator,
                    columns=["entry_time", "trade_type", "entry_price", "sl_price",
                             "tp_price", "sl_ratio", "tp_ratio", "exit_time"]
                )
                chunk_df['entry_time'] = pd.to_datetime(chunk_df['entry_time'], unit='ns')
                chunk_df['exit_time'] = pd.to_datetime(chunk_df['exit_time'], unit='ns')
                chunk_df['trade_type'] = np.where(chunk_df['trade_type'] == 1, 'buy', 'sell')
                chunk_df['outcome'] = 'win'

                chunk_df.to_csv(output_file, mode='a', header=is_first_write, index=False)
                total_trades_found += len(chunk_df)
                profitable_trades_accumulator.clear()
                is_first_write = False

    # 7. Final save for any remaining trades in the accumulator.
    if profitable_trades_accumulator:
        final_chunk_df = pd.DataFrame(
            profitable_trades_accumulator,
            columns=["entry_time", "trade_type", "entry_price", "sl_price",
                     "tp_price", "sl_ratio", "tp_ratio", "exit_time"]
        )
        final_chunk_df['entry_time'] = pd.to_datetime(final_chunk_df['entry_time'], unit='ns')
        final_chunk_df['exit_time'] = pd.to_datetime(final_chunk_df['exit_time'], unit='ns')
        final_chunk_df['trade_type'] = np.where(final_chunk_df['trade_type'] == 1, 'buy', 'sell')
        final_chunk_df['outcome'] = 'win'
        final_chunk_df.to_csv(output_file, mode='a', header=is_first_write, index=False)
        total_trades_found += len(final_chunk_df)

    if total_trades_found == 0:
        return f"No winning trades found for {filename} with the given parameters."
    return f"SUCCESS: {total_trades_found:,} trades found in {filename}."


def main() -> None:
    """
    Main execution function: handles file discovery, user interaction,
    and orchestrates the processing of each file.
    """
    start_time = time.time()

    core_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.abspath(os.path.join(core_dir, '..', 'raw_data'))
    bronze_data_dir = os.path.abspath(os.path.join(core_dir, '..', 'bronze_data'))
    os.makedirs(bronze_data_dir, exist_ok=True)

    # "Warm up" the Numba JIT compiler before processing real data. This
    # avoids the compilation overhead on the first actual data chunk.
    print("Warming up Numba JIT compiler...")
    find_winning_trades_numba(
        np.random.rand(10), np.random.rand(10), np.random.rand(10),
        np.random.randint(0, 10, 10, dtype=np.int64),
        np.random.rand(2), np.random.rand(2), 1, 0.0001, 10
    )
    print("[SUCCESS] Numba is ready.")

    files_to_process = []
    target_file_arg = sys.argv[1] if len(sys.argv) > 1 else None

    if target_file_arg:
        # Targeted Mode: Process a single file passed as a command-line argument.
        print(f"[TARGET] Targeted Mode: Processing '{target_file_arg}'")
        target_path = os.path.join(raw_data_dir, target_file_arg)
        if not os.path.exists(target_path):
            print(f"[ERROR] Target file not found: {target_path}")
        else:
            files_to_process = [target_file_arg]
    else:
        # Interactive Mode: Scan for new files and prompt the user for selection.
        print("[SCAN] Interactive Mode: Scanning for new files...")
        try:
            all_raw_files = sorted([f for f in os.listdir(raw_data_dir) if f.endswith('.csv')])
            bronze_files = os.listdir(bronze_data_dir)
            new_files = [f for f in all_raw_files if f not in bronze_files]

            if not new_files:
                print("[INFO] No new raw data files to process.")
            else:
                print("\n--- Select File(s) to Process ---")
                for i, f in enumerate(new_files):
                    print(f"  [{i+1}] {f}")
                print("\nSelect multiple files with comma-separated numbers (e.g., 1,3,5)")

                user_input = input("Enter number(s) to process: ").strip()
                if user_input:
                    try:
                        selected_indices = [int(i.strip()) - 1 for i in user_input.split(',')]
                        valid_indices = sorted(list(set(
                            idx for idx in selected_indices if 0 <= idx < len(new_files)
                        )))
                        files_to_process = [new_files[idx] for idx in valid_indices]
                    except ValueError:
                        print("[ERROR] Invalid input. Please enter numbers only.")
        except FileNotFoundError:
            print(f"[ERROR] The raw data directory was not found at: {raw_data_dir}")

    if not files_to_process:
        print("[INFO] No files selected or found for processing.")
    else:
        print(f"\n[QUEUE] Queued {len(files_to_process)} file(s): {files_to_process}")

        # Main execution loop processes files serially for stability and
        # clear logging, while parallelizing the work *within* each file.
        for filename in files_to_process:
            config, spread_cost = get_config_from_filename(filename)
            if config:
                input_path = os.path.join(raw_data_dir, filename)
                output_path = os.path.join(bronze_data_dir, filename)
                result = process_file_pipelined(input_path, output_path, config, spread_cost)
                print("\n" + "="*60)
                print(f"Summary for {filename}: {result}")
                print("="*60)
            else:
                print(f"[ERROR] Could not generate configuration for {filename}. Skipping.")

    end_time = time.time()
    print(f"\nBronze data generation complete. Total time: {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
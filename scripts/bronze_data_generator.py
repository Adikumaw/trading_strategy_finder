# bronze_data_generator.py (V15 - Final Polish & Documentation)

"""
Bronze Layer: The Possibility Engine

This script serves as the foundational data generation layer for the quantitative
trading strategy discovery pipeline. Its primary purpose is to systematically
scan historical price data and generate a comprehensive dataset of every
conceivable winning trade based on a predefined grid of Stop-Loss (SL) and
Take-Profit (TP) ratios.

This generated "universe of possibilities" acts as the bedrock for all
subsequent analysis. The script operates by performing a brute-force simulation
on every candlestick, testing thousands of SL/TP combinations, and recording
only those that would have resulted in a profitable outcome.

Architectural Highlights:
- Parquet Output: Saves data in the highly efficient, columnar Parquet format,
  which is significantly faster for downstream scripts to read than CSV.
- Numba JIT Compilation: The core simulation logic is heavily accelerated with
  Numba for C-like performance.
- Producer-Consumer Model: Utilizes an intra-file parallelism model where
  multiple worker processes ("producers") simulate trades on data chunks. A
  single main process ("consumer") writes the results to disk.
- Ordered & Memory-Safe: Employs `multiprocessing.Pool.imap()` to ensure
  results are processed in chronological order, preventing memory overload.
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

# This script requires the 'pyarrow' library for Parquet functionality.
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print("[FATAL] 'pyarrow' library not found. Please run 'pip install pyarrow' to continue.")
    sys.exit(1)


# --- GLOBAL CONFIGURATION ---
MAX_CPU_USAGE: int = max(1, cpu_count() - 2)
OUTPUT_CHUNK_SIZE: int = 500_000
INPUT_CHUNK_SIZE: int = 5_000

# --- SPREAD CONFIGURATION ---
SPREAD_PIPS: Dict[str, float] = {
    "DEFAULT": 3.0, "EURUSD": 1.5, "GBPUSD": 2.0, "AUDUSD": 2.5,
    "USDJPY": 2.0, "USDCAD": 2.5, "XAUUSD": 20.0,
}

# --- TIMEFRAME PRESETS ---
TIMEFRAME_PRESETS: Dict[str, Dict[str, Any]] = {
    "1m": {"SL_RATIOS": np.arange(0.0005, 0.0105, 0.0005), "TP_RATIOS": np.arange(0.0005, 0.0205, 0.0005), "MAX_LOOKFORWARD": 200},
    "5m": {"SL_RATIOS": np.arange(0.001, 0.0155, 0.0005), "TP_RATIOS": np.arange(0.001, 0.0305, 0.0005), "MAX_LOOKFORWARD": 300},
    "15m": {"SL_RATIOS": np.arange(0.002, 0.0255, 0.001), "TP_RATIOS": np.arange(0.002, 0.0505, 0.001), "MAX_LOOKFORWARD": 400},
    "30m": {"SL_RATIOS": np.arange(0.003, 0.0355, 0.001), "TP_RATIOS": np.arange(0.003, 0.0705, 0.001), "MAX_LOOKFORWARD": 500},
    "60m": {"SL_RATIOS": np.arange(0.005, 0.0505, 0.001), "TP_RATIOS": np.arange(0.005, 0.1005, 0.001), "MAX_LOOKFORWARD": 600},
    "240m": {"SL_RATIOS": np.arange(0.010, 0.1005, 0.001), "TP_RATIOS": np.arange(0.010, 0.2005, 0.001), "MAX_LOOKFORWARD": 800},
}

# --- WORKER-SPECIFIC GLOBAL VARIABLES ---
worker_df: Optional[pd.DataFrame] = None
worker_config: Optional[Dict[str, Any]] = None
worker_spread_cost: Optional[float] = None
worker_max_lookforward: Optional[int] = None


def init_worker(df: pd.DataFrame, config: Dict, spread_cost: float, max_lookforward: int) -> None:
    """Initializer for each worker process in the multiprocessing Pool."""
    global worker_df, worker_config, worker_spread_cost, worker_max_lookforward
    worker_df, worker_config, worker_spread_cost, worker_max_lookforward = \
        df, config, spread_cost, max_lookforward


# --- NUMBA-ACCELERATED CORE LOGIC ---
@njit
def find_winning_trades_numba(
    close_prices: np.ndarray, high_prices: np.ndarray, low_prices: np.ndarray, timestamps: np.ndarray,
    sl_ratios: np.ndarray, tp_ratios: np.ndarray, max_lookforward: int, spread_cost: float,
    processing_limit: int
) -> List[Tuple]:
    """Executes the core trade simulation logic at high speed using Numba."""
    all_profitable_trades = []
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
                    if high_prices[j] >= (tp_price + spread_cost):
                        all_profitable_trades.append((entry_time, 1, entry_price, sl_price, tp_price, sl_r, tp_r, timestamps[j]))
                        break
                    if low_prices[j] <= sl_price:
                        break

        # --- Simulate SELL trades ---
        for sl_r in sl_ratios:
            for tp_r in tp_ratios:
                sl_price = entry_price * (1 + sl_r)
                tp_price = entry_price * (1 - tp_r)
                limit = min(i + 1 + max_lookforward, len(close_prices))
                for j in range(i + 1, limit):
                    if low_prices[j] <= (tp_price - spread_cost):
                        all_profitable_trades.append((entry_time, -1, entry_price, sl_price, tp_price, sl_r, tp_r, timestamps[j]))
                        break
                    if high_prices[j] >= sl_price:
                        break
    return all_profitable_trades


# --- HELPER & WORKER FUNCTIONS ---

def get_config_from_filename(filename: str) -> Tuple[Optional[Dict], Optional[float]]:
    """Parses a filename to extract instrument and timeframe, returning the config."""
    match = re.search(r"([A-Z0-9_]+?)(\d+)\.csv$", filename, re.IGNORECASE)
    if not match:
        print(f"[WARN] Could not parse timeframe or instrument from '{filename}'. Skipping.")
        return None, None

    instrument, timeframe_num = match.group(1).replace("_", ""), match.group(2)
    timeframe_key = f"{timeframe_num}m"
    if timeframe_key not in TIMEFRAME_PRESETS:
        print(f"[WARN] No preset for timeframe '{timeframe_key}' in '{filename}'. Skipping.")
        return None, None

    print(f"[INFO] Config '{timeframe_key}' detected for {instrument.upper()}.")
    pip_size = 0.01 if "JPY" in instrument.upper() or "XAU" in instrument.upper() else 0.0001
    spread_in_pips = SPREAD_PIPS.get(instrument.upper(), SPREAD_PIPS["DEFAULT"])
    spread_cost = spread_in_pips * pip_size
    print(f"   -> Instrument: {instrument.upper()} | Pip Size: {pip_size:.4f} | "
          f"Spread: {spread_in_pips} pips ({spread_cost:.5f})")
    return TIMEFRAME_PRESETS[timeframe_key], spread_cost


def process_chunk_task(task_indices: Tuple[int, int]) -> List[Tuple]:
    """The "Producer" worker function, executed in parallel by the Pool."""
    start_index, end_index = task_indices
    df_slice = worker_df.iloc[start_index:end_index]
    processing_limit = min(INPUT_CHUNK_SIZE, len(worker_df) - start_index)
    processing_limit = min(processing_limit, len(df_slice) - worker_max_lookforward)
    if processing_limit <= 0: return []

    close, high, low = df_slice["close"].values, df_slice["high"].values, df_slice["low"].values
    timestamps = df_slice["time"].values.astype("datetime64[ns]").astype(np.int64)
    sl_ratios, tp_ratios = worker_config["SL_RATIOS"], worker_config["TP_RATIOS"]
    return find_winning_trades_numba(
        close, high, low, timestamps, sl_ratios, tp_ratios,
        worker_max_lookforward, worker_spread_cost, processing_limit
    )


def _create_df_from_results(data: List[Tuple]) -> pd.DataFrame:
    """Converts a list of raw trade tuples into a structured, optimized DataFrame."""
    if not data: return pd.DataFrame()
    df = pd.DataFrame(data, columns=[
        "entry_time", "trade_type", "entry_price", "sl_price", "tp_price",
        "sl_ratio", "tp_ratio", "exit_time"
    ])
    df['entry_time'] = pd.to_datetime(df['entry_time'], unit='ns')
    df['exit_time'] = pd.to_datetime(df['exit_time'], unit='ns')
    df['trade_type'] = np.where(df['trade_type'] == 1, 'buy', 'sell')
    df['trade_type'] = df['trade_type'].astype('category')
    df['outcome'] = 'win'
    df['outcome'] = df['outcome'].astype('category')
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    return df


# --- MAIN PROCESSING ORCHESTRATOR ---
def process_file_pipelined(
    input_file: str, output_file: str, config: Dict, spread_cost: float
) -> str:
    """Orchestrates data generation for a single file, writing to Parquet."""
    filename = os.path.basename(input_file)
    print(f"\n[INFO] Starting pipelined processing for {filename}...")
    try:
        df = pd.read_csv(input_file, sep=None, engine="python", header=None)
        if df.shape[1] < 5: return f"[ERROR] Invalid file format: Expected at least 5 columns."
        df.columns = ["time", "open", "high", "low", "close", "volume"][:df.shape[1]]
        
        # ### <<< CHANGE: Added more robust data cleaning and validation.
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        numeric_cols = ["open", "high", "low", "close"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        initial_rows = len(df)
        df.dropna(subset=['time'] + numeric_cols, inplace=True)
        if len(df) < initial_rows:
            print(f"[WARN] Dropped {initial_rows - len(df)} rows with invalid data.")
        df = df.sort_values('time').reset_index(drop=True)

    except Exception as e:
        return f"[ERROR] Failed to load or parse '{filename}': {e}"
    
    max_lookforward = config["MAX_LOOKFORWARD"]
    if len(df) <= max_lookforward:
        return f"[ERROR] Not enough data to perform lookforward of {max_lookforward} candles."

    if os.path.exists(output_file):
        print(f"[WARN] Output file {os.path.basename(output_file)} already exists. Overwriting.")
        os.remove(output_file)

    tasks = [(i, i + INPUT_CHUNK_SIZE + max_lookforward) for i in range(0, len(df), INPUT_CHUNK_SIZE)]
    if not tasks: return "[INFO] No processable chunks found."

    profitable_trades_accumulator = []
    total_trades_found = 0
    writer = None

    # ### <<< CHANGE: Added a try...finally block to guarantee the writer is closed.
    # This prevents corrupted/incomplete Parquet files if an error occurs mid-run.
    try:
        pool_init_args = (df, config, spread_cost, max_lookforward)
        with Pool(processes=MAX_CPU_USAGE, initializer=init_worker, initargs=pool_init_args) as pool:
            results_iterator = pool.imap(process_chunk_task, tasks)
            for results_list in tqdm(results_iterator, total=len(tasks), desc=f"Simulating Chunks for {filename}"):
                if results_list:
                    profitable_trades_accumulator.extend(results_list)
                if len(profitable_trades_accumulator) >= OUTPUT_CHUNK_SIZE:
                    chunk_df = _create_df_from_results(profitable_trades_accumulator)
                    if not chunk_df.empty:
                        table = pa.Table.from_pandas(chunk_df, preserve_index=False)
                        if writer is None:
                            writer = pq.ParquetWriter(output_file, table.schema)
                        writer.write_table(table)
                        total_trades_found += len(chunk_df)
                    profitable_trades_accumulator.clear()
        if profitable_trades_accumulator:
            final_chunk_df = _create_df_from_results(profitable_trades_accumulator)
            if not final_chunk_df.empty:
                table = pa.Table.from_pandas(final_chunk_df, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(output_file, table.schema)
                writer.write_table(table)
                total_trades_found += len(final_chunk_df)
    finally:
        if writer:
            writer.close()

    if total_trades_found == 0:
        if os.path.exists(output_file): os.remove(output_file)
        return f"No winning trades found for {filename} with the given parameters."
    
    return f"SUCCESS: Found {total_trades_found:,} trades. Saved to {os.path.basename(output_file)}."


def _select_files_interactively(raw_data_dir: str, bronze_data_dir: str) -> List[str]:
    """Scans for new files and prompts the user to select which ones to process."""
    print("[INFO] Interactive Mode: Scanning for new files...")
    try:
        all_raw_files = sorted([f for f in os.listdir(raw_data_dir) if f.endswith('.csv')])
        bronze_bases = {os.path.splitext(f)[0] for f in os.listdir(bronze_data_dir) if f.endswith('.parquet')}
        new_files = [f for f in all_raw_files if os.path.splitext(f)[0] not in bronze_bases]

        if not new_files:
            print("[INFO] No new raw data files to process.")
            return []

        print("\n--- Select File(s) to Process ---")
        for i, f in enumerate(new_files): print(f"  [{i+1}] {f}")
        print("  [a] Process All New Files")
        print("\nEnter selection (e.g., 1,3 or a):")
        user_input = input("> ").strip().lower()
        if not user_input: return []
        if user_input == 'a': return new_files

        selected_files = []
        try:
            indices = {int(i.strip()) - 1 for i in user_input.split(',')}
            for idx in sorted(indices):
                if 0 <= idx < len(new_files): selected_files.append(new_files[idx])
                else: print(f"[WARN] Invalid selection '{idx + 1}' ignored.")
            return selected_files
        except ValueError:
            print("[ERROR] Invalid input. Please enter numbers (e.g., 1,3) or 'a'.")
            return []
    except FileNotFoundError:
        print(f"[ERROR] The raw data directory was not found at: {raw_data_dir}")
        return []


def main() -> None:
    """Main execution function."""
    start_time = time.time()
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    RAW_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'raw_data'))
    BRONZE_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'bronze_data'))
    os.makedirs(BRONZE_DATA_DIR, exist_ok=True)

    print("--- Bronze Layer: The Possibility Engine (Parquet Edition) ---")
    print(f"Using up to {MAX_CPU_USAGE} CPU cores for simulation.")
    
    print("[INFO] Initializing simulation engine (Numba JIT)...")
    find_winning_trades_numba(np.random.rand(10), np.random.rand(10), np.random.rand(10), np.random.randint(0, 10, 10, dtype=np.int64), np.random.rand(2), np.random.rand(2), 1, 0.0001, 10)
    print("[SUCCESS] Engine is ready.")

    files_to_process = []
    target_file_arg = sys.argv[1] if len(sys.argv) > 1 else None
    if target_file_arg:
        print(f"\n[INFO] Targeted Mode: Processing '{target_file_arg}'")
        if not os.path.exists(os.path.join(RAW_DATA_DIR, target_file_arg)):
            print(f"[ERROR] Target file not found: {os.path.join(RAW_DATA_DIR, target_file_arg)}")
        else:
            files_to_process = [target_file_arg]
    else:
        files_to_process = _select_files_interactively(RAW_DATA_DIR, BRONZE_DATA_DIR)

    if not files_to_process:
        print("\n[INFO] No files selected or found for processing. Exiting.")
    else:
        print(f"\n[QUEUE] Queued {len(files_to_process)} file(s): {', '.join(files_to_process)}")
        for filename in files_to_process:
            config, spread_cost = get_config_from_filename(filename)
            if config:
                input_path = os.path.join(RAW_DATA_DIR, filename)
                output_path = os.path.join(BRONZE_DATA_DIR, filename.replace('.csv', '.parquet'))
                result = process_file_pipelined(input_path, output_path, config, spread_cost)
                print("\n" + "="*80)
                print(f"SUMMARY FOR {filename}: {result}")
                print("="*80)
            else:
                print(f"[ERROR] Could not generate configuration for {filename}. Skipping.")

    end_time = time.time()
    print(f"\nBronze data generation finished. Total time: {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
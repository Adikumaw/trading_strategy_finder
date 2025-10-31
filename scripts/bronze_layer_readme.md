# Bronze Layer: The Possibility Engine (`bronze_data_generator.py`)

This script is the first and most foundational step in the Strategy Finder pipeline. Its sole purpose is to perform a high-speed, brute-force simulation on raw historical price data to discover every conceivable "winning" trade. It generates a massive, unbiased dataset that serves as the foundation for all subsequent analysis and enrichment.

## 🎯 Purpose in the Pipeline

The Bronze Layer acts as the **possibility engine**. Instead of testing predefined indicators, it works backward from the outcome. For every single candlestick in the input data, it asks: _"Given a grid of thousands of potential Stop Loss (SL) and Take Profit (TP) values, which combinations would have resulted in a winning trade?"_

This process creates a comprehensive "universe of possibilities" that will be filtered, enriched, and analyzed by the downstream layers.

---

## ✨ Key Features

- **High-Performance Simulation:** The core simulation loop is written in a restricted Python dialect and Just-In-Time (JIT) compiled to native machine code using **Numba**, achieving performance comparable to C or Fortran.
- **Massively Parallel:** It leverages the `multiprocessing` library to utilize all available CPU cores (configurable via `MAX_CPU_USAGE`), processing multiple chunks of the input data simultaneously.
- **Memory Safe:** By using a producer-consumer architecture with `pool.imap()`, the script can process gigabytes of historical data on a standard machine without running out of memory. Results are streamed to disk periodically.
- **Efficient Parquet Output:** Results are saved in the Apache Parquet format. This columnar, binary format is significantly smaller on disk and dramatically faster for the Silver Layer to read compared to traditional CSV files.
- **Chronologically Ordered Output:** Despite parallel processing, the output Parquet file is guaranteed to be sorted by entry time, which is essential for time-series integrity in later layers.
- **Realistic Spread Simulation:** The script accounts for the bid-ask spread by incorporating a configurable `SPREAD_PIPS` cost into the Take-Profit calculations, ensuring more realistic results.

---

## ⚙️ How It Works: The Logic

1.  **File Discovery:** The script can be run in two modes:
    - **Interactive Mode:** Scans the `/raw_data` directory and interactively prompts the user to select which new `.csv` files to process. It is smart enough to ignore files that already have a corresponding `.parquet` output in `/bronze_data`.
    - **Targeted Mode:** Processes a single file specified as a command-line argument.
2.  **Configuration Parsing:** It automatically detects the instrument (e.g., `EURUSD`) and timeframe (e.g., `15m`) from the filename and loads the appropriate simulation parameters (`SL_RATIOS`, `TP_RATIOS`, `MAX_LOOKFORWARD`) from the `TIMEFRAME_PRESETS`.
3.  **Data Loading & Validation:** The raw CSV data is loaded into memory once and validated to ensure it has the correct format and contains no invalid data that would crash the simulation.
4.  **Parallel Simulation (The "Producers"):**
    - A pool of worker processes is created. Each worker is initialized with a read-only copy of the full dataset, avoiding slow data transfer between processes (a critical optimization).
    - Each worker receives a chunk of data and executes the Numba-accelerated `find_winning_trades_numba` function, testing tens of thousands of SL/TP combinations for every candle.
5.  **Sequential Writing (The "Consumer"):**
    - The main process collects the results from the workers _in the correct order_.
    - Once a memory buffer (`OUTPUT_CHUNK_SIZE`) is filled, the results are converted to a DataFrame and appended as a new "row group" to the output Parquet file using a `ParquetWriter`. This process is highly efficient and memory-safe.

---

## 🛠️ Dependencies

This script requires the **pyarrow** library to handle the Parquet file format. Install it via pip:

```bash
pip install pyarrow
```

---

## 🔧 Configuration

All key parameters can be tuned directly in the global configuration section at the top of the script:

- `MAX_CPU_USAGE`: Set the number of CPU cores to use for simulation.
- `OUTPUT_CHUNK_SIZE`: Controls how many results are held in memory before writing to disk. Larger values can be faster but use more RAM.
- `INPUT_CHUNK_SIZE`: Defines how many candles are in each work package for the CPU cores.
- `SPREAD_PIPS`: A dictionary to define the estimated spread cost in pips for different instruments.
- `TIMEFRAME_PRESETS`: The core simulation grid. Here you can define the exact SL/TP ratios and the maximum trade holding period (`MAX_LOOKFORWARD`) for different chart timeframes.

---

## 🚀 Usage

Execute the script from the root directory of the project.

**1. Interactive Mode (Recommended):**

The script will scan for new files and present a menu.

````bash
python scripts/bronze_data_generator.py```

**2. Targeted Mode:**

To process a specific file from the `/raw_data` directory, pass its name as an argument.

```bash
python scripts/bronze_data_generator.py EURUSD15.csv
````

---

## 📄 Output

The script generates a `.parquet` file in the `/bronze_data/` directory with the same base name as the input file. This file contains every profitable trade found, with the following columns:

| Column        | Data Type        | Description                                             |
| :------------ | :--------------- | :------------------------------------------------------ |
| `entry_time`  | `datetime64[ns]` | The timestamp when the trade was initiated.             |
| `trade_type`  | `category`       | The direction of the trade (`buy` or `sell`).           |
| `entry_price` | `float32`        | The price at which the trade was entered (close price). |
| `sl_price`    | `float32`        | The calculated Stop Loss price level.                   |
| `tp_price`    | `float32`        | The calculated Take Profit price level.                 |
| `sl_ratio`    | `float32`        | The stop loss as a percentage of the entry price.       |
| `tp_ratio`    | `float32`        | The take profit as a percentage of the entry price.     |
| `exit_time`   | `datetime64[ns]` | The timestamp when the Take Profit level was hit.       |
| `outcome`     | `category`       | The result of the trade (always `win` in this layer).   |

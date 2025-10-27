# 🥉 Bronze Layer: The Possibility Engine

This script is the **foundational data generation layer** of the entire strategy discovery pipeline. Its purpose is to systematically scan historical price data and generate a vast, high-quality dataset of every conceivable winning trade based on a predefined set of rules.

This "universe of possibilities" forms the bedrock upon which all subsequent analysis is built.

## What It Is

The Bronze Layer is a high-performance, parallelized Python script that performs a "brute-force" simulation. For every single candle in a historical dataset, it simulates thousands of potential `buy` and `sell` trades, each with a different Stop-Loss (SL) and Take-Profit (TP) combination. It then looks forward in time to see which of these hypothetical trades would have resulted in a win.

The final output is a massive CSV file—the **Bronze Dataset**—containing a detailed log of every successful trade simulation.

## Why It Is

Quantitative trading research begins with a simple question: "What has worked in the past?" This script answers that question in the most comprehensive way possible.

Instead of starting with a biased idea (e.g., "let's test a moving average crossover"), we start with the outcome. By generating a complete dataset of _all_ profitable trades first, we create an unbiased foundation. The subsequent layers (Silver, Gold) can then mine this data to discover the market conditions and indicators that were predictive of these successful outcomes.

This "outcome-first" approach is designed to uncover hidden, non-obvious patterns that traditional strategy development might miss.

## How It Works: An Architectural Overview

The script is engineered for maximum speed, memory safety, and cross-platform stability to handle the billions of simulations it needs to perform. It operates on a **file-by-file basis**, parallelizing the workload _within_ each file.

1.  **File Discovery & Configuration:** The script scans the `raw_data/` directory. When a file is selected for processing (e.g., `XAUUSD15.csv`), it intelligently parses the filename to determine the instrument and timeframe.

2.  **Smart Configuration Loading:** Based on the detected metadata, it loads two key sets of parameters:

    - **SL/TP Presets:** Timeframe-specific `numpy` arrays of SL/TP ratios, accounting for different volatility levels.
    - **Spread Costs:** Instrument-specific spread values (e.g., 1.5 pips for EURUSD, 20 pips for XAUUSD) to ensure the simulations are realistic.

3.  **Pipelined Parallel Processing (Producer-Consumer Model):**

    - **Task Preparation:** The main historical data file is divided into thousands of small, overlapping **input chunks**.
    - **Worker Initialization:** A pool of worker processes is created, utilizing all but two CPU cores by default (`MAX_CPU_USAGE`). Crucially, the large historical dataset and configuration are loaded _once_ into each worker's memory using an `init_worker` function. This avoids costly data serialization on every task and ensures stability, especially on Windows.
    - **Parallel Production:** Each worker process ("producer") receives a chunk of data and hands it off to a core simulation function.

4.  **Numba-Accelerated Simulation:** The core simulation logic, which contains intensive nested loops, is Just-In-Time (JIT) compiled by `numba`. This transforms the Python code into highly optimized machine code, delivering a massive speedup (10-50x) compared to pure Python.

5.  **Ordered & Memory-Safe Output:**
    - The main process ("consumer") collects the results from the workers. By using `pool.imap()`, it guarantees that results are processed in their original chronological order.
    - The millions of discovered winning trades are collected in a temporary in-memory buffer.
    - Once this buffer reaches a set size (`OUTPUT_CHUNK_SIZE`), the data is formatted into a DataFrame and appended to the final output CSV on disk. The buffer is then cleared. This technique allows the script to generate enormous, multi-gigabyte files without ever crashing due to low memory.

## 📁 Folder Structure

```
project_root/
├── raw_data/                 # INPUT: Directory with raw OHLC CSV files
│   └── XAUUSD15.csv
│
├── bronze_data/              # OUTPUT: Auto-created directory for the generated data
│   └── XAUUSD15.csv
│
└── scripts/
    └── bronze_data_generator.py   # This script
```

## 📈 Input & Output

### Input File Format (`raw_data/`)

The script expects raw data files in a standard OHLC format. The filename must follow a specific pattern for auto-configuration: **`[INSTRUMENT][TIMEFRAME].csv`**.

- **Examples:** `EURUSD15.csv`, `XAUUSD60.csv`, `GBPUSD1.csv`
- **Columns (headerless):** `time, open, high, low, close, [volume]`
  - The script is robust to files with or without a volume column.

### Output File Format (`bronze_data/`)

The output is a single, large CSV file for each input file, containing a log of all discovered winning trades.

| Column        | Description                                            | Example               |
| :------------ | :----------------------------------------------------- | :-------------------- |
| `entry_time`  | Timestamp of the candle where the trade was initiated. | `2023-10-18 14:30:00` |
| `trade_type`  | The direction of the trade.                            | `buy` or `sell`       |
| `entry_price` | The closing price of the entry candle.                 | `1925.45`             |
| `sl_price`    | The calculated Stop-Loss price for this trade.         | `1921.20`             |
| `tp_price`    | The calculated Take-Profit price for this trade.       | `1935.90`             |
| `sl_ratio`    | The stop-loss percentage used (e.g., 0.5%).            | `0.005`               |
| `tp_ratio`    | The take-profit percentage used (e.g., 1.0%).          | `0.010`               |
| `exit_time`   | Timestamp of the candle where the Take-Profit was hit. | `2023-10-18 17:15:00` |
| `outcome`     | The result of the trade (always "win" in this layer).  | `win`                 |

## 🚀 Possible Enhancements & Future Improvements

1.  **Dynamic Lookahead:** The `MAX_LOOKFORWARD` parameter is currently fixed per timeframe. A more advanced version could calculate the average volatility (e.g., using ATR) and dynamically adjust the lookahead period, potentially speeding up simulations in low-volatility environments.
2.  **Generate Losing Trades:** For some advanced modeling techniques (like building a binary classifier), it would be beneficial to also have a dataset of losing trades. An option could be added to the script to generate and save a `bronze_data/losses/{instrument}.csv` file. This would dramatically increase the data size but could enable different types of analysis in later layers.
3.  **Alternative Exit Logic:** The current model only considers SL/TP exits. The engine could be expanded to include other exit conditions, such as a time-based stop (e.g., "exit trade if not profitable after N candles").
4.  **Configuration File:** The `TIMEFRAME_PRESETS` and `SPREAD_PIPS` dictionaries are currently hardcoded. For easier tuning and experimentation, these could be moved to an external `config.yaml` or `config.json` file.

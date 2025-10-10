# 📊 Bronze Layer: Trade Combination Generator

This Python script automatically generates an exhaustive dataset of all possible winning trade combinations based on **Stop-Loss (SL)** and **Take-Profit (TP)** ratios. It processes raw historical price data and serves as the foundational data generation step for the entire strategy discovery pipeline.

It is designed to build a **Bronze Dataset**—a comprehensive catalog of successful trade outcomes that will be enriched and analyzed by subsequent layers.

---

## ⚙️ How It Works

1.  **Scan & Detect:** The script scans the `raw_data/` directory for `.csv` files and automatically detects the timeframe from each filename (e.g., `EURUSD15.csv` → 15m).
2.  **Apply Presets:** It applies timeframe-specific SL/TP ratio ranges from a built-in configuration.
3.  **Simulate Trades:** For every single candle in the input file, it simulates all possible "Buy" and "Sell" trades based on the preset ratios.
4.  **Forward Test:** It iterates forward in time (up to a `MAX_LOOKFORWARD` limit) to see which SL/TP combinations would have resulted in a winning trade (i.e., TP is hit before SL).
5.  **Efficiently Save Results:** It appends the vast number of generated winning trades to an output file in memory-efficient chunks, preventing RAM overloads.

---

## 📁 Folder Structure

```
project_root/
├── raw_data/         # INPUT: Directory with timeframe-based CSVs
│   └── EURUSD1.csv
│
├── bronze_data/      # OUTPUT: Auto-created directory for the generated data
│   └── EURUSD1.csv
│
└── scripts/
    └── bronze_data_generator.py   # This script
```

---

## 🧩 CSV Input Format

Your raw data files must be in an OHLC format. The script is robust and auto-detects column names based on their order.

**Expected Columns:**
`time, open, high, low, close`
_(Any additional columns, like volume, are ignored)._

---

## ⚖️ SL/TP Preset Configuration

Each timeframe uses different percentage-based ranges for generating SL/TP combinations to account for varying volatility.

| Timeframe | SL Range   | TP Range   | Lookahead   |
| :-------- | :--------- | :--------- | :---------- |
| 1m        | 0.05%–1.0% | 0.05%–2.0% | 200 candles |
| 5m        | 0.1%–1.5%  | 0.1%–3.0%  | 300 candles |
| 15m       | 0.2%–2.5%  | 0.2%–5.0%  | 400 candles |
| 30m       | 0.3%–3.5%  | 0.3%–7.0%  | 500 candles |
| 60m       | 0.5%–5.0%  | 0.5%–10.0% | 600 candles |
| 240m      | 1.0%–10.0% | 1.0%–20.0% | 800 candles |

---

## ⚡ Performance & Multiprocessing

This script is optimized for both speed and memory safety.

-   **Memory Efficiency:** Results are saved to disk in **chunks of 1 million trades**. This allows the script to process huge input files and generate billions of trade simulations without exceeding system RAM.
-   **Multiprocessing:** You can choose to process multiple input files (e.g., `AUDUSD1.csv` and `EURUSD15.csv`) in parallel to utilize all available CPU cores. Each file is handled by a separate process.

**Example:**
When you run the script, you will be prompted to choose a processing mode:

```
Use multiprocessing? (y/n): y
```

This allows you to tailor the script's resource usage to your specific machine.

---

## 🧱 Output File Description

Each output CSV file in `bronze_data/` is a large table containing all discovered winning trades.

| Column        | Description                                       |
| :------------ | :------------------------------------------------ |
| `entry_time`  | Timestamp of the trade entry candle.              |
| `trade_type`  | "buy" or "sell".                                  |
| `entry_price` | The closing price of the entry candle.            |
| `sl_price`    | The calculated Stop-Loss price.                   |
| `tp_price`    | The calculated Take-Profit price.                 |
| `sl_ratio`    | The Stop-Loss percentage used for this trade.     |
| `tp_ratio`    | The Take-Profit percentage used for this trade.   |
| `exit_time`   | The timestamp of the candle where the TP was hit. |
| `outcome`     | "win".                                            |

**Note:** This dataset is the direct input for the Silver Layer.

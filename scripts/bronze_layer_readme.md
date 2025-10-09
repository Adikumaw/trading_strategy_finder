# 📊 Trade Combination Generator (SL/TP Strategy Dataset Builder)

This Python script automatically generates **all possible combinations of Stop-Loss (SL)** and **Take-Profit (TP)** targets for each candle in your trading data, based on the timeframe detected in your CSV filenames.

It’s designed to help you build a **Bronze Dataset** — a base-level, exhaustive trade outcome dataset used for later strategy backtesting, optimization, and AI analysis.

---

## ⚙️ How It Works

-   Scans the `raw_data/` directory for `.csv` files.
-   Determines the timeframe from the filename (e.g. `EURUSD15.csv` → 15m).
-   Applies timeframe-specific SL/TP ratio ranges from a preset configuration.
-   For each candle:
    -   Simulates Buy and Sell trades.
    -   Iterates forward up to a configurable lookahead window.
    -   Marks trades as winning if TP is hit before SL.
-   Stores all generated trades in `bronze_data/` as CSV files.

---

## 📁 Folder Structure

```
project_root/
├── raw_data/         # Input directory with timeframe-based CSVs
│   ├── EURUSD1.csv
│   ├── EURUSD15.csv
│   └── ...
├── bronze_data/      # Output directory (auto-created)
│   ├── EURUSD1.csv
│   ├── EURUSD15.csv
│   └── ...
└── scripts/
    └── bronze_data_generator.py   # This script
```

---

## 🧩 CSV Input Format

Your raw data files must be **OHLCV-style** but can contain more columns — only the first five are used:

```
time, open, high, low, close
```

-   `time`: Timestamp of each candle
-   `open, high, low, close`: Numeric price values

Extra columns are ignored.

---

## 🕒 Timeframe Detection

The script automatically detects the timeframe from the filename:

-   `EURUSD1.csv` → 1m
-   `EURUSD15.csv` → 15m
-   `EURUSD60.csv` → 60m

If no valid timeframe number is found, the file is skipped.

---

## ⚖️ SL/TP Preset Configuration

Each timeframe uses different **percentage-based ranges** for generating SL/TP combinations and a maximum lookahead window:

| Timeframe | SL Range   | TP Range   | Lookahead   |
| --------- | ---------- | ---------- | ----------- |
| 1m        | 0.05%–1.0% | 0.05%–2.0% | 200 candles |
| 5m        | 0.1%–1.5%  | 0.1%–3.0%  | 300 candles |
| 15m       | 0.2%–2.5%  | 0.2%–5.0%  | 400 candles |
| 30m       | 0.3%–3.5%  | 0.3%–7.0%  | 500 candles |
| 60m       | 0.5%–5.0%  | 0.5%–10.0% | 600 candles |
| 240m      | 1.0%–10.0% | 1.0%–20.0% | 800 candles |

Ratios are applied **relative to each candle’s closing price**.

---

## ⚡ Processing Flow

For each data file:

1. Load and clean OHLC data.
2. For each candle:
    - Record entry time and price.
    - Generate all SL/TP price levels for both buy and sell sides.
    - Iterate forward up to `MAX_LOOKFORWARD` candles.
    - Check whether TP or SL is hit first.
3. Append profitable (TP-hit) trades to a temporary list.
4. Save results to `bronze_data/` in memory-efficient chunks.

---

## 🔄 Multiprocessing Support

-   Each file is processed independently.
-   Progress bars (`tqdm`) show live updates per file.
-   Multi-core mode uses all CPUs except two (for system stability).

**Example:**

When you run the script:

```
Use multiprocessing? (y/n): y
```

If you select `n`, you’ll be prompted to enter how many processes to use manually.

---

## 🧮 Example Workflow

1. **Place raw files:**
    ```
    raw_data/
    ├── EURUSD1.csv
    ├── GBPUSD15.csv
    ├── BTCUSD60.csv
    ```
2. **Run the script:**
    ```bash
    python scripts/bronze_data_generator.py
    ```
3. **Choose mode:**
    ```
    Use multiprocessing? (y/n): y
    ```
4. **Output:**
    ```
    bronze_data/
    ├── EURUSD1.csv
    ├── GBPUSD15.csv
    ├── BTCUSD60.csv
    ```

Each file contains detailed simulated trades:

```csv
entry_time,trade_type,entry_price,sl_price,tp_price,sl_ratio,tp_ratio,exit_time,outcome
2024-06-15 09:45:00,buy,1.12345,1.12289,1.12467,0.0005,0.0011,2024-06-15 09:47:00,win
...
```

---

## 🧱 Output File Description

Each output CSV (`bronze_data/*.csv`) contains:

| Column      | Description                                           |
| ----------- | ----------------------------------------------------- |
| entry_time  | Time of trade entry                                   |
| trade_type  | "buy" or "sell"                                       |
| entry_price | Entry price at candle close                           |
| sl_price    | Stop-loss price                                       |
| tp_price    | Take-profit price                                     |
| sl_ratio    | Relative SL percentage                                |
| tp_ratio    | Relative TP percentage                                |
| exit_time   | Time when TP or SL was hit                            |
| outcome     | "win" if TP hit before SL, "loss" if SL hit before TP |

**Note:** This dataset is the direct input for the Silver Layer, which performs advanced feature engineering and outcome enrichment.

---

## ⚠️ Notes & Recommendations

-   The script is memory-efficient and saves data in chunks (1 million rows at a time).
-   Avoid putting too many large files in `raw_data` at once — processing time can be long.
-   Ensure filenames include timeframe numbers (e.g. `EURUSD60.csv`).
-   Already processed files are automatically skipped.
-   Use multiprocessing only if your system has sufficient memory.
-   If you plan to use ATR-based SL/TP later, you can integrate ATR calculation and store it as metadata alongside each trade.

---

## 📈 Example Use Case

This dataset is perfect for:

-   Finding statistically profitable SL/TP combinations
-   Machine learning model training
-   Historical performance analysis
-   Volatility and timeframe sensitivity testing

---

## 🧠 Future Enhancements

-   ✅ Include SL-hit trades (for loss analysis)

---

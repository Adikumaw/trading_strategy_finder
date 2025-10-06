# 📊 Trade Combination Generator (SL/TP Strategy Dataset Builder)

This Python script automatically generates **all possible combinations of Stop-Loss (SL)** and **Take-Profit (TP)** targets for each candle in your trading data, based on the timeframe detected in your CSV filenames.

It’s designed to help you build a _"Bronze Dataset"_ — a base-level, exhaustive trade outcome dataset used for later **strategy backtesting, optimization, and AI analysis.**

---

## ⚙️ How It Works

1. The script scans the `raw_data/` directory for `.csv` files.
2. For each file, it determines the **timeframe** (e.g. `EURUSD15.csv → 15m`).
3. It applies timeframe-specific SL/TP ratio ranges from a preset configuration.
4. For each candle:

    - It simulates **Buy** and **Sell** trades.
    - Iterates forward up to a configurable lookahead window.
    - Marks trades as _winning_ if TP is hit before SL.

5. All generated trades are stored in `bronze_data/` as CSV files.

---

## 📁 Folder Structure

```
project_root/
│
├── raw_data/             # Input directory with timeframe-based CSVs
│   ├── EURUSD1.csv
│   ├── EURUSD15.csv
│   └── ...
│
├── bronze_data/          # Output directory (auto-created)
│   ├── EURUSD1.csv
│   ├── EURUSD15.csv
│   └── ...
│
└── scripts/
    └── bronze_data_generator.py   # This script
```

---

## 🧩 CSV Input Format

Your raw data files must be **OHLCV-style** but can contain more columns — only the first five are used.

**Expected Columns (auto-detected):**

```
time, open, high, low, close
```

-   `time`: Timestamp of each candle
-   `open, high, low, close`: Numeric price values

If more than 5 columns exist, extras are ignored.

---

## 🕒 Timeframe Detection

The script automatically detects timeframe based on the number in your filename.

Example:

```
EURUSD1.csv   → timeframe: 1m
EURUSD15.csv  → timeframe: 15m
EURUSD60.csv  → timeframe: 60m
```

If no valid timeframe number is found, the file is skipped.

---

## ⚖️ SL/TP Preset Configuration

Each timeframe uses different **percentage-based ranges** for generating Stop-Loss (SL) and Take-Profit (TP) combinations, as well as a maximum lookahead window (in candles).

| Timeframe | SL Range     | TP Range     | Lookahead   |
| --------- | ------------ | ------------ | ----------- |
| 1m        | 0.05% → 1.0% | 0.05% → 2.0% | 200 candles |
| 5m        | 0.1% → 1.5%  | 0.1% → 3.0%  | 300 candles |
| 15m       | 0.2% → 2.5%  | 0.2% → 5.0%  | 400 candles |
| 30m       | 0.3% → 3.5%  | 0.3% → 7.0%  | 500 candles |
| 60m       | 0.5% → 5.0%  | 0.5% → 10.0% | 600 candles |
| 240m      | 1.0% → 10.0% | 1.0% → 20.0% | 800 candles |

These ratios are applied **relative to each candle’s closing price**.

---

## ⚡ Processing Flow

For each data file:

1. Load and clean OHLC data.
2. For each candle:

    - Record entry time and price.
    - Generate all SL/TP price levels for both buy and sell sides.
    - Iterate forward up to `MAX_LOOKFORWARD` candles.
    - Check whether TP or SL is hit first.

3. Append profitable (TP-hit) trades into a dataset.
4. Save results to `bronze_data/` with the same filename.

---

## 🔄 Multiprocessing Support

You can choose between **single-process** or **multi-core** mode.

### 🧠 Logic:

-   Each file is processed independently.
-   Progress bars (`tqdm`) show live updates per file.
-   Multi-core mode uses all CPUs except two (for system stability).

### ⚙️ Example:

When you run the script:

```
Use multiprocessing? (y/n): y
```

If you select `n`, you’ll be prompted to enter how many processes to use manually.

---

## 🧮 Example Workflow

1. **Place raw files**

    ```
    raw_data/
    ├── EURUSD1.csv
    ├── GBPUSD15.csv
    ├── BTCUSD60.csv
    ```

2. **Run the script**

    ```bash
    python scripts/bronze_data_generator.py
    ```

3. **Choose mode**

    ```
    Use multiprocessing? (y/n): y
    ```

4. **Output**

    ```
    bronze_data/
    ├── EURUSD1.csv
    ├── GBPUSD15.csv
    ├── BTCUSD60.csv
    ```

Each file now contains detailed simulated trades:

```csv
entry_time,trade_type,entry_price,sl_price,tp_price,sl_ratio,tp_ratio,exit_time,outcome
2024-06-15 09:45:00,buy,1.12345,1.12289,1.12467,0.0005,0.0011,2024-06-15 09:47:00,win
...
```

---

## 🧱 Output File Description

Each output CSV (`bronze_data/*.csv`) contains:

| Column      | Description                 |
| ----------- | --------------------------- |
| entry_time  | Time of trade entry         |
| trade_type  | “buy” or “sell”             |
| entry_price | Entry price at candle close |
| sl_price    | Stop-loss price             |
| tp_price    | Take-profit price           |
| sl_ratio    | Relative SL percentage      |
| tp_ratio    | Relative TP percentage      |
| exit_time   | Time when TP or SL was hit  |
| outcome     | “win” if TP hit before SL   |

---

## ⚠️ Notes & Recommendations

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

-   ✅ Add ATR-based dynamic SL/TP generation
-   ✅ Include SL-hit trades (for loss analysis)
-   🔜 Support multi-symbol parallel backtesting
-   🔜 Store volatility and candle pattern metadata

---

# 🪙 Silver Data Generator (Hybrid Optimized Version)

This project automates the creation of **Silver Datasets** for algorithmic trading and quantitative analysis — combining raw OHLCV market data with technical indicators, support/resistance levels, and other engineered features to create a clean, model-ready dataset.

Built for **large-scale datasets (7–10 GB)** and **optimized for speed and memory efficiency** using:

-   🧮 **Numba** for hybrid fast computation
-   📈 **TA-Lib** + **TA** for indicators
-   💾 **Chunked CSV Processing**
-   ⚙️ **Smart Downcasting** for low memory use

---

## 🚀 Overview

### 📘 What it does

This pipeline transforms:

```

RAW (OHLCV data) + BRONZE (Trade entries)
↓
SILVER (Feature-rich dataset)

```

Each _Silver_ dataset includes:

-   40+ Technical Indicators
-   Support & Resistance Zones
-   Candle & Price Action Stats
-   Market Session Features
-   Trend & Volatility Regimes
-   Cleaned, Merged Trade Records

---

## 🧩 Pipeline Summary

| Step                       | Description                                                                                                      |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **1️⃣ Load Raw Data**       | Loads OHLCV candles, cleans, and formats columns.                                                                |
| **2️⃣ Add Features**        | Calculates all technical indicators and patterns using TA-Lib, TA, and Numba-based support/resistance detection. |
| **3️⃣ Warmup Cutoff**       | Removes the initial unstable rows (e.g., first 200 candles).                                                     |
| **4️⃣ Load Bronze Data**    | Loads trades in chunks (1M rows at a time).                                                                      |
| **5️⃣ Merge Efficiently**   | Uses `pd.merge_asof` to attach the latest available candle features to each trade.                               |
| **6️⃣ Write Incrementally** | Saves the merged dataset to disk without loading everything in memory.                                           |

---

## ⚙️ Configuration

You can customize parameters at the top of the script:

```python
SMA_PERIODS = [20, 50, 100, 200]
EMA_PERIODS = [8, 13, 21, 50]
BBANDS_PERIOD, BBANDS_STD_DEV = 20, 2.0
RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
ATR_PERIOD, ADX_PERIOD = 14, 14
SR_LOOKBACK = 200
PAST_LOOKBACKS = [3, 5, 10, 20, 50]

INDICATOR_WARMUP_PERIOD = 200
BRONZE_CHUNK_SIZE = 1_000_000
```

| Variable                  | Description                                            |
| ------------------------- | ------------------------------------------------------ |
| `SMA_PERIODS`             | Periods for simple moving averages                     |
| `EMA_PERIODS`             | Periods for exponential moving averages                |
| `SR_LOOKBACK`             | Candles to look back when detecting Support/Resistance |
| `BRONZE_CHUNK_SIZE`       | Number of rows processed per chunk from bronze data    |
| `INDICATOR_WARMUP_PERIOD` | Rows to skip until indicators stabilize                |

---

## 🧠 Feature Engineering Breakdown

### 📊 Trend Indicators

-   `SMA_xx`, `EMA_xx`
-   `MACD`, `MACD_signal`, `MACD_hist`

### ⚡ Momentum & Oscillators

-   `RSI_14`, `CCI_20`, `MOM_10`

### 📉 Volatility Indicators

-   `ATR_14`
-   `BB_upper`, `BB_lower`, `BB_width`

### 🕯️ Candlestick Patterns

-   Adds all TA-Lib patterns (`CDLENGULFING`, `CDLDOJI`, etc.)

### 💎 Support & Resistance (Hybrid)

-   Uses Numba-accelerated hybrid S/R detection algorithm.
-   Returns pivot-based support and resistance levels validated across lookback.

### 🕰️ Market Session Features

Automatically assigns:

-   `London`
-   `London_NY_Overlap`
-   `New_York`
-   `Asian`

Also includes:

-   `hour`, `weekday`

### 📈 Price Action Stats

For each lookback window (`3, 5, 10, 20, 50`):

-   `bullish_ratio_last_n`
-   `avg_body_last_n`
-   `avg_range_last_n`
-   `close_SMA20_ratio_n`
-   `EMA8_EMA21_ratio_n`

### 📉 Regime Detection

Defines:

-   `trend_regime` → `"trend"` or `"range"`
-   `vol_regime` → `"high_vol"` or `"low_vol"`

---

## 🧮 Support/Resistance Hybrid Logic

The hybrid version combines **pivot detection** and **rolling validation**:

-   Finds local highs/lows (pivot points).
-   Confirms whether they remain unbroken for `SR_LOOKBACK` candles.
-   Produces stable and realistic support/resistance zones.

It’s powered by **Numba JIT** for near-C-speed execution.

---

## 💾 Handling Huge Data (7–10 GB CSVs)

The pipeline is designed for scalability:

-   Loads trades in **chunks** (`BRONZE_CHUNK_SIZE`)
-   Merges efficiently using **as-of merge**
-   **Downcasts** datatypes (`float64 → float32`)
-   Writes incrementally to avoid memory overflows

---

## 🧰 Directory Structure

```
project/
│
├── raw_data/           # Raw OHLCV data files
│   ├── AUDUSD1.csv
│   ├── EURUSD1.csv
│   └── ...
│
├── bronze_data/        # Trade data files (entries, exits)
│   ├── AUDUSD1.csv
│   └── ...
│
├── silver_data/        # Output feature-rich datasets
│   ├── AUDUSD1_silver.csv
│   └── ...
│
└── generate_silver_data.py  # This script
```

---

## ▶️ How to Run

1. **Place your raw and bronze data** in respective folders:

    ```
    ./raw_data/
    ./bronze_data/
    ```

2. **Install dependencies**:

    ```bash
    pip install pandas numpy ta talib tqdm numba
    ```

3. **Run the script**:

    ```bash
    python generate_silver_data.py
    ```

4. **Output** will appear in:

    ```
    ./silver_data/
    ```

---

## 🧩 Integration with Higher Timeframes (Optional)

The script supports merging with **higher timeframe features** (e.g., 60m or 240m data) for richer context.

Example:

```
AUDUSD1.csv  +  AUDUSD60.csv  →  AUDUSD1_silver.csv
```

This multi-timeframe design improves ML model accuracy and trade signal reliability.

---

## 🧱 Tech Stack

| Library         | Purpose                                    |
| --------------- | ------------------------------------------ |
| **Pandas**      | Data processing & chunk management         |
| **NumPy**       | Vectorized calculations                    |
| **Numba**       | Accelerated Support/Resistance computation |
| **TA-Lib / ta** | Technical indicators                       |
| **TQDM**        | Progress visualization                     |

---

## 🧠 Key Advantages

✅ Handles datasets up to **10GB** easily
✅ Near **C-speed S/R detection** with Numba
✅ **Modular and extensible** design
✅ Generates **clean, ML-ready data**
✅ Automatic **session classification**
✅ Efficient **RAM usage** (downcasting + chunking)

---

## 📎 Example Output Columns

| Column                              | Description                          |
| ----------------------------------- | ------------------------------------ |
| `time`                              | Candle timestamp                     |
| `open, high, low, close, volume`    | OHLCV data                           |
| `EMA_21, SMA_200, RSI_14`           | Technical indicators                 |
| `support_points, resistance_points` | Hybrid support/resistance            |
| `session`                           | Trading session                      |
| `trend_regime, vol_regime`          | Market regimes                       |
| `bullish_ratio_last_10`             | % of bullish candles in last 10 bars |
| `entry_time, exit_time, pnl`        | Merged from bronze trade data        |

---

## 📚 Notes

-   Works with both **5-column** and **6-column** raw data formats.
-   The script auto-corrects separators and formats.
-   Minimum 200 candles required before indicator stability.

---

## 🧠 Future Enhancements

-   📦 Multi-Timeframe Feature Stacking (HTF + LTF)
-   🔥 Parallel chunk processing with `multiprocessing`
-   🧰 Optional SQLite/Parquet backend
-   🧠 Built-in model training hooks

---

## 🤝 Author

**Gaurav Kumawat**
Front-End & Quant Developer
💬 Built with ❤️ and caffeine for large-scale trading analytics.

---

## 📜 License

This project is distributed under the **MIT License**.
You’re free to use, modify, and integrate it into your projects with attribution.

---

### ✨ “From raw noise to structured signal — the Silver Data Generator bridges chaos and clarity.” ⚡

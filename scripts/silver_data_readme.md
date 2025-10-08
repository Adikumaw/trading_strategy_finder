# 🧠 Silver Data Generator (Decoupled Feature & Outcome Engine)

This Python script is a critical architectural component that transforms raw market data and trade simulations into a clean, efficient, and logically sound **"Silver Dataset"**. It is designed to prevent data leakage and prepare the ground for robust machine learning analysis.

The script's core purpose is to **decouple the market context (features) from the trade results (outcomes)**. This fixes a major logical flaw where a single market event could be over-represented, leading to biased and unreliable models.

---

## ⚙️ How It Works

This script follows a two-step process to create two distinct, synchronized datasets:

1.  **Step 1: Generate a `features` Dataset:**

    -   It scans for a `raw_data` file (e.g., `AUDUSD1.csv`).
    -   It loads the entire historical OHLCV data.
    -   It calculates a comprehensive set of over 200 market features (indicators, patterns, etc.) for **every single candle**.
    -   After removing the initial indicator warmup period, it saves this clean, non-duplicated data to `silver_data/features/`. Each row in this file represents the unique state of the market at one point in time.

2.  **Step 2: Generate an `outcomes` Dataset:**
    -   It finds the corresponding `bronze_data` file.
    -   It reads the massive trade simulation file in memory-efficient chunks.
    -   It **filters** these trades, keeping only those whose `entry_time` occurs on or after the first available timestamp in the `features` dataset. This synchronizes the two files.
    -   It saves the filtered trade results to `silver_data/outcomes/`.

This creates a powerful **one-to-many relationship**, where the `time` column in the `features` file acts as a unique key to link one set of market conditions to the many possible trade outcomes in the `outcomes` file.

---

## 📁 New Folder Structure

The script now produces a structured output within the `silver_data` directory, separating market context from trade results.

```
project_root/
│
├── raw_data/             # INPUT: Raw OHLCV data (no header)
│   └── AUDUSD1.csv
│
├── bronze_data/          # INPUT: Trade simulations from the bronze script
│   └── AUDUSD1.csv
│
├── silver_data/          # OUTPUT: Parent directory for decoupled data
│   │
│   ├── features/         # OUTPUT 1: Unique market features, one row per candle
│   │   └── AUDUSD1.csv
│   │
│   └── outcomes/         # OUTPUT 2: All trade outcomes, many rows per candle
│       └── AUDUSD1.csv
│
└── scripts/
    └── silver_data_generator.py   # This script
```

---

## 🚀 Feature Engineering Engine

This script generates a rich set of market features for **every candle**, ensuring you have a complete basis for discovering any possible strategy. The features are calculated in efficient batches to optimize performance.

| Feature Category           | Examples                                                                      | Purpose                                                                 |
| :------------------------- | :---------------------------------------------------------------------------- | :---------------------------------------------------------------------- |
| **Technical Indicators**   | `SMA`, `EMA`, `RSI`, `MACD`, `Bollinger Bands`, `ATR`, `ADX`, `CCI`, `OBV`    | Quantify market momentum, trend, volatility, and volume dynamics.       |
| **Candlestick Patterns**   | `CDLDOJI`, `CDLENGULFING`, `CDLHAMMER`, etc. (All 61 from `talib`)            | Capture classic price action signals and potential reversals.           |
| **Support & Resistance**   | `support`, `resistance` (calculated with a fast, Numba-accelerated algorithm) | Identify key historical price levels that might influence future price. |
| **Price Action & Regimes** | `bullish_ratio_last_N`, `avg_body_last_N`, `trend_regime`, `vol_regime`       | Characterize recent price behavior and the overall market state.        |
| **Time-Based Features**    | `session` (Asian, London, NY), `hour`, `weekday`                              | Allow the model to find patterns related to time of day or week.        |

---

## ⚡ Performance & Efficiency

-   **Efficient Feature Calculation:** Features are computed in optimized batches and concatenated in a single operation to avoid memory fragmentation and improve speed.
-   **Chunking for Outcomes:** The script processes the massive bronze data file in chunks to generate the outcomes file, ensuring it can run on systems with limited RAM.
-   **Indicator Warmup:** Skips the first `200` candles of historical data to ensure all indicators are based on stable, meaningful values.
-   **Numba JIT Compilation:** The Support and Resistance algorithm is accelerated with Numba for high-speed calculation.
-   **Downcasting:** Automatically reduces the memory footprint of the final `features` dataset by downcasting data types.

---

## 🧮 Example Workflow

1.  **Prerequisite:** Ensure you have already run `bronze_data_generator.py`.

2.  **Run the script:**

    ```bash
    python scripts/silver_data_generator.py
    ```

3.  **Monitor the output:**

    ```
    =========================
    Processing: AUDUSD1.csv
    =========================
    STEP 1: Creating Silver Features dataset (unique per candle)...
    ...
    ✅ Silver Features saved to: silver_data\features\AUDUSD1.csv (99800 rows)

    STEP 2: Creating Silver Outcomes dataset (from bronze data)...
    ...
    ✅ Silver Outcomes saved to: silver_data\outcomes\AUDUSD1.csv
    ```

---

## 🧱 Output File Description

The script generates two distinct and crucial files for each instrument.

### 1. `silver_data/features/{instrument}.csv`

This file contains the complete market context. It has **one row for every candle** and a very large number of columns.

| Column Category                          | Description                                                               |
| :--------------------------------------- | :------------------------------------------------------------------------ |
| `time`                                   | **Unique Key.** The timestamp of the candle.                              |
| `open`, `high`, `low`, `close`, `volume` | The core OHLCV data for the candle.                                       |
| **Indicator Columns**                    | `SMA_20`, `EMA_8`, `RSI_14`, `BB_width`, `ADX`, `MACD_hist`, etc.         |
| **Candlestick Columns**                  | `CDLDOJI`, `CDLHAMMER`, etc. A column for each of the 61 TA-Lib patterns. |
| **S/R Columns**                          | `support`, `resistance`. The last known support/resistance levels.        |
| **Time Columns**                         | `session`, `hour`, `weekday`.                                             |
| **Price Action Columns**                 | `bullish_ratio_last_3`, `avg_body_last_10`, `close_SMA20_ratio_50`, etc.  |
| **Regime Columns**                       | `trend_regime`, `vol_regime`.                                             |

### 2. `silver_data/outcomes/{instrument}.csv`

This file contains all the simulated trade results from the Bronze layer, synchronized with the features file. It can have **many rows for every candle**.

| Column                 | Description                                                                              |
| :--------------------- | :--------------------------------------------------------------------------------------- |
| `entry_time`           | **Foreign Key.** The timestamp of the trade entry. Links to `time` in the features file. |
| `trade_type`           | “buy” or “sell”.                                                                         |
| `entry_price`          | Entry price at candle close.                                                             |
| `sl_price`, `tp_price` | Stop-loss and Take-profit price levels.                                                  |
| `sl_ratio`, `tp_ratio` | The relative SL/TP percentages used for this trade.                                      |
| `exit_time`            | The timestamp when the trade concluded.                                                  |
| `outcome`              | "win" (or "loss" if the bronze script generates them).                                   |

---

## 📈 Example Use Case

These two decoupled files are the **correct** inputs for the next stage of the pipeline (`gold_data_generator.py`). The next script will:

1.  Load the `outcomes.csv` to calculate a **win rate for each `entry_time`**.
2.  Join this win rate onto the `features.csv` using the `time` and `entry_time` columns as the key.
3.  This creates a clean dataset for training a model to predict which market conditions lead to a high probability of success, fixing the "one candle, many votes" logical flaw.

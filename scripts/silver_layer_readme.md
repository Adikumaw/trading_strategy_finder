# 🧠 Silver Data Generator (Advanced Feature & Outcome Engine)

This Python script is a critical architectural component that transforms raw market data and trade simulations into a clean, efficient, and logically sound **"Silver Dataset"**. It is designed to prevent data leakage and prepare the ground for robust machine learning analysis.

The script's core purpose is to **decouple the market context (features) from the trade results (outcomes)** and enrich those outcomes with advanced positioning metrics.

---

## ⚙️ How It Works

This script follows a two-step process to create two distinct, synchronized datasets:

1.  **Step 1: Generate a `features` Dataset:**

    -   It scans for a `raw_data` file (e.g., `AUDUSD1.csv`).
    -   It calculates a comprehensive set of over 200 market features (indicators, patterns, etc.) for **every single candle**.
    -   It saves this clean, non-duplicated data to `silver_data/features/`. Each row represents the unique state of the market at one point in time.

2.  **Step 2: Generate an Enriched `outcomes` Dataset:**
    -   It finds the corresponding `bronze_data` file.
    -   It reads the massive trade simulation file in memory-efficient chunks.
    -   For each trade, it **enriches** it by calculating **two types of advanced positioning features**:
        1.  **Distance in Basis Points (`_bps`):** How far the SL/TP is from an indicator.
        2.  **Percentage Placement (`_placement_pct_to`):** Where the SL/TP is placed on a scale from the entry to an indicator.
    -   It saves these highly detailed trade results to `silver_data/outcomes/`.

This creates a powerful **one-to-many relationship**, where the `time` column in the `features` file acts as a unique key to link one set of market conditions to the many possible trade outcomes.

---

## 📁 New Folder Structure

```
project_root/
│
├── raw_data/             # INPUT: Raw OHLCV data
├── bronze_data/          # INPUT: Trade simulations
│
├── silver_data/          # OUTPUT: Parent directory
│   │
│   ├── features/         # OUTPUT 1: Unique market features
│   │   └── AUDUSD1.csv
│   │
│   └── outcomes/         # OUTPUT 2: Enriched trade outcomes
│       └── AUDUSD1.csv
│
└── scripts/
    └── silver_data_generator.py
```

---

## 🚀 Feature Engineering Engine

This script now generates two distinct categories of positioning features, enabling a far more sophisticated analysis in the next layer.

| Feature Category            | Examples                                                     | Purpose                                                                                                                                                                                    |
| :-------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Technical Indicators**    | `SMA`, `EMA`, `RSI`, `MACD`, `Bollinger Bands`, `ATR`, `ADX` | Quantify market momentum, trend, volatility, and volume dynamics.                                                                                                                          |
| **Candlestick Patterns**    | `CDLDOJI`, `CDLENGULFING`, `CDLHAMMER`                       | Capture classic price action signals.                                                                                                                                                      |
| **Support & Resistance**    | `support`, `resistance`                                      | Identify key historical price levels.                                                                                                                                                      |
| **Price Action & Regimes**  | `bullish_ratio_last_N`, `trend_regime`, `vol_regime`         | Characterize recent price behavior and the market state.                                                                                                                                   |
| **Time-Based Features**     | `session` (Asian, London, NY), `hour`, `weekday`             | Allow models to find time-based patterns.                                                                                                                                                  |
| **Positioning (Distance)**  | `sl_dist_to_SMA_50_bps`                                      | Quantifies the _raw distance_ from a trade target to an indicator in basis points.                                                                                                         |
| **Positioning (Placement)** | `tp_placement_pct_to_resistance`                             | Quantifies _where_ a target is placed on a scale from the entry price (0%) to an indicator (100%). **Handles directionality** (e.g., a negative value means it's placed behind the entry). |

---

## ⚡ Performance & Efficiency

-   **Efficient Feature Calculation:** Features are computed in optimized batches and concatenated in a single operation to avoid memory fragmentation and improve speed.
-   **Chunking for Outcomes:** The script processes the massive bronze data file in chunks to generate the outcomes file, ensuring it can run on systems with limited RAM.
-   **Indicator Warmup:** Skips the first `200` candles of historical data to ensure all indicators are based on stable, meaningful values.
-   **Numba JIT Compilation:** The Support and Resistance algorithm is accelerated with Numba for high-speed calculation.
-   **Downcasting:** Automatically reduces the memory footprint of the final `features` dataset by downcasting data types.

---

## 🧮 Example Workflow

1. Bronze Layer generates exhaustive trade combinations.
2. Silver Layer loads bronze data, computes advanced features, and splits into:
    - `features/{instrument}.csv`: Market context per candle.
    - `outcomes/{instrument}.csv`: Trade results with positioning metrics.
3. These files are used by the Gold Layer for ML preprocessing.

## 📈 Example Use Case

Use the Silver Layer outputs to:

-   Analyze market conditions for trade success.
-   Prepare data for ML model training (Gold Layer).

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

This file contains all simulated trades, now enriched with **two sets of powerful positioning features**.

| Column                                 | Description                                                                                                                                                                                                                               |
| :------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `entry_time`                           | **Foreign Key.** Links to `time` in the features file.                                                                                                                                                                                    |
| `trade_type`                           | “buy” or “sell”.                                                                                                                                                                                                                          |
| `entry_price`                          | Entry price at candle close.                                                                                                                                                                                                              |
| `sl_price`, `tp_price`                 | Stop-loss and Take-profit price levels.                                                                                                                                                                                                   |
| `sl_ratio`, `tp_ratio`                 | The relative SL/TP percentages used for this trade.                                                                                                                                                                                       |
| `exit_time`                            | The timestamp when the trade concluded.                                                                                                                                                                                                   |
| `outcome`                              | "win" (or "loss").                                                                                                                                                                                                                        |
| **`..._dist_to_[indicator]_bps`**      | **Distance Feature.** The distance from SL/TP to an indicator in **basis points**. Negative means below, positive means above.                                                                                                            |
| **`..._placement_pct_to_[indicator]`** | **Placement Feature.** The placement of SL/TP as a percentage of the distance from entry to the indicator. A value of `0.75` means 75% of the way there. A value of `-0.2` means 20% of the way in the opposite direction (behind entry). |

---

## 📈 Example Use Case

These two decoupled files are the **correct** inputs for the next stage of the pipeline (`gold_data_generator.py`). The next script will:

1.  Load the `outcomes.csv` to calculate a **win rate for each unique combination of `sl_ratio`, `tp_ratio`, and positioning features**.
2.  Join this win rate onto the `features.csv` using the `time` and `entry_time` columns as the key.
3.  This creates a clean dataset for training a model to predict which market conditions and trade structures lead to a high probability of success.

# 🧠 Silver Layer: Feature Engineering & Enrichment Engine

This Python script serves as the central feature engineering hub of the pipeline. It takes the raw trade simulations from the **Bronze Layer** and enriches them with a vast array of market context features, producing a clean, decoupled, and analysis-ready **Silver Dataset**.

Its core purpose is twofold:

1.  **Decouple Context from Outcomes:** It separates the "market conditions" (`features`) from the "trade results" (`outcomes`) to prevent data leakage and enable robust analysis.
2.  **Enrich Outcomes:** It calculates advanced positioning metrics for every trade, transforming simple SL/TP ratios into sophisticated, relational data points.

---

## ⚙️ How It Works

The script executes a two-step process for each instrument:

1.  **Step 1: Build the `features` Dataset:**

    -   It loads the raw OHLCV data from the `raw_data` directory.
    -   It calculates a comprehensive suite of over 200 technical indicators, candlestick patterns, and custom market regime features for **every single candle**.
    -   This complete market context is saved as a single, clean file to `silver_data/features/`.

2.  **Step 2: Build the Enriched `outcomes` Dataset:**
    -   It reads the massive `bronze_data` file in memory-efficient chunks.
    -   It merges each chunk with the relevant indicator levels from the `features` dataset.
    -   It then calculates **two types of advanced positioning features** for every trade, providing deep relational context.
    -   The final, enriched trade data is written incrementally to a file in `silver_data/outcomes/`.

---

## 📁 Folder Structure

```
project_root/
├── bronze_data/          # INPUT: Raw trade simulations
│   └── AUDUSD1.csv
│
├── silver_data/          # OUTPUT: Parent directory for enriched data
│   ├── features/         # OUTPUT 1: Market context, one row per candle
│   │   └── AUDUSD1.csv
│   │
│   └── outcomes/         # OUTPUT 2: Enriched trades, many rows per candle
│       └── AUDUSD1.csv
│
└── scripts/
    └── silver_data_generator.py
```

---

## 🚀 Advanced Feature Engineering

This script generates a rich set of features, providing the foundation for all subsequent analysis.

| Feature Category            | Examples                                                     | Purpose                                                                                               |
| :-------------------------- | :----------------------------------------------------------- | :---------------------------------------------------------------------------------------------------- |
| **Technical Indicators**    | `SMA`, `EMA`, `RSI`, `MACD`, `Bollinger Bands`, `ATR`, `ADX` | Quantify market momentum, trend, volatility, and volume.                                              |
| **Candlestick Patterns**    | `CDLDOJI`, `CDLENGULFING` (All 61 TA-Lib patterns)           | Capture classic price action signals.                                                                 |
| **Support & Resistance**    | `support`, `resistance` (Numba-accelerated)                  | Identify key historical price levels.                                                                 |
| **Market Regimes**          | `trend_regime`, `vol_regime`, `session`                      | Characterize the overall market state and time-based behavior.                                        |
| **Positioning (Distance)**  | `sl_dist_to_SMA_50_bps`                                      | Measures the raw distance from a SL/TP to an indicator in **basis points**.                           |
| **Positioning (Placement)** | `tp_placement_pct_to_resistance`                             | Measures _where_ a SL/TP is placed on a directional scale from the entry (0%) to an indicator (100%). |

---

## ⚡ Performance & Efficiency

-   **Memory Safety:** The script is designed to handle gigantic `bronze_data` files by processing them in **chunks**, ensuring a low and stable RAM footprint.
-   **Optimized Calculations:** Feature calculations are batched, and the most intensive algorithm (Support & Resistance) is JIT-compiled with `numba` for C-like speed.
-   **Downcasting:** Data types are automatically downcast before saving to reduce file size and speed up future loading times.

---

## 🧱 Output File Descriptions

### 1. `silver_data/features/{instrument}.csv`

Contains the complete market context, with **one unique row for every candle**. This file includes all technical indicators and market regime information.

### 2. `silver_data/outcomes/{instrument}.csv`

Contains every winning trade simulation, now enriched with powerful relational data.

| Column                                 | Description                                                                                                                                                                                                                                                 |
| :------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `entry_time`, `sl_ratio`, etc.         | Core trade information from the Bronze layer.                                                                                                                                                                                                               |
| **`..._dist_to_[indicator]_bps`**      | **Distance Feature.** The distance from the SL/TP to an indicator, in basis points. A negative value means the target is _below_ the indicator.                                                                                                             |
| **`..._placement_pct_to_[indicator]`** | **Placement Feature.** The placement of the SL/TP as a percentage of the distance from the entry to the indicator. This is a directional metric; a value of `-0.2` means the target is placed 20% of the way _behind_ the entry, relative to the indicator. |

**Note:** These two files serve as the complete and necessary inputs for the Platinum Layer, where strategy patterns are discovered.

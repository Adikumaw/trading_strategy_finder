# 🥇 Gold Data Generator (ML-Ready Feature Processor)

This Python script represents the final and most critical step in the data preparation pipeline. Its sole purpose is to take the human-readable `silver_data/features` dataset and transform it into a fully normalized, standardized, and purely numerical **"Gold Dataset"** ready for direct input into machine learning models.

This script completes the feature engineering process by converting all data points into a mathematical format that algorithms can effectively learn from.

---

## ⚙️ How It Works

The script operates exclusively on the `silver_data/features` files and performs a series of ML-specific preprocessing steps in a precise order:

1.  **Convert Absolute Prices to Relative Distances:**

    -   It identifies all columns containing absolute price data (e.g., `SMA_50`, `BB_upper`, `support`, `ATR_level_up_1x`).
    -   For each, it calculates a normalized distance from the current candle's `close` price using the formula: `(close - indicator_price) / close`.
    -   This transforms all price-based features into a scale-independent, relational format (e.g., `SMA_50_dist_norm`).

2.  **Encode Categorical Features:**

    -   It takes high-level categorical columns like `session` and `trend_regime`.
    -   It applies **one-hot encoding** to convert them into a binary format that models can understand.

3.  **Compress Candlestick Patterns:**

    -   The raw output from the `TA-Lib` library for candlestick patterns is a score from -100 to 100.
    -   This script compresses these scores into a simple 5-point scale (`-1.0`, `-0.5`, `0`, `0.5`, `1.0`) to capture the essence of the pattern (strong bearish, bearish, neutral, bullish, strong bullish) while reducing noise.

4.  **Standardize All Numeric Features:**

    -   It applies a `StandardScaler` to all remaining numerical columns.
    -   This crucial step rescales the data to have a **mean of 0 and a standard deviation of 1**, preventing features with larger scales from unfairly dominating the learning process.

5.  **Clean Up:**
    -   Finally, it drops all the original, non-transformed columns (like the absolute price indicators), leaving only the ML-ready feature set. The `time` column is preserved as the primary key for joining.

---

## 📁 New Folder Structure

This script reads from `silver_data/features` and writes its output to a new `gold_data/features` directory.

```
project_root/
│
├── silver_data/
│   ├── features/         # INPUT: Human-readable market features
│   │   └── AUDUSD1.csv
│   │
│   └── outcomes/         # (This data is NOT used by this script)
│       └── AUDUSD1.csv
│
├── gold_data/
│   │
│   └── features/         # OUTPUT: ML-ready, normalized features
│       └── AUDUSD1.csv
│
└── scripts/
    └── gold_data_generator.py   # This script
```

---

## 🧱 Output File Description: `gold_data/features/{instrument}.csv`

The output is a single, powerful feature file where **every column is a machine-ready numerical feature**, except for the `time` key.

| Column Category          | Example Columns                          | Description                                                                             |
| :----------------------- | :--------------------------------------- | :-------------------------------------------------------------------------------------- |
| **`time`**               | `2023-10-27 14:00:00`                    | **Primary Key.** Unchanged from the silver layer, used for joining with outcomes later. |
| **Relational Distances** | `SMA_50_dist_norm`, `BB_upper_dist_norm` | **(Scaled)** The normalized distance from the `close` to key indicators.                |
| **Standard Indicators**  | `RSI_14`, `MACD_hist`, `ADX`             | **(Scaled)** All non-price indicators are standardized (mean 0, std 1).                 |
| **Candlestick Patterns** | `CDLDOJI`, `CDLHAMMER`                   | **(Compressed)** Binned into a `[-1.0, ..., 1.0]` scale.                                |
| **One-Hot Encoded**      | `session_London`, `trend_regime_trend`   | Binary (0 or 1) columns representing the market state.                                  |

**Note:** All original absolute price columns (`open`, `high`, `low`, `close`, `SMA_20`, `support`, etc.) have been **removed** and replaced by their `_dist_norm` counterparts.

---

## 🎯 The Role in the ML Pipeline: Preparing the `X`

This script is intentionally designed to prepare the **features (the `X` matrix)** for a machine learning model. It does **not** process the `silver_data/outcomes` file.

The next logical step, which typically happens within the model training script itself, is:

1.  **Load the `gold_data/features` file.**
2.  **Load the `silver_data/outcomes` file.**
3.  **Create the Target Variable (`y`):** Aggregate the outcomes data to create a target metric. For example, calculating the average win rate for each unique combination of `sl_ratio`, `tp_ratio`, and positioning features.
4.  **Join:** Merge the target variable (`y`) with the feature set (`X`) using the `time` column.

This decoupled approach ensures a clean, modular, and logically sound pipeline, preventing data leakage and making the entire process easier to debug and manage.

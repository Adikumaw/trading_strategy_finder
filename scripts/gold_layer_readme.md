# 🥇 Gold Layer: ML-Ready Feature Processor

This Python script represents the final feature processing stage. Its sole purpose is to take the human-readable `silver_data/features` dataset and transform it into a fully normalized, standardized, and purely numerical **Gold Dataset** ready for direct input into machine learning models.

This script completes the data preparation by converting all market context into a mathematical format that algorithms can effectively learn from.

---

## ⚙️ How It Works

The script operates on the `silver_data/features` files, performing a series of ML-specific preprocessing steps for each file:

1.  **Relational Transformation:** It converts all absolute price columns (e.g., `SMA_50`, `support`) into a normalized distance relative to the candle's `close` price. This makes the features scale-independent.
2.  **Categorical Encoding:** It applies **one-hot encoding** to categorical features like `session` and `trend_regime`, converting them into a binary format.
3.  **Pattern Compression:** It compresses the -100 to 100 scores from `TA-Lib`'s candlestick patterns into a simple 5-point scale (`-1.0` to `1.0`) to capture the signal while reducing noise.
4.  **Standardization:** It applies a `StandardScaler` to all remaining numerical columns, rescaling them to have a mean of 0 and a standard deviation of 1. This is crucial for preventing features with large scales from dominating the ML learning process.
5.  **Cleanup:** Finally, it drops all the original, non-transformed columns, leaving only the ML-ready feature set alongside the `time` primary key.

---

## ⚡ Performance & Parallel Processing

This script is optimized for speed by processing multiple instrument files in parallel.

-   **Multiprocessing:** It uses Python's `multiprocessing.Pool` to assign each `silver_features` file to a different CPU core. This provides a significant speed-up when you have generated features for multiple instruments.
-   **User Configurable:** The script will ask you how many CPU cores you wish to use, allowing you to balance speed with system resource usage.

---

## 📁 Folder Structure

```
project_root/
├── silver_data/
│   └── features/         # INPUT: Human-readable market features
│       └── AUDUSD1.csv
│
├── gold_data/
│   └── features/         # OUTPUT: ML-ready, normalized features
│       └── AUDUSD1.csv
│
└── scripts/
    └── gold_data_generator.py   # This script
```

---

## 🧱 Output File Description: `gold_data/features/{instrument}.csv`

The output is a powerful feature file where every column is a numerical, machine-ready feature (except for the `time` key).

| Column Category          | Example Columns                         | Description                                                          |
| :----------------------- | :-------------------------------------- | :------------------------------------------------------------------- |
| **`time`**               | `2023-10-27 14:00:00`                   | **Primary Key.** For joining with target data.                       |
| **Relational Distances** | `SMA_50_dist_norm`, `support_dist_norm` | **(Scaled)** Normalized distances from `close`.                      |
| **Standard Indicators**  | `RSI_14`, `ADX`, `BB_width`             | **(Scaled)** All non-price indicators standardized to mean 0, std 1. |
| **Candlestick Patterns** | `CDLDOJI`, `CDLHAMMER`                  | **(Compressed)** Binned into a `[-1.0, ..., 1.0]` scale.             |
| **One-Hot Encoded**      | `session_London`, `trend_regime_trend`  | Binary (0 or 1) columns representing the market state.               |

---

## 🎯 The Role in the ML Pipeline

This script prepares the **features (the `X` matrix)** for the Platinum Layer. The `gold_data/features` files are merged with the pre-computed target data (`platinum_data/targets`) to create the final training sets for the strategy discovery models.

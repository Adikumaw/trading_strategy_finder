# 🧠 Silver Data Generator (AI Feature Enrichment Pipeline)

This Python script transforms the basic trade simulations from the **"Bronze Dataset"** into a feature-rich **"Silver Dataset"** ready for advanced machine learning analysis. It acts as the critical bridge between raw trade outcomes and intelligent strategy discovery.

The script's core purpose is to enrich each simulated trade with a deep snapshot of the market's context at the moment of entry, asking: **_What was the market doing and looking like when this trade was initiated?_**

---

## ⚙️ How It Works

1.  **Matching Files:** The script finds corresponding files in `raw_data/` and `bronze_data/` (e.g., `EURUSD1.csv` in both).
2.  **Historical Feature Calculation:** It first loads the entire `raw_data` (OHLCV) file to calculate a vast array of historical technical indicators and patterns. This is done once per file and cached in memory for efficiency.
3.  **Chunk Processing:** It reads the potentially massive `bronze_data` file in manageable chunks to avoid memory overload.
4.  **Contextual Merging:** For each trade in a chunk, it looks up the `entry_time` and merges the entire pre-calculated market context (all indicators, patterns, etc.) onto that trade's row.
5.  **Relational Feature Generation:** This is the script's most powerful step. After merging, it calculates **new relational features** that describe where the trade's Stop-Loss (SL) and Take-Profit (TP) levels are positioned relative to key market structures (like Support/Resistance, Bollinger Bands, and moving averages).
6.  **Saving the Silver Dataset:** The final, enriched data for each file is appended to a new CSV in the `silver_data/` directory.

---

## 📁 Folder Structure

This script consumes data from `raw_data` and `bronze_data` to produce its output in `silver_data`.

```
project_root/
│
├── raw_data/             # INPUT: Raw OHLCV data (no header)
│   ├── EURUSD1.csv
│   └── ...
│
├── bronze_data/          # INPUT: Trade simulations from the bronze script
│   ├── EURUSD1.csv
│   └── ...
│
├── silver_data/          # OUTPUT: ML-ready enriched datasets (auto-created)
│   ├── EURUSD1.csv
│   └── ...
│
└── scripts/
    └── silver_data_generator.py   # This script
```

---

## 🧩 Input Data Format

The script requires two input files for each instrument:

1.  **`bronze_data/{instrument}.csv`**:
    -   Must contain a header row.
    -   Expected columns include `entry_time`, `sl_price`, `tp_price`, etc.
2.  **`raw_data/{instrument}.csv`**:
    -   Must **NOT** contain a header row.
    -   Expected columns are `time, open, high, low, close, [volume]`.

The script will fail if the corresponding raw data file for a bronze file is missing.

---

## 🚀 Feature Engineering Engine

This script is a comprehensive feature generation powerhouse. For every single trade, it calculates:

| Feature Category           | Examples                                                                      | Purpose                                                                 |
| :------------------------- | :---------------------------------------------------------------------------- | :---------------------------------------------------------------------- |
| **Technical Indicators**   | `SMA`, `EMA`, `RSI`, `MACD`, `Bollinger Bands`, `ATR`, `ADX`, `CCI`, `OBV`    | Quantify market momentum, trend, volatility, and volume dynamics.       |
| **Candlestick Patterns**   | `CDLDOJI`, `CDLENGULFING`, `CDLHAMMER`, etc. (Dozens from `talib`)            | Capture classic price action signals and potential reversals.           |
| **Support & Resistance**   | `support`, `resistance` (calculated with a fast, Numba-accelerated algorithm) | Identify key historical price levels that might influence future price. |
| **Price Action & Regimes** | `bullish_ratio_last_N`, `avg_body_last_N`, `trend_regime`, `vol_regime`       | Characterize recent price behavior and the overall market state.        |
| **Time-Based Features**    | `session` (Asian, London, NY), `hour`, `weekday`                              | Allow the model to find patterns related to time of day or week.        |

---

## ✨ The Secret Sauce: Relational Features

Beyond standard indicators, this script generates novel features that give a model true market structure awareness. Instead of just knowing a trade's SL/TP ratio, the model learns **where the SL and TP are placed in relation to the market.**

This helps answer critical questions like:

-   _Is the Stop Loss safely behind a strong support level or exposed in open space?_
-   _Is the Take Profit aiming for a realistic target just before a resistance level, or is it unlikely to be hit?_
-   _How far is the TP from the upper Bollinger Band?_

#### Example Generated Features:

-   `sl_dist_to_support_norm`
-   `tp_dist_to_resistance_norm`
-   `sl_dist_to_bb_upper_norm`
-   `tp_dist_to_sma_50_norm`
-   ...and many more for all key indicators.

These features are **normalized by the closing price**, making them comparable across different assets and timeframes.

---

## ⚡ Performance & Efficiency

-   **Chunking:** Processes massive bronze files in `1,000,000` row chunks to operate on systems with limited RAM.
-   **Indicator Warmup:** Skips the first `200` candles of historical data to ensure that indicators have enough data to produce stable, meaningful values.
-   **Numba JIT Compilation:** The Support and Resistance algorithm is accelerated with Numba for high-speed calculation on large datasets.
-   **Downcasting:** Automatically reduces the memory footprint of the final dataset by downcasting data types (e.g., `float64` to `float32`).

---

## 🧮 Example Workflow

1.  **Prerequisite:** Ensure you have already run `bronze_data_generator.py` and have populated the `bronze_data` folder.

2.  **Run the script:**

    ```bash
    python scripts/silver_data_generator.py
    ```

3.  **Monitor the output:** The script will process each file pair it finds.

    ```
    =========================
    Processing: AUDUSD1.csv
    =========================
    Loading raw OHLCV to determine indicator warmup period...
    Calculating all historical features...
    ✅ Historical features calculated and cached in memory.
    ...
    Merging Bronze Chunks: 100%|██████████| 5/5 [00:30<00:00, 6.00s/it]

    ✅ Success! Rich dataset with all chunks merged saved to AUDUSD1.csv
    ```

4.  **Final Output:** The `silver_data` folder will now contain CSVs with a massive number of columns, ready for analysis.

---

## 🧱 Output File Description

Each output CSV in `silver_data/` is an extremely wide file containing every column from the original bronze data file, plus dozens of new feature columns.

**Structure:** `[Original Bronze Columns] + [Market Data Columns] + [Indicator Columns] + [Pattern Columns] + [Relational Feature Columns]`

#### A Small Sample of Added Columns:

| Column                         | Description                                                                  |
| :----------------------------- | :--------------------------------------------------------------------------- |
| `open`, `high`, `low`, `close` | The OHLC of the entry candle.                                                |
| `SMA_20`, `SMA_200`            | 20-period and 200-period Simple Moving Averages.                             |
| `RSI_14`                       | The 14-period Relative Strength Index.                                       |
| `CDLENGULFING`                 | A flag (100 for bullish, -100 for bearish) if an engulfing pattern occurred. |
| `support`                      | The most recent valid support level.                                         |
| `trend_regime`                 | The market state ('trend' or 'range') based on the ADX.                      |
| `sl_dist_to_support_norm`      | The normalized price distance from the SL to the support level.              |
| `tp_dist_to_sma_100_norm`      | The normalized price distance from the TP to the 100-period SMA.             |
| ...and over 100 more.          |                                                                              |

---

## 📈 Example Use Case

The silver dataset is the ideal input for training supervised machine learning models (e.g., RandomForest, XGBoost, Neural Networks) to predict the `outcome` ('win' or 'loss') of a trade based on the rich market context provided.

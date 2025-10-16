# 🥈 Silver Layer: The Enrichment Engine

This script is the **central feature engineering hub** of the entire pipeline. It takes the raw, high-volume trade simulations from the Bronze Layer and transforms them into an intelligent, context-rich dataset ready for machine learning and pattern discovery.

Its purpose is to add the "why" to the "what." The Bronze layer tells us _what_ trades were profitable; the Silver layer tells us _what the market looked like_ at the moment each of those trades was initiated.

## What It Is

The Silver Layer is a sophisticated data processing script that operates in two distinct, sequential stages for each instrument:

1.  **Market Feature Generation:** It first consumes the raw OHLC price data and calculates a massive suite of over 200 technical indicators, candlestick patterns, and custom market context features for _every single candle_ in the historical data. This creates a complete, candle-by-candle "fingerprint" of the market's state over time.
2.  **Trade Enrichment & Chunking:** It then reads the enormous Bronze Dataset in memory-safe chunks. For each winning trade, it merges the pre-calculated market features from the entry candle. Crucially, it then calculates a powerful set of **relational positioning features**, which describe _where_ a trade's SL and TP were placed relative to the market structures at that time.

The final output is not a single large file, but two distinct sets of data: a clean market features file and a directory of enriched, chunked outcome files.

## Why It Is

Raw trade data is insufficient for discovering strategies. A winning trade is not just an entry and an exit; it's an action taken within a specific market context. The Silver Layer is critical for two reasons:

1.  **Preventing Lookahead Bias:** By generating all market features _before_ enriching the trades, and then using a point-in-time merge (`merge_asof`), we guarantee that the features associated with a trade were known at the time of entry. This is a critical step to prevent data leakage and ensure the integrity of the downstream machine learning models.
2.  **Creating Deep Contextual Features:** This is the most innovative part of the layer. Instead of just knowing that `RSI` was `25`, our system now knows that "the TP was placed 90% of the way to the daily resistance" or "the SL was placed just 10 basis points behind the lower Bollinger Band." This transforms the problem from simple indicator-checking to a much more sophisticated analysis of trade structure relative to market structure.

## How It Works

The script is a highly optimized, two-step process designed for performance and memory safety.

1.  **Step 1: Build the `features` Dataset:**

    - It loads the raw OHLCV data from the `raw_data` directory.
    - It efficiently calculates features in batches:
      - **Indicators:** Standard indicators like `SMA`, `EMA`, `RSI`, `MACD`, `Bollinger Bands`, and `ATR` are calculated using the `ta` library.
      - **Candlestick Patterns:** All 61 classic candlestick patterns are generated using the `talib` library.
      - **Support & Resistance:** Key price levels are identified using a fractal-based algorithm, which is JIT-compiled with `numba` for C-like speed.
      - **Market Regimes:** Custom features like trading session, day of the week, and market state (trending/ranging, high/low volatility) are derived.
    - This complete market context is saved as a single, clean file to `silver_data/features/`.

2.  **Step 2: Build the Enriched `outcomes` Chunks:**
    - It reads the massive `bronze_data` file in manageable chunks (e.g., 500,000 rows at a time) to maintain a low memory footprint.
    - For each chunk of trades, it performs a `pandas.merge_asof` to join the market features that were valid at each trade's `entry_time`.
    - It then calculates the **positioning features** for every trade, providing deep relational context.
    - The final, enriched trade data is written out directly into a new `chunk_X.csv` file in the `silver_data/chunked_outcomes/{instrument}/` directory. This avoids creating another single, multi-gigabyte file and prepares the data perfectly for the parallel processing required by the Platinum Layer.

## 📁 Folder Structure

```
project_root/
├── bronze_data/                # INPUT: Raw trade simulations
│   └── XAUUSD15.csv
│
├── raw_data/                   # INPUT: Raw OHLC data
│   └── XAUUSD15.csv
│
├── silver_data/                # OUTPUT: Parent directory for enriched data
│   ├── features/               # OUTPUT 1: Market context, one row per candle
│   │   └── XAUUSD15.csv
│   │
│   └── chunked_outcomes/       # OUTPUT 2: Enriched trades, split into chunks
│       └── XAUUSD15/
│           ├── chunk_1.csv
│           └── chunk_2.csv
│
└── scripts/
    └── silver_data_generator.py   # This script
```

## 📈 Input & Output

### Input Files

1.  **`bronze_data/{instrument}.csv`**: The massive CSV of winning trades from the Bronze Layer.
2.  **`raw_data/{instrument}.csv`**: The original OHLC data for the same instrument.

### Output Files

1.  **`silver_data/features/{instrument}.csv`**: A CSV where each row corresponds to a single candle and columns represent all calculated market features (indicators, patterns, regimes).
2.  **`silver_data/chunked_outcomes/{instrument}/`**: A directory containing the enriched trade data, split into multiple numbered `chunk_X.csv` files. Each row is a winning trade from the Bronze layer, now augmented with dozens of new **positioning features**.

#### Positioning Feature Examples:

| New Column Name                  | Description                                                                                                                                                                                                                                                              |
| :------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sl_dist_to_SMA_50_bps`          | The distance from the Stop-Loss price to the 50-period SMA, measured in basis points (1/100th of a percent).                                                                                                                                                             |
| `tp_placement_pct_to_resistance` | Where the Take-Profit was placed on a scale from the entry price (0%) to the resistance level (100%). A value of `1.1` means the TP was 10% _beyond_ the resistance. A value of `-0.2` means it was placed 20% of the way _behind_ the entry, relative to the indicator. |

## 🚀 Possible Enhancements & Future Improvements

1.  **More Advanced Features:** The engine can be expanded with more sophisticated features, such as:
    - **Fundamental Data:** Merging news event timestamps or sentiment scores.
    - **Order Flow Data:** Incorporating metrics from volume profiles or order book imbalances if such data is available.
    - **Intermarket Correlation:** Calculating the correlation of the instrument with other assets (e.g., EURUSD vs. DXY index) as a feature.
2.  **Feature Selection:** The script currently generates over 200 features, many of which may be redundant or noisy. A future version could incorporate a preliminary feature selection step (e.g., using correlation analysis or a simple model's feature importance) to produce a more refined, less-dimensional `features` dataset.
3.  **Configurable Feature Generation:** Similar to the Bronze layer, the lists of indicator periods (`SMA_PERIODS`, `EMA_PERIODS`) could be moved to an external configuration file to allow for easier experimentation without modifying the code.

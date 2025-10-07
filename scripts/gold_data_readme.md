# 🏅 Gold Data Generator (ML Preprocessing & Normalization Pipeline)

This Python script represents the final and most critical step in the data preparation pipeline. It takes the feature-rich **"Silver Dataset"** and transforms it into a fully normalized, ML-ready **"Gold Dataset"**.

The primary goal of this script is to convert all features, especially market indicators, from absolute price values into a **relational and standardized format**. This process is essential for allowing machine learning models to discover meaningful patterns that are independent of an asset's current price.

---

## ⚙️ How It Works

1.  **Scan for Silver Files:** The script automatically scans the `silver_data/` directory for all `.csv` files to process.
2.  **Separate Target Variable:** For each file, it immediately isolates the `outcome` column, which will be our prediction target (`y`), and converts it into a binary format (`1` for 'win', `0` for 'loss').
3.  **Price-Relative Transformation:** This is the script's core function. It programmatically identifies all columns that contain absolute price values (e.g., `open`, `high`, `low`, `close`, `SMA_20`, `support`, `BB_upper`). For each, it creates a new relational feature by calculating its normalized distance from the entry candle's closing price.
4.  **Specialized Feature Handling:** It intelligently processes different feature types:
    -   **Categorical Data** (like `session`, `trade_type`) is **One-Hot Encoded**.
    -   **Candlestick Patterns** (`CDL*` columns) are **bucketized** into a simple scale (`-1.0` to `1.0`) but are **not scaled further** to preserve their sparse, event-driven nature.
5.  **Two-Pass Normalization:** To handle massive files, it uses a robust two-pass approach:
    -   **Pass 1:** It iterates through the entire file in chunks to learn the global mean and standard deviation of all numeric features (excluding candlestick patterns).
    -   **Pass 2:** It iterates through the file again, applying the learned scaling to every chunk, ensuring consistent normalization across the entire dataset.
6.  **File Size Optimization:** After scaling, all high-precision numbers are downcasted to `float32`, significantly reducing the final file size without losing meaningful information.
7.  **Save the Gold Dataset:** The final, fully processed data is saved to a corresponding file in the `gold_data/` directory.

---

## 📁 Folder Structure

This script completes the data pipeline, consuming from `silver_data` and producing the final `gold_data`.

```
project_root/
│
├── silver_data/          # INPUT: Feature-rich data from the silver script
│   ├── AUDUSD1.csv
│   └── ...
│
├── gold_data/            # OUTPUT: Fully normalized, ML-ready datasets (auto-created)
│   ├── AUDUSD1.csv
│   └── ...
│
└── scripts/
    └── gold_data_generator.py   # This script
```

---

## ✨ The Core Transformation Logic

A machine learning model cannot effectively learn from a feature like `SMA_20 = 1.2345`. This value is meaningless without knowing the current price. The model needs to know if the price is _above_ or _below_ the SMA and by _how much_.

This script solves that problem by converting every absolute price feature into a relational one using the formula:
**`new_feature = (close_price - feature_price) / close_price`**

| Original Column (Absolute) | New Column (Relational) | Example Interpretation                                                  |
| :------------------------- | :---------------------- | :---------------------------------------------------------------------- |
| `SMA_20`                   | `SMA_20_dist_norm`      | A value of `-0.005` means the close was 0.5% _below_ the SMA_20.        |
| `support`                  | `support_dist_norm`     | A value of `0.001` means the close was 0.1% _above_ the support level.  |
| `BB_upper`                 | `BB_upper_dist_norm`    | A negative value indicates the price is below the upper Bollinger Band. |

This transformation is applied to **all** price-based indicators, preserving their information while making them vastly more powerful for pattern recognition.

---

## 📊 Feature Processing Breakdown

The script intelligently handles different types of data according to best practices:

| Feature Type                | How It's Processed                                                           | Reason                                                                                  |
| :-------------------------- | :--------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------- |
| **Absolute Price Features** | Converted to relational distance from `close`; originals are dropped.        | Makes features independent of the asset's price, focusing on structure.                 |
| **Categorical Features**    | **One-Hot Encoded** (e.g., `session_London`, `trade_type_sell`).             | Prevents the model from assuming a false ordinal relationship between categories.       |
| **Candlestick Patterns**    | **Bucketized** into a `[-1, -0.5, 0, 0.5, 1]` scale. **Not Scaled Further.** | Simplifies the feature space and preserves the powerful, sparse "event" signal.         |
| **Other Numeric Features**  | **Standard Scaled** (Z-score) after all other transformations.               | Puts all final features on the same scale, improving model convergence and performance. |

---

## ⚡ Performance & Efficiency

-   **Automated Batch Processing:** Automatically finds and processes all `.csv` files in the `silver_data` directory.
-   **Chunking:** Reads and writes large files in `500,000` row chunks to ensure it can run on systems with limited RAM.
-   **Robust Two-Pass Scaling:** Guarantees that normalization is calculated based on the entire dataset's statistics, providing consistent and accurate scaling even on huge files.
-   **Memory Optimization:** Downcasts final data to `float32` to dramatically reduce disk space usage of the output files.

---

## 🧮 Example Workflow

1.  **Prerequisite:** Ensure you have successfully run `silver_data_generator.py`.

2.  **Run the script:**

    ```bash
    python scripts/gold_data_generator.py
    ```

3.  **Monitor the output:**

    ```
    Found 3 silver file(s) to process.

    =========================
    Processing: AUDUSD1.csv
    =========================
    Pass 1: Fitting scaler on transformed data...
    Fitting Scaler: 100%|██████████| 18/18 [04:11<00:00, 13.97s/it]

    Pass 2: Transforming data and saving to gold file...
    Transforming Chunks: 100%|██████████| 18/18 [04:30<00:00, 15.01s/it]

    ✅ Success! Normalized and size-optimized 'Gold' data saved to: gold_data/AUDUSD1.csv
    ...
    ==================================================
    ✅ All gold data generation complete.
    ```

---

## 🧱 Output File Description

The output "Gold" CSV is the final artifact, ready to be loaded directly into a machine learning framework. It is **fully numeric** and **standardized**.

-   All original price-based indicators are replaced by their `_dist_norm` counterparts.
-   All categorical text is replaced by binary `_category` columns.
-   The `outcome` column is the final column, encoded as `1` for a win and `0` for a loss.

#### A Sample of Final Columns:

| Column                           | Description                                                                                     |
| :------------------------------- | :---------------------------------------------------------------------------------------------- |
| `sl_ratio`, `tp_ratio`, `RSI_14` | The original ratios and oscillators, now Z-score standardized.                                  |
| `CDLENGULFING`                   | Bucketized pattern signal (`1.0`, `-1.0`, `0.0`, etc.). **Not scaled.**                         |
| `sl_dist_to_support_norm`        | The pre-calculated relational feature, now Z-score standardized.                                |
| `open_dist_norm`                 | **New Feature:** The normalized distance from close to open, Z-score standardized.              |
| `SMA_50_dist_norm`               | **New Feature:** The normalized distance from close to the 50-period SMA, Z-score standardized. |
| `session_London`                 | **New Feature:** Binary flag (`1` or `0`) from One-Hot Encoding.                                |
| `trade_type_sell`                | **New Feature:** Binary flag (`1` or `0`) from One-Hot Encoding.                                |
| `outcome`                        | The binary target variable (`1` for win, `0` for loss).                                         |

---

## 🧠 Example Use Case

This gold dataset is the ideal input for training any supervised classification model to predict trade outcomes.

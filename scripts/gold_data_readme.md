# 🏅 Gold Data Generator (ML Preprocessing & Normalization Pipeline)

This Python script represents the final and most critical step in the data preparation pipeline. It takes the feature-rich **"Silver Dataset"** and transforms it into a fully normalized, ML-ready **"Gold Dataset"**.

The primary goal of this script is to convert all features, especially market indicators, from absolute price values into a **relational and standardized format**. This process is essential for allowing machine learning models to discover meaningful patterns that are independent of the asset's current price.

---

## ⚙️ How It Works

1.  **Load Silver Data:** The script loads a specified CSV file from the `silver_data/` directory.
2.  **Separate Target Variable:** It immediately isolates the `outcome` column, which will be our prediction target (`y`), and converts it into a binary format (`1` for 'win', `0` for 'loss').
3.  **Price-Relative Transformation:** This is the script's core function. It programmatically identifies all columns that contain absolute price values (e.g., `open`, `high`, `low`, `close`, `SMA_20`, `support`, `BB_upper`). For each, it creates a new relational feature by calculating its normalized distance from the entry candle's closing price.
4.  **Drop Original Price Columns:** After creating the new relational features, the original, absolute-price columns are dropped, as their information is now captured in a superior, normalized format.
5.  **Encode Categorical Data:** It finds all categorical columns (like `session`, `trade_type`) and applies **One-Hot Encoding**, creating new binary columns for each category. This is the standard best practice for ML.
6.  **Process Candlestick Patterns:** All `CDL*` columns are "bucketized" into a simple five-point scale (`-1.0, -0.5, 0.0, 0.5, 1.0`) to make them more effective as features.
7.  **Final Scaling:** The script takes all remaining numeric columns—including the newly created relational features and pre-normalized indicators like `RSI`—and applies a `StandardScaler` (Z-score normalization). This ensures all features are on a level playing field, which is crucial for the performance of many ML algorithms.
8.  **Save the Gold Dataset:** The final, fully processed features (`X`) are combined with the binary target (`y`) and saved as a new CSV file in the `gold_data/` directory.

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
│   ├── gold_AUDUSD1.csv
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

| Feature Type                | How It's Processed                                                    | Reason                                                                                  |
| :-------------------------- | :-------------------------------------------------------------------- | :-------------------------------------------------------------------------------------- |
| **Absolute Price Features** | Converted to relational distance from `close`; originals are dropped. | Makes features independent of the asset's price, focusing on structure.                 |
| **Categorical Features**    | **One-Hot Encoded** (e.g., `session_London`, `trade_type_sell`).      | Prevents the model from assuming a false ordinal relationship between categories.       |
| **Candlestick Patterns**    | **Bucketized** into a `[-1, -0.5, 0, 0.5, 1]` scale.                  | Simplifies the feature space and reduces noise for the model.                           |
| **Other Numeric Features**  | **Standard Scaled** (Z-score) after all other transformations.        | Puts all final features on the same scale, improving model convergence and performance. |

---

## 🧮 Example Workflow

1.  **Prerequisite:** Ensure you have successfully run `silver_data_generator.py`.

2.  **Configure (Optional):** Open `gold_data_generator.py` and change the `silver_file` variable at the bottom if you want to process a file other than `AUDUSD1.csv`.

3.  **Run the script:**

    ```bash
    python scripts/gold_data_generator.py
    ```

4.  **Monitor the output:**

    ```
    Loading silver data from: AUDUSD1.csv...
    Loaded 99800 rows with 165 columns.
    Transforming absolute price features into normalized relational features...
    Created 85 new relational features.
    Dropped 95 original absolute price and identifier columns.
    Applying One-Hot Encoding to categorical features...
    Compressing candlestick pattern features...
    Applying StandardScaler to all 148 numeric features...

    ==================================================
    ✅ Success! Normalized 'Gold' data saved to: gold_data/gold_AUDUSD1.csv
    💡 This dataset is now fully preprocessed and ready for model training.
    ==================================================
    ```

---

## 🧱 Output File Description

The output "Gold" CSV is the final artifact, ready to be loaded directly into a machine learning framework.

-   **Fully Numeric:** All columns are numeric.
-   **No Data Leakage:** All identifiers, timestamps, and absolute price levels have been removed.
-   **Binary Target:** The `outcome` column is the last column, encoded as `1` for a win and `0` for a loss.
-   **New Column Names:** You will see new column names reflecting the transformations, such as:
    -   `SMA_50_dist_norm`
    -   `resistance_dist_norm`
    -   `session_London`
    -   `trade_type_sell`
    -   `trend_regime_trend`

---

## 🧠 Example Use Case

This gold dataset is the ideal input for training any supervised classification model to predict trade outcomes.

# Gold Layer: The ML Preprocessor (`gold_data_generator.py`)

This script is the crucial bridge between human-readable market analysis and machine learning. Its sole purpose is to take the context-rich `.csv` feature files from the Silver Layer and transform them into a purely numerical, standardized, and ML-optimized dataset, which it saves in the efficient Parquet format.

## 🎯 Purpose in the Pipeline

The Gold Layer acts as a specialized **translator**. Machine learning models, particularly tree-based models and neural networks, cannot interpret raw prices or text-based categories like "London Session". They require a strictly numerical, clean, and well-scaled dataset. This script performs the four essential preprocessing steps required to create such a dataset.

The output of this layer is the final set of "predictors" or "features" that will be fed into the Platinum Layer's machine learning model to discover trading rules.

---

## ✨ Key Features & Transformations

This script is a pipeline of four distinct, sequential machine learning preprocessing steps:

1.  **Relational Transformation:**

    - **What it does:** Converts all absolute price levels (e.g., `SMA_50 = 1.1250`) into a normalized distance from the current close price (e.g., `SMA_50_dist_norm = -0.0021`).
    - **Why it's crucial:** This makes the features **scale-invariant**. A model can learn that a price bouncing off a moving average is significant, regardless of whether the instrument is trading at $1.12 or $2,000. It captures the _relationship_ between price and indicators, not their absolute values.

2.  **Categorical Encoding (One-Hot Encoding):**

    - **What it does:** Converts text-based features like `session_London` or `trend_regime_trend` into new binary (0/1) columns.
    - **Why it's crucial:** ML models cannot process text. This transformation turns abstract categories into a numerical format the model can use to find patterns, such as "is the London/NY overlap more profitable?".

3.  **Candlestick Pattern Compression:**

    - **What it does:** Takes the raw scores from the `talib` library (which range from -200 to +200) and bins them into a simple 5-point scale: `{-1.0, -0.5, 0, 0.5, 1.0}`.
    - **Why it's crucial:** The raw scores are often noisy. A score of 120 vs 130 is mathematically different but represents the same event: "a bullish engulfing pattern occurred". Compression reduces this noise, helping the model focus on the signal (the presence of a strong or weak pattern) rather than insignificant score variations.

4.  **Standardization (Scaling):**
    - **What it does:** Uses `sklearn.preprocessing.StandardScaler` to rescale all remaining numerical features so they have a mean of 0 and a standard deviation of 1.
    - **Why it's crucial:** Features like ATR and RSI have vastly different natural scales. Without standardization, features with larger numerical ranges could unfairly dominate the learning process. Scaling puts all features on a level playing field.

---

## ⚙️ How It Works: The Logic

1.  **File Discovery:** The script scans the `/silver_data/features` directory for new `.csv` files that do not yet have a corresponding `.parquet` file in the `/gold_data/features` directory.
2.  **Serial Processing:** It processes one file at a time. Since the transformations are computationally fast and memory-bound, parallel processing is not necessary.
3.  **Load Data:** It loads a single Silver features `.csv` file into a Pandas DataFrame.
4.  **Execute Pipeline:** The `create_gold_features` function orchestrates the four transformation steps described above in the correct sequence.
5.  **Save Output:** The final, fully transformed, purely numerical DataFrame is saved as a `.parquet` file in the `/gold_data/features` directory. This format ensures fast loading in the next stage of the pipeline.

---

## 🛠️ Dependencies

This script requires several specialized libraries. Install them via pip:

````bash
pip install pyarrow scikit-learn```

---

## 🚀 Usage

Execute the script from the root directory of the project.

**1. Interactive Mode (Recommended):**

The script will scan for new files and present a menu.

```bash
python scripts/gold_data_generator.py
````

**2. Targeted Mode:**

To process a specific file from the `/silver_data/features` directory, pass its name as an argument.

```bash
python scripts/gold_data_generator.py EURUSD15.csv
```

---

## 📄 Output

The script generates a `.parquet` file in the `/gold_data/features/` directory for each Silver feature file it processes. This Parquet file contains a purely numerical, ML-ready dataset with the same number of rows as the input.

The columns will consist of:

- The original `time` column.
- Normalized distance features (e.g., `SMA_50_dist_norm`).
- One-hot encoded binary features (e.g., `session_London`, `session_New_York`).
- Compressed candlestick pattern scores (e.g., `CDLEngulfing`).
- Standardized numerical features (e.g., `RSI_14`, `ADX_14`).

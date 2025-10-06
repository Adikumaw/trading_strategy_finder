# Quantitative Strategy Discovery Engine

This project is a complete, end-to-end Python pipeline for the automated discovery, validation, and ranking of quantitative trading strategies from raw market data. Instead of manually testing ideas, this engine brute-forces the discovery process by identifying every profitable trade in historical data and then uses machine learning to find recurring patterns and inefficiencies.

The entire workflow is designed to move data through a "quality funnel," starting with raw data and ending with a ranked list of robust, cross-asset validated trading strategies.

## The Data Funnel: From Raw to Platinum

The project is structured around a multi-tier data pipeline. Each step refines the data and brings us closer to a usable strategy.

[Raw Data] ==> (Bronze Script) ==> [Bronze Data] ==> (Silver Script) ==> [Silver Data] ==> (Gold Script) ==> [Gold Data] ==> (Platinum Script) ==> [Platinum Data]

-   **Raw Data:** Your initial input. Simple CSV files containing OHLC (Open, High, Low, Close) price data for various financial instruments.
-   **Bronze Data:** The "Ground Truth". This dataset contains a record of **every single theoretically profitable trade** that could have been made from the raw data, including every possible permutation of Stop Loss and Take Profit.
-   **Silver Data:** The "Enriched Dataset". Here, each profitable trade from the Bronze data is enriched with hundreds of descriptive features: every standard indicator (RSI, MACD, etc.), candlestick pattern, and custom-built relative-value metrics that describe the market conditions at the exact moment of entry.
-   **Gold Data:** The "Strategy Candidates". This is the output of the machine learning discovery engine. The script analyzes the Silver data to find the most predictive features and then tests thousands of combinations to find recurring patterns. The result is a ranked list of promising strategy **hypotheses**.
-   **Platinum Data:** The "Validated Edges". This is the final and most valuable output. The script takes the strategy hypotheses from the Gold data and subjects them to a rigorous out-of-sample backtest across multiple different financial instruments. Only the strategies that prove to be profitable and robust across these diverse datasets make it to the Platinum tier.

---

## Project Structure

For the scripts to work correctly, your project must be organized with the following folder structure:

```
/trading_strategy_finder/
├── raw_data/
│   ├── EURUSD15.csv
│   └── GBPUSD15.csv
├── test_data/
│   ├── AUDUSD15.csv
│   ├── USDJPY15.csv
│   └── XAUUSD15.csv
├── bronze_data/
├── silver_data/
├── gold_data/
├── platinum_data/
├── reports/
└── scripts/
    ├── 1_create_bronze_data.py
    ├── 2_create_silver_data.py
    ├── 3_create_gold_data.py
    └── 4_create_platinum_data.py
```

---

## Installation & Setup

This project uses Python. Ensure you have the necessary libraries installed.

1. **Clone the repository or set up the folder structure manually.**

2. **Install Dependencies:**
   It is highly recommended to use a virtual environment.

    ```bash
    pip install pandas numpy tqdm
    pip install ta
    pip install TA-Lib
    pip install numba
    pip install lightgbm
    pip install backtesting bokeh
    ```

    > **Note:** `TA-Lib` can be tricky to install. You may need to install the underlying C library first or find a pre-compiled wheel (`.whl` file) for your system and Python version.

---

## How to Use the Pipeline: A Step-by-Step Guide

Follow these steps in order to run the entire discovery and validation workflow.

### Step 1: Prepare Your Data

-   **`raw_data` folder:** Place the CSV file(s) you want to discover strategies on here. For example, if you want to find strategies for `EURUSD 15-minute`, place a long historical `EURUSD15.csv` file in this folder. The file should have 5 columns: `time, open, high, low, close`.
-   **`test_data` folder:** Place 3-5 CSV files of **different instruments** but the **same timeframe** here. For example, if you are discovering on `EURUSD15.csv`, this folder should contain files like `GBPUSD15.csv`, `AUDUSD15.csv`, `USDJPY15.csv`, etc. This is crucial for the final robustness test.

### Step 2: Create Bronze Data (The Ground Truth)

This script finds all possible profitable trades. It is multi-processed and can handle large files.

-   **Script:** `scripts/create_bronze_data.py`
-   **Action:** Run the script from the terminal.
    ```bash
    cd scripts/
    python 1_create_bronze_data.py
    ```
-   **Output:** It will populate the `bronze_data` folder with files corresponding to your raw data.

### Step 3: Create Silver Data (Enrichment)

This script adds hundreds of indicators and features to the Bronze data.

-   **Script:** `scripts/2_create_silver_data.py`
-   **Action:** Run the script from the terminal.
    ```bash
    python 2_create_silver_data.py
    ```
-   **Output:** It will populate the `silver_data` folder. These files will be very large.

### Step 4: Create Gold Data (Strategy Discovery)

This is the machine learning core of the project. It will analyze the huge Silver file, find the most important features, and test thousands of combinations to generate a ranked list of strategy candidates. This script processes data in chunks to handle files larger than your computer's RAM.

-   **Script:** `scripts/3_create_gold_data.py`
-   **Action:** You can configure the parameters inside the script (e.g., `N_TOP_FEATURES`, `STRATEGY_COMPLEXITY`) to control the search depth. Then, run it.
    ```bash
    python 3_create_gold_data.py
    ```
-   **Output:** It will create files in the `gold_data` folder. Each file is a CSV containing a ranked list of strategy _hypotheses_ ready for validation.

### Step 5: Create Platinum Data (Validation & Final Ranking)

This is the final step. The script takes a Gold file, systematically backtests every strategy candidate against all the files in your `test_data` folder, and saves only the strategies that prove to be robust and profitable across multiple instruments.

-   **Script:** `scripts/4_create_platinum_data.py`
-   **Action:**
    1. Open the script and configure the two main variables at the top:
        ```python
        GOLD_DATA_FILE = 'EURUSD15.csv' # The candidates you want to test
        MIN_PROFITABLE_TESTS = 3        # How many test files must a strategy be profitable on?
        ```
    2. Run the script. This will take a long time as it performs hundreds or thousands of backtests.
        ```bash
        python 4_create_platinum_data.py
        ```
-   **Output:** A single CSV file will be created in the `platinum_data` folder (e.g., `platinum_EURUSD15.csv`). This is your final result. The strategy at the top of this file is the most robust, validated trading edge that the entire system was able to find.

---

## Interpreting the Final "Platinum" Output

The `platinum_data` file is the culmination of the project. Each row represents a fully backtested and validated strategy.

-   **`strategy_definition`**: The exact set of rules for the strategy.
-   **`Avg. Return [%]`**: The average percentage return across all test instruments.
-   **`Avg. Sharpe Ratio`**: The average risk-adjusted return. This is often the best metric for ranking strategies.
-   **`Avg. Win Rate [%]`**: The average win rate across all tests.
-   **`Avg. Max. Drawdown [%]`**: The average maximum loss from a peak. A lower number is better.
-   **`Avg. # Trades`**: The average number of trades the strategy took on each instrument.

The strategy at the top of this file, ranked by your chosen metric, is the "winner" of the discovery process and the one most worthy of further research, optimization, and potential live testing.

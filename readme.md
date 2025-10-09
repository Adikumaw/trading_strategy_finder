# Trading Strategy Finder

A modular, multi-layered pipeline for discovering high-probability trading strategies using exhaustive simulation, advanced feature engineering, and machine learning.

---

## 📚 Project Structure

```
project_root/
│
├── raw_data/             # Input: OHLCV market data
├── bronze_data/          # Output: Exhaustive trade combinations
├── silver_data/
│   ├── features/         # Output: Market features per candle
│   └── outcomes/         # Output: Trade outcomes with positioning
├── gold_data/
│   └── features/         # Output: ML-ready, normalized features
├── platinum_data/
│   ├── combinations/     # Output: Strategy blueprints
│   ├── discovered_strategy/ # Output: Actionable strategies
│   └── blacklists/       # Input: Failed strategies (feedback loop)
└── scripts/              # All processing scripts
```

---

## 🚀 Pipeline Overview

1. **Bronze Layer**: Generates all possible SL/TP trade combinations for each candle.
2. **Silver Layer**: Computes advanced market features and trade outcomes, decoupling context from results.
3. **Gold Layer**: Normalizes and transforms features for ML, producing a clean feature matrix.
4. **Platinum Layer**: Synthesizes features and outcomes, mines for profitable strategies using ML, and outputs actionable rules for backtesting.

---

## 🛠️ How to Use

1. **Prepare Raw Data**: Place OHLCV CSVs in `raw_data/`.
2. **Run Bronze Layer**: Execute `bronze_data_generator.py` to generate exhaustive trade combinations.
3. **Run Silver Layer**: Execute `silver_data_generator.py` to compute features and outcomes.
4. **Run Gold Layer**: Execute `gold_data_generator.py` to normalize and transform features.
5. **Run Platinum Layer**: Execute `platinum_combinations_generator.py` and `platinum_strategy_discoverer.py` to discover strategies.
6. **Backtest**: Use the strategies in `platinum_data/discovered_strategy/` for backtesting and evaluation.

---

## 🧩 Example Workflow

```bash
# Step 1: Bronze Layer
python scripts/bronze_data_generator.py

# Step 2: Silver Layer
python scripts/silver_data_generator.py

# Step 3: Gold Layer
python scripts/gold_data_generator.py

# Step 4: Platinum Layer
python scripts/platinum_combinations_generator.py
python scripts/platinum_strategy_discoverer.py
```

---

## 📈 Outputs

-   **bronze_data/**: All possible trade outcomes for each candle.
-   **silver_data/features/**: Market context per candle.
-   **silver_data/outcomes/**: Trade results with advanced positioning.
-   **gold_data/features/**: ML-ready, normalized features.
-   **platinum_data/combinations/**: Strategy blueprints.
-   **platinum_data/discovered_strategy/**: Actionable, testable trading strategies.

---

## 🧠 Advanced Features

-   Efficient chunked processing for large datasets.
-   Multiprocessing support for speed.
-   Modular design for easy debugging and extension.
-   Feedback loop for iterative strategy improvement.

---

## 📋 Requirements

-   Python 3.8+
-   pandas, numpy, scikit-learn, lightgbm, tqdm, numba, TA-Lib

---

## 🏁 Next Steps

-   Backtest discovered strategies.
-   Analyze performance metrics (Sharpe, drawdown, etc.).
-   Iterate and refine using feedback loop.

---

For details on each layer, see the individual layer README files in `scripts/`.

# 🕵️ The Strategy Finder Project

This repository contains a professional-grade, end-to-end quantitative trading research framework. It is a systematic, data-driven engine designed to discover, validate, and analyze algorithmic trading strategies from raw historical price data.

The system is built on a layered "funnel" architecture. It starts by simulating a massive universe of billions of potential trades (the **Bronze** layer) and progressively filters, enriches, and validates this data through successive layers, culminating in a handful of statistically robust, fully analyzed trading strategies ready for human review.

This project is not just a backtester; it is a complete **alpha discovery pipeline**.

## 🏛️ The Architectural Philosophy: A Data-Driven Funnel

The entire system is designed as a multi-stage data processing pipeline, where each layer has a distinct purpose and prepares the data for the next. This layered approach allows for maximum efficiency, scalability, and analytical rigor.

![image](https_user-images.githubusercontent.com/12345/67890.png) _(Conceptual: You would replace this with an actual diagram)_

1.  **🥉 Bronze Layer (The Possibility Engine):** Brute-force simulation of all conceivable winning trades to create an unbiased "universe of possibilities."
2.  **🥈 Silver Layer (The Enrichment Engine):** Enriches the raw trade data with deep market context, calculating over 200 features and powerful "relational positioning" metrics.
3.  **🥇 Gold Layer (The ML Preprocessor):** Translates the human-readable Silver data into a purely numerical, normalized, and standardized format optimized for machine learning.
4.  **💎 Platinum Layer (The Discovery Engine):** The intelligent heart of the system. Uses a machine learning model (`DecisionTreeRegressor`) to mine the vast dataset for statistically significant patterns and explicit, human-readable trading rules.
5.  **💍 Diamond & Zircon Layers (The Validation Gauntlet):** A final, two-stage backtesting phase.
    - **Diamond (Mastery):** Identifies strategies with an exceptional edge on their "home" instrument.
    - **Zircon (Validation):** Subjects the "master" strategies to a rigorous out-of-sample test across multiple other markets to verify robustness and generate final, detailed reports and trade logs.
6.  **📊 The Analyser (The Detective Dashboard):** A powerful, interactive Streamlit UI for the final step: human-led analysis. It provides a "post-mortem" view of each surviving strategy, allowing for deep interrogation of its performance, weaknesses, and ideal operating conditions.

## ✨ Key Features

- **End-to-End Automation:** A complete workflow from raw `.csv` files to a final, interactive analysis dashboard.
- **High Performance:** Optimized for speed using `numba` for JIT compilation of critical loops and `multiprocessing` for parallel execution across all available CPU cores.
- **Memory Safe:** Engineered to process enormous, multi-gigabyte datasets on standard hardware by using intelligent chunking and streaming for both input and output.
- **Intelligent Discovery:** Goes beyond simple indicator crossovers by using machine learning to discover complex, multi-condition rules and "Goldilocks zones" for indicator values.
- **Robust Validation:** Implements a strict Mastery-then-Validation workflow to aggressively filter out curve-fit or flimsy strategies.
- **Realistic Simulation:** The final backtesting engine accounts for key real-world costs, including spreads, commissions, and a simplified model of slippage.
- **Iterative Learning System:** A built-in **blacklist feedback loop** allows the system to learn from its failures. Strategies proven unprofitable by the backtester are automatically ignored in subsequent discovery runs, making the system smarter and more efficient over time.
- **Deep, Actionable Analysis:** The final dashboard is not just a report viewer but a true analytical tool, designed to help the user understand the _why_ behind a strategy's performance through comparative visualizations.

## 🚀 The Workflow: From Data to Insight

The entire process is a sequence of script executions. You must run the layers in order.

1.  **Setup:** Place your raw, headerless OHLC `.csv` files (e.g., `XAUUSD15.csv`) into the `/raw_data/` directory.
2.  **Run Bronze Layer:**
    ````bash
    python scripts/bronze_data_generator.py
    ```    This generates the initial dataset of all winning trade possibilities.
    ````
3.  **Run Silver Layer:**
    ```bash
    python scripts/silver_data_generator.py
    ```
    This enriches the Bronze data, creating the market features and chunked outcomes.
4.  **Run Gold Layer:**
    ```bash
    python scripts/gold_data_generator.py
    ```
    This preprocesses the market features for machine learning.
5.  **Run Platinum Layer (3 Stages):**
    ```bash
    python scripts/platinum_combinations_generator.py
    python scripts/platinum_target_extractor.py
    python scripts/platinum_strategy_discoverer.py
    ```
    This is the core discovery phase that finds the candidate strategy rules.
6.  **Run Validation Gauntlet (3 Stages):**
    ```bash
    python scripts/diamond_data_prepper.py  # Run once for all markets
    python scripts/diamond_backtester.py     # Select a discovered strategy file
    python scripts/zircon_validator.py       # Select the resulting master strategy file
    ```
    This is the final backtesting and reporting phase.
7.  **Launch the Analysis Dashboard:**
    ```bash
    streamlit run backtest_analyser/app.py
    ```
    This launches the interactive UI in your web browser for the final analysis.

## 📁 Full Project Folder Structure

```
.
├── backtest_analyser/
│   └── app.py                 # The Streamlit UI dashboard
├── bronze_data/
├── diamond_data/
│   ├── prepared_data/
│   ├── backtesting_results/
│   └── trade_logs/
├── gold_data/
│   └── features/
├── platinum_data/
│   ├── blacklists/
│   ├── combinations/
│   ├── discovered_strategy/
│   └── targets/
├── raw_data/
├── silver_data/
│   ├── chunked_outcomes/
│   └── features/
├── zircon_data/
│   ├── input/
│   ├── results/
│   └── trade_logs/
└── scripts/
    ├── bronze_data_generator.py
    ├── silver_data_generator.py
    ├── gold_data_generator.py
    ├── platinum_combinations_generator.py
    ├── platinum_target_extractor.py
    ├── platinum_strategy_discoverer.py
    ├── diamond_data_prepper.py
    ├── diamond_backtester.py
    └── zircon_validator.py
```

## 🛠️ Future Improvements & Potential Enhancements

This framework is a powerful foundation that can be extended in many ways:

1.  **Walk-Forward Optimization:** Implement a walk-forward analysis module in the validation layers to provide a more rigorous, out-of-sample test that simulates how a strategy would adapt to changing market conditions over time.
2.  **Portfolio-Level Backtesting:** Create a final "Portfolio" layer to simulate the performance of multiple, uncorrelated strategies running concurrently, analyzing combined equity curves and portfolio-level drawdown.
3.  **Monte Carlo Simulation:** Add a module to stress-test final strategies by running Monte Carlo simulations on their trade logs. Shuffling the order of trades thousands of times helps determine if the historical performance was a result of a lucky sequence of wins.
4.  **Alternative ML Models:** While the `DecisionTreeRegressor` is excellent for interpretability, the Platinum Layer could be adapted to use other models like `RandomForestRegressor` (for feature importance analysis) or dedicated rule-learning libraries like `skope-rules`.
5.  **Dynamic Parameter Optimization:** The system currently uses fixed parameters for indicators and models. A meta-layer could be built to systematically experiment with different settings (e.g., different `max_depth` for the Decision Tree) to find the optimal configuration for discovering robust strategies.

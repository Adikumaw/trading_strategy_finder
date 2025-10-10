# 💎 Diamond Layer: The Multi-Market Backtesting Engine

The Diamond Layer is the final and most important stage of the entire strategy discovery pipeline. It acts as the ultimate judge, taking the statistically significant _patterns_ discovered by the Platinum Layer and subjecting them to a rigorous, multi-market backtest to determine if they are truly profitable and robust **trading strategies**.

Its core purpose is to simulate the real-world performance of a strategy, providing the definitive metrics needed to make a final go/no-go decision.

---

## 🏛️ Architectural Philosophy: The "Clean Room" Validation

This backtester is built on a critical architectural principle: **it trusts nothing**. To prevent any form of data leakage or lookahead bias, it does **not** use any pre-processed Silver or Gold data files.

Instead, for each market it tests, it performs a "clean room" simulation:

1.  It starts with **only** the raw OHLCV price data.
2.  It re-creates the entire Silver and Gold data pipelines **in-memory** from scratch.
3.  Only then does it apply the strategy rules and simulate trades.

This perfectly mimics how a live trading algorithm would work, ensuring the backtest results are as realistic and reliable as possible.

---

## ⚙️ How It Works: The Backtesting Process

The script follows a highly optimized, multi-stage workflow:

1.  **Interactive Setup:** The script begins by asking the user two questions:

    -   Which `discovered_strategy` file do you want to test?
    -   Which markets from your `raw_data` folder do you want to validate these strategies on? (You can select specific ones or `all`).

2.  **"Process Once, Test Many":** To maximize speed, the script prepares all necessary data upfront.

    -   It loops through each selected market _once_.
    -   For each market, it calculates all Silver and Gold features and stores them in a `market_data_cache` in RAM.
    -   This prevents redundant feature calculation and is a massive performance optimization.

3.  **The Main Simulation Loop:** The script then iterates through each strategy rule from the selected Platinum file.

    -   For each strategy, it loops through every market in the `market_data_cache`.
    -   **Find Entries:** It uses the strategy's `market_rule` to query the in-memory Gold data and identify all valid trade entry points.
    -   **Simulate Trades:** It runs the core simulation engine:
        -   Calculates the precise SL/TP price levels using the strategy definition and the in-memory Silver data.
        -   Applies a **Fixed Fractional Risk** model to calculate position size (e.g., risk 2% of capital per trade).
        -   Steps candle-by-candle through the price history to determine the trade's outcome (Win/Loss) and P&L.
        -   Keeps a running log of the account equity to calculate performance metrics.
    -   **Analyze Regimes:** It records the distribution of market conditions (`session`, `trend_regime`, `vol_regime`) for all trades taken by the strategy in that market.

4.  **Generate Reports & Feedback Loop:** After all simulations are complete:
    -   It compiles a **Detailed Report** with the performance of every strategy on every market tested.
    -   It creates a high-level **Summary Report**, grouping by strategy to show its average performance and robustness across all markets.
    -   It automatically identifies strategies that failed to meet performance thresholds (e.g., Profit Factor < 1.2) and adds them to a **Blacklist file**, completing the feedback loop for the Platinum Layer.

---

## 📁 Diamond Folder Structure

This layer produces the final, actionable reports of the entire project.

```
project_root/
│
├── platinum_data/
│   ├── discovered_strategy/ # INPUT: The strategies to be tested
│   └── blacklists/          # OUTPUT: Failed strategies are sent back here
│
├── diamond_data/
│   └── backtesting_results/ # OUTPUT: The final performance reports
│       ├── summary_report_AUDUSD1.csv
│       └── detailed_report_AUDUSD1.csv
│
└── scripts/
    └── diamond_backtester.py
```

---

## 🧱 Output File Descriptions

### 1. `diamond_data/backtesting_results/summary_report_{instrument}.csv`

This is the high-level overview. Each row represents one strategy, aggregated across all tested markets. It's the best place to start your analysis.

| Column                        | Description                                                                             | Example      |
| :---------------------------- | :-------------------------------------------------------------------------------------- | :----------- |
| `strategy_id`                 | A unique ID for this specific strategy rule.                                            | `a1b2c3d4e5` |
| `market_rule`, `sl_def`, etc. | The full definition of the strategy.                                                    |              |
| **`markets_passed`**          | **(CRITICAL)** How many markets the strategy was profitable on out of the total tested. | `4/5`        |
| **`avg_profit_factor`**       | **(CRITICAL)** The average Profit Factor across all tested markets.                     | `1.78`       |
| `avg_max_drawdown_pct`        | The average Maximum Drawdown across all tested markets.                                 | `12.5`       |
| `total_trades`                | The total number of trades taken across all markets.                                    | `1253`       |

### 2. `diamond_data/backtesting_results/detailed_report_{instrument}.csv`

This file provides the granular, per-market breakdown for deep-dive analysis.

| Column                                      | Description                                                                                                                               |
| :------------------------------------------ | :---------------------------------------------------------------------------------------------------------------------------------------- |
| `strategy_id`                               | The unique ID, used to group results for a single strategy.                                                                               |
| `market`                                    | The specific market this row's results are for (e.g., `EURUSD1.csv`).                                                                     |
| `profit_factor`, `max_drawdown_pct`, etc.   | The detailed performance metrics for this strategy _on this market_.                                                                      |
| **`session_pct`, `trend_regime_pct`, etc.** | **(The "Why")** A dictionary showing the percentage of trades that occurred in each market regime (e.g., 75% in 'trend', 20% in 'range'). |

---

## 📈 Next Steps: The Master Analyser

The two output reports from this Diamond Layer are the final inputs for your visual "Master Analyser."

-   Use the **Summary Report** to filter for robust strategies (e.g., `markets_passed == '5/5'` and `avg_profit_factor > 1.5`).
-   For each robust strategy, use its `strategy_id` to filter the **Detailed Report**.
-   From the detailed data, you can now plot:
    -   An equity curve for each market to visually compare performance.
    -   Bar charts of the regime analysis (`session_pct`, etc.) to understand _why_ the strategy performs differently across various markets.

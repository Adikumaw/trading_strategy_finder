# 💎 Diamond Layer: The Multi-Market Backtesting Engine

The Diamond Layer is the final and most important stage of the entire strategy discovery pipeline. It acts as the ultimate judge, taking the statistically significant _patterns_ discovered by the Platinum Layer and subjecting them to a rigorous, multi-market backtest to determine if they are truly profitable and robust **trading strategies**.

Its core purpose is to simulate the real-world performance of a strategy with proper risk management, providing the definitive metrics needed to make a final go/no-go decision.

---

## 🏛️ Architectural Overview: A Two-Stage Validation Process

The Diamond Layer is split into two distinct scripts to maximize efficiency and speed. This "prepare once, test many" approach avoids redundant, time-consuming calculations.

1.  **`diamond_data_prepper.py` (The Data Minter):**

    -   **Job:** A one-time, pre-processing script. It takes the raw market data and runs it through the entire `raw -> silver -> gold` feature generation pipeline.
    -   **Output:** It saves the final, ready-to-use Silver and Gold DataFrames in a highly efficient `.parquet` format to the `diamond_data/prepared_data/` directory.
    -   **Why:** This is a massive performance optimization. It performs the hours-long feature calculation **once**, so that the main backtester can run in minutes.

2.  **`diamond_backtester.py` (The Simulation Engine):**
    -   **Job:** This is the high-speed, parallelized simulation engine. It is now incredibly lightweight and fast.
    -   **Workflow:** It loads the pre-prepared data from the `prepared_data` cache, then uses a `multiprocessing.Pool` to test hundreds of strategies in parallel across all available CPU cores.
    -   **Why:** By separating data prep from simulation, this script can focus entirely on its core task: running thousands of backtests as quickly as possible.

---

## ⚙️ How It Works: The Backtesting Workflow

1.  **Run `diamond_data_prepper.py`:** You run this script once to prepare all the markets you intend to backtest on.
2.  **Run `diamond_backtester.py`:**
    -   **Interactive Setup:** The script asks you which `discovered_strategy` file you want to test and which of the _prepared markets_ you want to validate against.
    -   **Load Prepared Data:** It loads all the necessary Silver and Gold DataFrames from the `prepared_data` cache into RAM.
    -   **Parallel Simulation:** It distributes the list of strategies to a pool of worker processes. Each worker independently backtests its assigned strategies against all the in-memory market data.
    -   **Core Simulation Engine:** Each backtest performs:
        -   **Entry Identification:** Finds all valid trade entries using the strategy's `market_rule`.
        -   **Risk Management:** Applies a **Fixed Fractional Risk** model to calculate position size (e.g., risk 2% of capital per trade).
        -   **Trade Simulation:** Steps candle-by-candle to determine the outcome (Win/Loss) and P&L of every trade.
        -   **Performance & Regime Analysis:** Calculates final metrics (Profit Factor, Max Drawdown) and analyzes the market conditions (`session`, `trend_regime`) for the trades.
    -   **Generate Reports & Feedback Loop:** After all simulations are complete, it compiles the final **Summary** and **Detailed Reports** and automatically generates a **Blacklist** of underperforming strategies to feed back to the Platinum Layer.

---

## 📁 Diamond Folder Structure

This layer introduces a new directory for the pre-prepared data cache.

```
project_root/
│
├── platinum_data/
│   ├── discovered_strategy/ # INPUT: The strategies to be tested
│   └── blacklists/          # OUTPUT: Failed strategies are sent back here
│
├── diamond_data/
│   ├── prepared_data/       # INTERMEDIATE: The pre-calculated market data cache
│   └── backtesting_results/ # FINAL OUTPUT: The performance reports
│
└── scripts/
    ├── diamond_data_prepper.py    # Script 1
    └── diamond_backtester.py      # Script 2
```

---

## 🧱 Output File Descriptions

### 1. `diamond_data/backtesting_results/summary_report_{instrument}.csv`

The high-level overview. Each row is one strategy, with its performance aggregated across all tested markets.

| Column                        | Description                                                                                | Example      |
| :---------------------------- | :----------------------------------------------------------------------------------------- | :----------- |
| `strategy_id`                 | A unique ID for this specific strategy rule.                                               | `a1b2c3d4e5` |
| `market_rule`, `sl_def`, etc. | The full definition of the strategy.                                                       |              |
| **`markets_passed`**          | **(CRITICAL)** How many markets the strategy was profitable on.                            | `4/5`        |
| **`avg_profit_factor`**       | **(CRITICAL)** The average Profit Factor across all tested markets. A value > 1.2 is good. | `1.78`       |
| `avg_max_drawdown_pct`        | The average Maximum Drawdown. A lower value is better.                                     | `12.5`       |
| `total_trades`                | The total number of trades taken across all markets.                                       | `1253`       |

### 2. `diamond_data/backtesting_results/detailed_report_{instrument}.csv`

The granular, per-market breakdown for deep-dive analysis.

| Column                                      | Description                                                                                        |
| :------------------------------------------ | :------------------------------------------------------------------------------------------------- |
| `strategy_id`                               | The unique ID, used to group results for a single strategy.                                        |
| `market`                                    | The specific market this row's results are for (e.g., `EURUSD1.csv`).                              |
| `profit_factor`, `max_drawdown_pct`, etc.   | The detailed performance metrics for this strategy _on this market_.                               |
| **`session_pct`, `trend_regime_pct`, etc.** | **(The "Why")** A dictionary showing the percentage of trades that occurred in each market regime. |

---

## 📈 Next Steps: The Master Analyser

The two output reports are the final inputs for your visual "Master Analyser."

1.  Use the **Summary Report** to filter for robust strategies (e.g., `markets_passed == '5/5'` and `avg_profit_factor > 1.5`).
2.  For each robust strategy, use its `strategy_id` to filter the **Detailed Report**.
3.  From the detailed data, you can now plot an equity curve for each market and create charts of the regime analysis to understand _why_ a strategy performs differently in various market conditions.

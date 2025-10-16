# 💎 Diamond & Zircon Layers: The Validation Gauntlet

The Diamond and Zircon layers represent the **final and most critical phase** of the entire strategy discovery pipeline. Together, they form a two-stage validation gauntlet designed to rigorously test the statistical patterns from the Platinum Layer and determine if they are truly robust, profitable trading strategies.

This is where theory meets a simulated reality. The purpose of these layers is to provide the definitive, evidence-backed metrics needed to make a final "go/no-go" decision on any discovered strategy.

## What It Is

This phase is composed of two distinct but interconnected scripts that separate the process of **Mastery** from **Validation**.

1.  **`diamond_backtester.py` (The Mastery Engine):** This script is a lean, fast, single-market backtester. It takes the strategies discovered for a specific instrument (e.g., `XAUUSD15`) and backtests them _only_ on that same instrument's historical data. Its job is to filter for strategies that demonstrate exceptional performance or "mastery" on their home turf.

2.  **`zircon_validator.py` (The Ultimate Judge):** This script is the final, multi-market reporting engine. It takes the small, pre-filtered list of "master" strategies from the Diamond Layer and subjects them to a rigorous out-of-sample test. It automatically finds all other available markets of the same timeframe and runs a full backtest against them. Crucially, this is the stage that generates the detailed **trade logs** for the elite strategies that survive, providing the deepest level of data for final analysis.

## Why It Is

A strategy that looks perfect on one dataset is often a fluke—a result of overfitting. A truly robust strategy must demonstrate a consistent edge across different market conditions, which we simulate by using different instruments. This two-stage approach is critical for several reasons:

1.  **To Isolate "Alpha":** The Diamond Layer first allows us to find highly specialized, "sniper" strategies that are exceptionally good at trading a single instrument. This is the "alpha" or the core edge.
2.  **To Test for Robustness:** The Zircon Layer then asks the most important question: "Is this alpha real, or was it just luck?" By testing the master strategy on other markets, we verify if the edge is genuine and transferable, or if it was simply curve-fit to the original dataset.
3.  **Efficiency and Focus:** This workflow is highly efficient. The computationally intensive task of backtesting thousands of initial candidates is done quickly in the Diamond Layer without the overhead of saving detailed logs. The resource-intensive process of generating detailed trade logs is reserved for only the handful of elite strategies that make it to the Zircon Layer.
4.  **To Create a Definitive Feedback Loop:** The Diamond Layer is the final arbiter for the `blacklist`. By testing on the origin market, it provides a clean verdict on whether a blueprint is fundamentally flawed, which then intelligently prunes the search space for future Platinum Layer runs.

## How It Works

The workflow is a clear, sequential process of filtering and deep validation.

#### **Stage 1: Run `diamond_data_prepper.py`**

- This is a one-time utility that must be run first. It takes all the raw market data, runs it through the full `Silver -> Gold` feature generation pipeline, and saves the final, analysis-ready `_silver.parquet` and `_gold.parquet` files to the `diamond_data/prepared_data/` directory. This "prepare once, test many" approach makes the backtesting stages incredibly fast.

#### **Stage 2: Run `diamond_backtester.py` (Mastery Engine)**

- **Job:** You select a `discovered_strategy` file to test (e.g., `XAUUSD15.csv`).
- **Logic:**
  1.  The script automatically identifies the origin market (`XAUUSD15`) from the filename.
  2.  It loads the corresponding prepared data from the cache.
  3.  Using multiprocessing, it backtests all candidate strategies against this **single market**.
  4.  The simulation uses a realistic **Fixed Fractional Risk** model and accounts for **spreads, commissions, and slippage**.
  5.  It calculates a full suite of performance metrics, including Profit Factor, Sharpe Ratio, and Max Drawdown.
- **Output:**
  1.  `zircon_data/input/master_strategies_...csv`: A filtered list of only the strategies that passed the strict "mastery" performance thresholds.
  2.  `diamond_data/backtesting_results/diamond_report_...csv`: A detailed report of the performance of _all_ tested strategies on the origin market.
  3.  `platinum_data/blacklists/...csv`: An updated blacklist of the parent blueprints of all failed strategies.

#### **Stage 3: Run `zircon_validator.py` (The Ultimate Judge)**

- **Job:** You select a `master_strategies_...csv` file from the `zircon_data/input/` directory.
- **Logic:**
  1.  The script identifies the origin timeframe (e.g., `15m`) from the filename.
  2.  It finds all prepared markets of the **same timeframe**, including the origin market.
  3.  It then iteratively backtests the small list of master strategies against this full set of markets.
  4.  For each strategy and market, it performs the same high-fidelity simulation.
- **Output (The Final Product for the UI):**
  1.  **`zircon_data/trade_logs/{strategy_id}/{market}.csv`**: Detailed, trade-by-trade logs for every elite strategy on every market it was tested on.
  2.  **`zircon_data/results/summary_report_...csv`**: A high-level summary, aggregating the performance of each strategy across all tested markets.
  3.  **`zircon_data/results/detailed_report_...csv`**: A granular, per-market breakdown of performance and regime analysis for each strategy.

## 📁 Folder Structure

```
project_root/
├── platinum_data/
│   ├── discovered_strategy/   # INPUT: Strategies to test
│   └── blacklists/            # OUTPUT: Failed blueprints
│
├── diamond_data/
│   ├── prepared_data/         # CACHE: Pre-calculated market data
│   └── backtesting_results/   # OUTPUT: Detailed mastery reports
│
├── zircon_data/
│   ├── input/                 # INPUT: Master strategies from Diamond
│   ├── results/               # OUTPUT: Final summary & detailed reports for UI
│   └── trade_logs/            # OUTPUT: Granular trade logs for UI
│
└── scripts/
    ├── diamond_data_prepper.py
    ├── diamond_backtester.py
    └── zircon_validator.py
```

## 📈 Input & Output

### Input Files

1.  **`platinum_data/discovered_strategy/{instrument}.csv`**: The initial list of candidate strategies.
2.  **`diamond_data/prepared_data/`**: The cache of all market data.

### Final Output Files

1.  **`zircon_data/results/summary...` & `detailed...`**: The definitive reports used by the `app.py` analyzer. The summary aggregates performance, while the detailed report provides a per-market breakdown and regime analysis.
2.  **`zircon_data/trade_logs/...`**: The trade-by-trade histories that power the most advanced visualizations in the analyzer, allowing for performance to be viewed in the context of the market's macro trend.

## 🚀 Possible Enhancements & Future Improvements

1.  **Walk-Forward Optimization:** The current backtest is a static, in-sample test. A more advanced validation method is walk-forward analysis, where the system would train on one period (e.g., 2018-2020), test on the next (2021), then retrain on 2019-2021 and test on 2022, and so on. This provides a more realistic simulation of how a strategy would adapt to changing market conditions.
2.  **Portfolio-Level Backtesting:** The current engine tests each strategy in isolation. A final "Portfolio" layer could be built to take the best strategies discovered across different instruments and timeframes and simulate them together as a diversified portfolio, analyzing overall portfolio drawdown and correlated returns.
3.  **Monte Carlo Simulation:** To test a strategy's sensitivity to luck, a Monte Carlo module could be added. It would take the trade log from the Zircon layer and randomly shuffle the order of trades thousands of times to generate a statistical distribution of possible equity curves and maximum drawdowns, revealing if the historical result was an outlier.
4.  **More Complex Risk Management:** The Fixed Fractional model is robust. However, more advanced models could be implemented, such as risk models that adjust position size based on market volatility (e.g., risking less in high-volatility periods).

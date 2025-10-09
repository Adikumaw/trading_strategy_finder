# 💎 Platinum Layer: The Strategy Discovery Engine

The Platinum Layer is the culmination of the entire data pipeline. It acts as an intelligent, automated analyst that sifts through millions of trade possibilities to discover concrete, high-probability trading strategies.

Its core purpose is to synthesize the market context from the **Gold Layer** with the trade results from the **Silver Layer**, using Machine Learning to find profitable, rule-based strategies.

---

## 🏛️ Architectural Overview

The Platinum Layer operates in a two-stage process:

1.  **`platinum_combinations_generator.py` (The Blueprint Creator):** This script is now a sophisticated scanner that generates advanced strategy "blueprints." It no longer just checks if a SL/TP is _at_ a level, but rather **bins the placement into 10% increments**. It creates blueprints like: _"SL placed 70-80% of the way to resistance, TP is a 0.5% ratio."_ This vastly expands the search space for nuanced strategies.

2.  **`platinum_strategy_discoverer.py` (The Rule Miner):** This is the main engine. It takes each binned blueprint and uses a **Decision Tree Regressor** to find specific market conditions under which that blueprint has a high _predicted win rate_. It is designed to be run repeatedly, incorporating new data and feedback from backtesting.

---

## ⚙️ How It Works: The Discovery Process

1.  **Load Binned Blueprints:** It loads the master list of granular strategy definitions (e.g., `sl_def=resistance`, `sl_bin=7`).
2.  **State Management:** It checks for already discovered strategies and blacklists, ensuring it only processes **new** or **blacklisted** combinations.
3.  **Iterate and Filter:** For each blueprint, it performs a memory-efficient scan of the `silver_data/outcomes` file to find all trades matching that exact structure (e.g., all trades where the SL was placed between 70% and 80% of the way to resistance).
4.  **Aggregate to Prevent Bias:** To solve the "multiple votes" problem, it groups the matching trades by their `entry_time` and calculates the **average win rate for each candle**. This ensures every market event gets one fair vote.
5.  **Create Training Data:** It fetches the ML-ready market features from the `gold_data/features` file for each of these unique candles.
6.  **Machine Learning Rule Discovery:**
    -   It trains a **Decision Tree Regressor** on this aggregated data. The model's goal is to predict the **win rate** of the strategy blueprint based on the market conditions.
    -   The Decision Tree is used because it produces simple, human-readable **IF-THEN rules**.
7.  **Extract & Save Strategies:** The script extracts all rules from the tree that predict a win rate above a set threshold (e.g., >60%). Each rule is saved as a complete, actionable strategy.

---

## 📁 Platinum Folder Structure

This layer reads from the Silver and Gold layers and produces a new set of highly valuable data artifacts.

```
project_root/
│
├── silver_data/
│   └── outcomes/         # INPUT: Enriched trade outcomes with positioning data
│
├── gold_data/
│   └── features/         # INPUT: ML-ready, normalized market features
│
├── platinum_data/
│   │
│   ├── combinations/     # OUTPUT 1: Master list of all unique strategy blueprints
│   │   └── AUDUSD1.csv
│   │
│   ├── discovered_strategy/ # OUTPUT 2: The final, actionable trading strategies
│   │   └── AUDUSD1.csv
│   │
│   └── blacklists/       # INPUT (Feedback Loop): Strategies that failed backtesting
│       └── AUDUSD1.csv
│
└── scripts/
    ├── platinum_combinations_generator.py
    └── platinum_strategy_discoverer.py
```

---

## 🔄 State Management: Blacklists & The Feedback Loop

The Platinum Layer is designed to learn and adapt over time.

-   **Skipping:** The discoverer will not re-analyze a strategy blueprint if valid rules for it already exist in the `discovered_strategy` file.
-   **Blacklisting:** The **Backtester** (the next layer) is responsible for validating these discovered strategies. If a strategy looks good on paper but fails a rigorous backtest (e.g., due to poor risk-reward over time), the backtester will add its definition to a `blacklist` file.
-   **Re-Discovery:** On its next run, the `platinum_strategy_discoverer` will see this blacklisted blueprint and **re-process it**, attempting to find a _different, better_ market rule for that same strategy structure. This creates a powerful feedback loop for continuous improvement.

---

## 🧱 Output File Descriptions

### 1. `platinum_data/combinations/{instrument}.csv`

This file now contains highly granular blueprints for testing.

| Column   | Description                                                                                                    | Example                  |
| :------- | :------------------------------------------------------------------------------------------------------------- | :----------------------- |
| `type`   | The type of binned strategy.                                                                                   | `Semi-Dynamic-SL-Binned` |
| `sl_def` | The SL definition (level or ratio).                                                                            | `resistance`             |
| `sl_bin` | **(NEW)** The 10% placement bin. A value of `7` means 70-80%. A value of `-1` means -10% to 0% (behind entry). | `7`                      |
| `tp_def` | The TP definition (level or ratio).                                                                            | `0.005`                  |
| `tp_bin` | **(NEW)** The 10% placement bin for the TP.                                                                    | `NaN`                    |

### 2. `platinum_data/discovered_strategy/{instrument}.csv`

This is the primary output. Each row is a complete, testable trading strategy with more context.

| Column                           | Description                                                         | Example                                                |
| :------------------------------- | :------------------------------------------------------------------ | :----------------------------------------------------- |
| `type`, `sl_def`, `sl_bin`, etc. | The complete strategy blueprint definition.                         |                                                        |
| `market_rule`                    | **The discovered ML rule.** The market conditions for entry.        | `` `RSI_14` > 1.25 and `vol_regime_low_vol` == True `` |
| `win_prob`                       | The **predicted win rate** for candles matching this rule.          | `0.7152`                                               |
| `num_candles`                    | The number of unique historical candles that fit this rule.         | `88`                                                   |
| `total_trades`                   | The total number of trade simulations represented by those candles. | `451`                                                  |

---

## 🧮 Example Workflow

1. Load `gold_data/features` and `silver_data/outcomes`.
2. Generate strategy blueprints (`combinations`).
3. Discover actionable strategies using ML (`discovered_strategy`).
4. Output is ready for backtesting and real-world evaluation.

---

## 📈 Next Steps

The `discovered_strategy` files are the direct input for the final **Backtesting Layer**. The backtester will simulate these strategies over the entire dataset to evaluate their long-term performance, including metrics like Sharpe ratio, max drawdown, and overall profitability, ultimately determining their real-world viability.

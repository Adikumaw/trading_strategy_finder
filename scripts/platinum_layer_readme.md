# 💎 Platinum Layer: The Strategy Discovery Engine

The Platinum Layer is the culmination of the entire data pipeline. It acts as an intelligent, automated analyst that sifts through millions of trade possibilities to discover concrete, high-probability trading strategies.

Its core purpose is to synthesize the market context from the **Gold Layer** with the trade results from the **Silver Layer**, using Machine Learning to find profitable, rule-based strategies.

---

## 🏛️ Architectural Overview

The Platinum Layer operates in a two-stage process, using two distinct scripts for maximum efficiency and modularity:

1.  **`platinum_combinations_generator.py` (The Blueprint Creator):** This script acts as a preliminary scanner. It reads the massive `silver_data/outcomes` file once and extracts every unique _strategy definition_ (e.g., "SL at SMA_50, TP at 0.5% ratio"). It saves this master list of testable "blueprints," preventing redundant work later.

2.  **`platinum_strategy_discoverer.py` (The Rule Miner):** This is the main engine. It takes each blueprint from the combinations file and uses a Decision Tree model to find specific market conditions under which that blueprint is profitable. It's designed to be run repeatedly, incorporating new data and feedback from backtesting.

---

## ⚙️ How It Works: The Discovery Process

The `platinum_strategy_discoverer.py` script follows a sophisticated workflow for each instrument:

1.  **Load Blueprints:** It loads the master list of strategy definitions from `platinum_data/combinations/`.
2.  **State Management:** It checks for already discovered strategies (`discovered_strategy/`) and any blacklisted strategies (`blacklists/`). It intelligently decides to process only **new** or **blacklisted** combinations, saving immense amounts of time on subsequent runs.
3.  **Iterate and Filter:** For each blueprint, it performs a memory-efficient scan of the large `silver_data/outcomes` file (in chunks) to find all historical trades that perfectly match that strategy's structure (e.g., all trades where the SL was placed within 10 basis points of the 20-period SMA).
4.  **Create Training Data:** It fetches the corresponding ML-ready market features from the `gold_data/features` file for each of these trades, creating a dedicated training dataset for this _single_ blueprint.
5.  **Machine Learning Rule Discovery:**
    -   It trains a **Decision Tree Classifier** on this dataset. The goal is to teach the model to distinguish between winning and losing trades _for this specific strategy structure_.
    -   The Decision Tree is used because it produces simple, human-readable **IF-THEN rules**.
6.  **Extract & Save Strategies:** The script extracts all high-probability rules from the trained tree (e.g., `IF 'RSI_14' > 1.2 AND 'session_London' == True THEN...`). Each rule, along with its performance metrics (win probability, number of trades), is saved as a complete, actionable strategy to `platinum_data/discovered_strategy/`.

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

A simple but crucial blueprint file. It lists every unique way a trade's SL and TP can be defined.

| Column   | Description                                      | Example              |
| :------- | :----------------------------------------------- | :------------------- |
| `type`   | The type of strategy structure.                  | `Semi-Dynamic-SL`    |
| `sl_def` | The Stop-Loss definition (a level or a ratio).   | `SMA_20`             |
| `tp_def` | The Take-Profit definition (a level or a ratio). | `0.005` (i.e., 0.5%) |

### 2. `platinum_data/discovered_strategy/{instrument}.csv`

This is the primary, high-value output of the entire pipeline up to this point. Each row is a complete, testable trading strategy.

| Column        | Description                                                           | Example                                                |
| :------------ | :-------------------------------------------------------------------- | :----------------------------------------------------- |
| `type`        | The strategy structure type.                                          | `Semi-Dynamic-SL`                                      |
| `sl_def`      | The Stop-Loss definition.                                             | `SMA_20`                                               |
| `tp_def`      | The Take-Profit definition.                                           | `0.005`                                                |
| `market_rule` | **The discovered ML rule.** The market conditions required for entry. | `` `RSI_14` > 1.25 and `vol_regime_low_vol` == True `` |
| `win_prob`    | The historical win probability of trades matching this exact rule.    | `0.7152` (i.e., 71.52%)                                |
| `num_trades`  | The number of historical trades that fit this rule.                   | `152`                                                  |

---

## 📈 Next Steps

The `discovered_strategy` files are the direct input for the final **Backtesting Layer**. The backtester will simulate these strategies over the entire dataset to evaluate their long-term performance, including metrics like Sharpe ratio, max drawdown, and overall profitability, ultimately determining their real-world viability.

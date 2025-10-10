# 💎 Platinum Layer: The Strategy Discovery Engine

The Platinum Layer is the culmination of the entire data pipeline. It acts as an intelligent, automated analyst that sifts through millions of winning trades to find **robust and statistically significant patterns**.

Its core purpose has evolved. Since the input data only contains wins, this layer no longer tries to predict profitability. Instead, it uses Machine Learning to solve a more important problem: **filtering out noise**. It finds strategy blueprints and market conditions that have historically produced a **high density of successful trades**, indicating a strong, repeatable pattern worthy of backtesting.

---

## 🏛️ Architectural Overview

The Platinum Layer operates in a two-stage process:

1.  **`platinum_combinations_generator.py` (The Blueprint Creator):** This sophisticated scanner generates advanced strategy "blueprints." It bins SL/TP placements into 10% increments (including negative/directional placements), creating blueprints like: _"SL placed 70-80% of the way to resistance, TP is a 0.5% ratio."_

2.  **`platinum_strategy_discoverer.py` (The Pattern Significance Engine):** This is the main engine. It takes each blueprint and uses a **Decision Tree Regressor** to find specific market conditions that historically led to a **high frequency of successful trades** (a high "trade density").

---

## ⚙️ How It Works: The Discovery Process

1.  **Load Binned Blueprints:** It loads the master list of granular strategy definitions (e.g., `sl_def=resistance`, `sl_bin=7`).
2.  **State Management:** It checks for already discovered strategies and blacklists, ensuring it only processes **new** or **blacklisted** combinations.
3.  **Iterate and Filter:** For each blueprint, it performs a memory-efficient scan of the `silver_data/outcomes` file to find all winning trades that match that exact structure.
4.  **Aggregate for Significance:** It groups the matching trades by their `entry_time` and calculates the **total number of trades for each candle**. This is the core metric.
5.  **Create Training Data:** It fetches the ML-ready market features from the `gold_data/features` file for each of these unique candles.
6.  **Machine Learning Rule Discovery:**
    -   It trains a **Decision Tree Regressor**. The model's goal is no longer to predict a win rate, but to predict the **trade count** (the "density") for the strategy blueprint based on market conditions.
    -   It learns to identify "hotspots"—market conditions where this blueprint was historically a very common and successful pattern.
7.  **Extract & Save Strategies:** The script extracts rules from the tree that predict a **high trade density** above a set threshold. Each rule is saved as a complete, statistically significant strategy ready for the final validation step.

---

## 📁 Platinum Folder Structure

_(Folder structure is unchanged.)_

---

## 🔄 State Management: Blacklists & The Feedback Loop

*(This logic is unchanged, but its purpose is now clearer: it allows the engine to re-evaluate blacklisted blueprints to find more *robust* patterns, not just more *profitable* ones.)*

---

## 🧱 Output File Descriptions

### 1. `platinum_data/combinations/{instrument}.csv`

This file contains the highly granular blueprints for testing.

| Column   | Description                                                                                                    | Example                  |
| :------- | :------------------------------------------------------------------------------------------------------------- | :----------------------- |
| `type`   | The type of binned strategy.                                                                                   | `Semi-Dynamic-SL-Binned` |
| `sl_def` | The SL definition (level or ratio).                                                                            | `resistance`             |
| `sl_bin` | **(NEW)** The 10% placement bin. A value of `7` means 70-80%. A value of `-1` means -10% to 0% (behind entry). | `7`                      |
| `tp_def` | The TP definition (level or ratio).                                                                            | `0.005`                  |
| `tp_bin` | **(NEW)** The 10% placement bin for the TP.                                                                    | `NaN`                    |

### 2. `platinum_data/discovered_strategy/{instrument}.csv`

This is the primary output. Each row is a **statistically significant pattern** that is a candidate for being a real strategy.

| Column                           | Description                                                                                                                                                             | Example                                                |
| :------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------- |
| `type`, `sl_def`, `sl_bin`, etc. | The complete strategy blueprint definition.                                                                                                                             |                                                        |
| `market_rule`                    | **The discovered ML rule.** The market conditions for the pattern.                                                                                                      | `` `RSI_14` > 1.25 and `vol_regime_low_vol` == True `` |
| `avg_trade_density`              | **(NEW)** The average number of successful trades generated per candle when this rule is met. A higher number indicates a more robust and frequently occurring pattern. | `8.54`                                                 |
| `num_candles`                    | The number of unique historical candles that fit this rule.                                                                                                             | `88`                                                   |
| `total_trades`                   | The total number of successful trade simulations represented by those candles.                                                                                          | `751`                                                  |

---

## 📈 Next Steps

The `discovered_strategy` files are the direct input for the final **Backtesting Layer**. The backtester's job is now even more critical: it will take these statistically significant _patterns_ and simulate them to determine if they are actually _profitable_, evaluating their performance with metrics like Sharpe ratio, max drawdown, and overall return.

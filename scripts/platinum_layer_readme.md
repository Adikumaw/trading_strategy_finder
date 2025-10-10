# 💎 Platinum Layer: The Strategy Discovery Engine

The Platinum Layer is the intelligent core of the entire pipeline. It transforms the vast, raw data from the Silver Layer into a refined list of **statistically significant trading patterns**.

Its purpose is to act as a powerful noise filter. By analyzing millions of winning trades, it identifies specific strategy blueprints and market conditions that have historically produced a **high density of successful trades**, indicating strong, repeatable patterns that are worthy of the final backtesting stage.

---

## 🏛️ Architectural Overview: A Four-Stage Assembly Line

The Platinum Layer uses a highly efficient, four-script architecture designed to handle massive datasets and maximize CPU utilization. Each script is a specialized stage that prepares data for the next, moving from broad ideas to specific, actionable rules.

1.  **`platinum_combinations_generator.py` (The Architect):**

    -   **Job:** This script scans the huge `silver_data/outcomes.csv` file in memory-safe chunks.
    -   **Output:** It discovers every unique strategy "blueprint" that exists in the data (e.g., "SL placed 70-80% of the way to resistance, TP is a 0.5% ratio") and saves this master list to `platinum_data/combinations/`.
    -   **Why:** It creates the complete search space of all possible strategies we need to investigate.

2.  **`platinum_chunk_maker.py` (The Slicer):**

    -   **Job:** A simple, one-time utility. It takes the same `silver_data/outcomes.csv` file and slices it into smaller, manageable, numbered chunks (e.g., `chunk_1.csv`).
    -   **Why:** This prepares the data for the next, highly parallelized stage.

3.  **`platinum_target_extractor.py` (The Pre-Processor):**

    -   **Job:** This is the heavy-lifting engine. It reads each chunk and filters it against _all 12,000+ strategy blueprints_ at once.
    -   **Output:** It generates thousands of tiny CSV files in `platinum_data/targets/`, where each file contains only the `entry_time` and `trade_count` for a single blueprint.
    -   **Why:** This is a massive optimization that pre-computes all the necessary data, eliminating the I/O bottleneck for the final stage.

4.  **`platinum_strategy_discoverer.py` (The Rule Miner):**
    -   **Job:** This is the final, high-speed machine learning stage. It iterates through the blueprints, reads their tiny, pre-computed target files, and trains a **Decision Tree Regressor**.
    -   **Logic:** The model finds market rules that predict a high trade density. These rules are the final output.
    -   **Why:** Because the heavy lifting is already done, this script is incredibly fast and can be massively parallelized.

---

## 📁 Platinum Folder Structure

This layer introduces several intermediate directories for maximum efficiency.

```
project_root/
│
├── silver_data/
│   ├── chunked_outcomes/  # INTERMEDIATE: Sliced outcome files
│   └── outcomes/          # INPUT: The main source file
│
├── platinum_data/
│   ├── combinations/      # STAGE 1 OUTPUT: The strategy blueprints
│   ├── targets/           # STAGE 2 OUTPUT: Pre-computed results for each blueprint
│   ├── discovered_strategy/ # FINAL OUTPUT: The discovered strategy rules
│   └── blacklists/        # (Feedback loop input)
│
└── scripts/
    ├── platinum_combinations_generator.py # Script 1
    ├── platinum_chunk_maker.py            # Script 2
    ├── platinum_target_extractor.py       # Script 3
    └── platinum_strategy_discoverer.py    # Script 4
```

---

## ⚙️ The Discovery Workflow

The process must be run in this specific order:

1.  **Run `platinum_combinations_generator.py`** to create the master list of blueprints.
2.  **Run `platinum_chunk_maker.py`** to slice the large outcomes file.
3.  **Run `platinum_target_extractor.py`**. This is a long-running but efficient process that will pre-compute all the target data.
4.  **Run `platinum_strategy_discoverer.py`**. This final step is lightning-fast and produces the final `discovered_strategy` files.

This entire workflow is **fully resumable**. You can stop and restart the scripts at any stage without losing progress.

---

## 🧱 Output File Descriptions

### 1. `platinum_data/combinations/{instrument}.csv`

A crucial blueprint file listing every unique way a trade's SL and TP can be defined.

| Column                                 | Description                                                   |
| :------------------------------------- | :------------------------------------------------------------ |
| `type`                                 | The type of binned strategy (e.g., `Semi-Dynamic-SL-Binned`). |
| `sl_def`, `sl_bin`, `tp_def`, `tp_bin` | The specific parameters defining the blueprint.               |

### 2. `platinum_data/targets/{instrument}/{strategy_key}.csv`

An intermediate file. A tiny CSV containing just `entry_time` and `trade_count` for one blueprint.

### 3. `platinum_data/discovered_strategy/{instrument}.csv`

The final, high-value output. Each row is a candidate strategy ready for the Diamond Layer.

| Column                 | Description                                                                                                       |
| :--------------------- | :---------------------------------------------------------------------------------------------------------------- |
| `type`, `sl_def`, etc. | The complete strategy blueprint definition.                                                                       |
| `market_rule`          | The specific market conditions discovered by the ML model.                                                        |
| `avg_trade_density`    | The average number of successful trades per candle when the rule is met. Our measure of statistical significance. |
| `num_candles`          | The number of unique historical candles that fit this rule.                                                       |
| `total_trades`         | The total number of successful simulations represented by those candles.                                          |

---

## 📈 Next Steps

The `discovered_strategy` files are the direct input for the final **Diamond Layer**. The backtester will take these statistically significant patterns and subject them to a rigorous multi-market simulation to determine their real-world profitability and robustness.

# 💎 Platinum Layer: The Strategy Discovery Engine

The Platinum Layer is the intelligent heart of the entire pipeline. It transforms the vast, enriched data from the Silver Layer into a refined and manageable list of **statistically significant trading patterns**. This layer is where raw data becomes actionable intelligence.

Its purpose is to act as a powerful noise filter and pattern detector. By analyzing millions of winning trade simulations, it identifies specific strategy "blueprints" and the precise market conditions under which they have historically shown a high probability of success.

## What It Is

The Platinum Layer is not a single script but a highly efficient, **three-stage data processing assembly line**. Each script is a specialized tool designed to perform one part of the discovery process. This architecture is intentionally designed to handle massive datasets, maximize CPU utilization, and create a fully resumable workflow.

The three stages are:

1.  **Combination Generation:** Identifies every unique strategy "idea" (blueprint) in the data.
2.  **Target Extraction:** A heavy-lifting, "map-reduce" style pre-processing step that prepares the data for the final ML stage.
3.  **Strategy Discovery:** The final, high-speed machine learning stage that uses a Decision Tree to find explicit, human-readable trading rules.

## Why It Is

The preceding layers produce a dataset of millions or even billions of winning trades, each with hundreds of features. This is too much data to analyze manually or with a simple model. The Platinum Layer is essential to systematically navigate this enormous search space.

1.  **To Make an Infinite Problem Finite:** The `combinations_generator` uses a technique called **discretization** (binning). Instead of treating infinite possible SL/TP placements as unique, it groups them into finite buckets (e.g., "SL placed 70-80% of the way to a support level"). This transforms an impossibly large search space into a manageable one.
2.  **To Solve the I/O Bottleneck:** A naive approach would require the final ML script to read a multi-gigabyte data file thousands of times, which would be incredibly slow. The `target_extractor` solves this by pre-computing all the necessary data into thousands of tiny, fast-loading files. This is a massive optimization that shifts the workload to an offline, parallelizable task.
3.  **To Find Explicit Rules, Not Black Boxes:** We use a `DecisionTreeRegressor` because its output is not a probabilistic prediction but a set of simple, human-readable rules (e.g., `RSI <= 30 and session == 'London'`). This is critical for creating strategies that can be understood, validated, and ultimately trusted.
4.  **To Ensure Statistical Significance:** The "lift" filter ensures that we don't just find _any_ rule, but rules that identify market conditions where a strategy performs **significantly better** than its own baseline average. This focuses the output on high-quality, high-probability patterns.

## How It Works

The three scripts must be run in the correct order. The entire workflow is fully **resumable**; you can stop and restart any script without losing progress.

#### **Stage 1: `platinum_combinations_generator.py` (The Architect)**

- **Job:** Scans the pre-chunked outcome files in `silver_data/chunked_outcomes/{instrument}/`.
- **Logic:** It discovers every unique strategy "blueprint" that exists in the data by binning the relational positioning features. For example, a trade with a TP placed 85% of the way to resistance is categorized into "TP to resistance, Bin 8".
- **Output:** It saves a master list of all unique blueprints to `platinum_data/combinations/`. This defines the complete search space.

#### **Stage 2: `platinum_target_extractor.py` (The Pre-Processor)**

- **Job:** This is the heavy-lifting engine. It is a classic "Map" operation in a Map-Reduce paradigm.
- **Logic:** For each blueprint in the combinations file, it generates a unique hash **`key`**. It then iterates through each `chunked_outcomes` file and, for every blueprint, finds all matching trades. It aggregates these trades by their `entry_time` to get a `trade_count`.
- **Output:** It generates thousands of tiny CSV files in `platinum_data/targets/{instrument}/`, each named with a blueprint's unique `key`. Crucially, it then saves the `key` for each blueprint **back to the original combinations file**, creating a single source of truth for the next stage.

#### **Stage 3: `platinum_strategy_discoverer.py` (The Rule Miner)**

- **Job:** This is the final, high-speed machine learning stage—the "Reduce" operation.
- **Logic:** It iterates through each blueprint from the combinations file (which now includes the `key`). For each one:
  1.  It uses the `key` to load the corresponding tiny target file (the `y` variable).
  2.  It merges this with the full ML-ready Gold `features` file (the `X` variables).
  3.  It trains a `DecisionTreeRegressor` model to find rules in `X` that predict a high `y` (high trade density).
  4.  It applies a `DENSITY_LIFT_THRESHOLD` filter to ensure the discovered rule provides a significant statistical edge over the blueprint's baseline.
  5.  It leverages a **blacklist** of keys to ignore blueprints that have already been proven unprofitable, making the system smarter over time.
- **Output:** The final, high-value `discovered_strategy` files, containing the complete strategy definitions (blueprint + market rule).

## 📁 Folder Structure

```
project_root/
├── silver_data/
│   └── chunked_outcomes/      # INPUT: Enriched, chunked trade data
│       └── XAUUSD15/
│           ├── chunk_1.csv
│           └── ...
├── gold_data/
│   └── features/              # INPUT: ML-ready market features
│       └── XAUUSD15.csv
│
├── platinum_data/
│   ├── combinations/          # STAGE 1 OUTPUT, STAGE 2 MODIFIES: Strategy blueprints (key is added)
│   │   └── XAUUSD15.csv
│   ├── targets/               # STAGE 2 OUTPUT: Pre-computed target data
│   │   └── XAUUSD15/
│   │       ├── {key1}.csv
│   │       └── {key2}.csv
│   ├── discovered_strategy/   # STAGE 3 OUTPUT: Final discovered strategies
│   │   └── XAUUSD15.csv
│   └── blacklists/            # FEEDBACK LOOP INPUT: Unprofitable blueprint keys
│       └── XAUUSD15.csv
│
└── scripts/
    ├── platinum_combinations_generator.py
    ├── platinum_target_extractor.py
    └── platinum_strategy_discoverer.py
```

## 📈 Input & Output

### Input Files

1.  **`silver_data/chunked_outcomes/{instrument}/`**: A directory of CSVs containing the enriched trades from the Silver Layer.
2.  **`gold_data/features/{instrument}.csv`**: The ML-ready market features.
3.  **`platinum_data/blacklists/{instrument}.csv` (Optional):** A file generated by backtesting layers containing blueprint `keys` to ignore.

### Output Files

1.  **`platinum_data/discovered_strategy/{instrument}.csv`**: The final product of this layer. Each row is a complete, testable strategy candidate.

| Column                 | Description                                                              | Example                                               |
| :--------------------- | :----------------------------------------------------------------------- | :---------------------------------------------------- |
| `key`                  | The unique hash identifier for the parent strategy blueprint.            | `a1b2c3d4e5f6g7h8`                                    |
| `type`, `sl_def`, etc. | The complete strategy blueprint definition.                              | `Semi-Dynamic-TP-Binned`, `0.025`, `resistance`, `-7` |
| `market_rule`          | The specific market conditions discovered by the ML model.               | `` `RSI_14` <= 30.5 and `ADX` > 25.0 ``               |
| `avg_trade_density`    | The average number of successful trades per candle when the rule is met. | `44.35`                                               |
| `num_candles`          | The number of unique historical candles that fit this rule.              | `96`                                                  |
| `total_trades`         | The total number of successful simulations represented by those candles. | `4257`                                                |

## 🚀 Possible Enhancements & Future Improvements

1.  **Alternative Models for Rule Extraction:** While `DecisionTreeRegressor` is excellent for its interpretability, other models could be used:
    - **Skope-rules:** A Python library specifically designed for finding high-performing, simple rules (often called "rule-based learners").
    - **Explainable Boosting Machines (EBMs):** These models are as accurate as Gradient Boosting but are designed to be interpretable, allowing for the extraction of feature contributions and simple rules.
2.  **More Sophisticated Target Variables:** The current target is `trade_count`. This could be enhanced to `trade_count * avg_pnl_pips` to create a "density-profitability" score, optimizing for rules that find not just frequent wins, but large wins.
3.  **Automated Parameter Tuning:** The `DECISION_TREE_MAX_DEPTH` and other parameters are currently fixed. A meta-layer could be added to experiment with different parameter settings to see which configuration yields the most robust final strategies.
4.  **Feature Importance Analysis:** As an alternative to rule extraction, a `RandomForestRegressor` could be trained to produce a "feature importance" ranking. By aggregating these rankings across all blueprints, the system could identify which of the 200+ indicators are most consistently predictive, helping to simplify the feature set over time.

# Platinum Layer: Strategy Discoverer (`platinum_strategy_discoverer.py`)

This script is the intelligent heart of the entire Strategy Finder pipeline. It uses a machine learning model (`DecisionTreeRegressor`) to act as a **rule miner**, sifting through the millions of potential trades to find explicit, human-readable trading rules that define a robust strategy.

## 🎯 Purpose in the Pipeline

Where the previous layers prepare the data, the Platinum Discoverer _interrogates_ it. For each profitable "blueprint" identified by the `platinum_preprocessor`, this script asks the question: _"Is there a specific, repeatable market context (a 'rule') where this blueprint is exceptionally profitable?"_

It translates the statistical correlations in the data into concrete, testable hypotheses in the form of `if...then` trading rules, which are then passed on to the Diamond and Zircon layers for rigorous backtesting.

## ✨ Key Features & Architecture

This script embodies a **"Two-Phase Learning"** architecture for maximum efficiency and continuous improvement.

### Phase 1: Discovery

This is a comprehensive, one-time run to find initial rules for all new and unprocessed strategy blueprints. It scans for blueprints that have never been seen before and subjects them to the Decision Tree model to find a first set of high-quality trading rules. This entire phase is **resumable**; if it's interrupted, it will pick up where it left off on the next run.

### Phase 2: Iterative Improvement

This is a fast, targeted run that forms the core of the system's **feedback loop**. It ONLY re-evaluates blueprints for which the backtester has provided new negative feedback (i.e., new rules added to that blueprint's blacklist). It uses a **"data pruning"** technique:

1. It loads the known unprofitable rules for a blueprint.
2. It finds all the data points (candles) that match those bad rules.
3. It temporarily sets the profitability of those data points to zero.
4. It then re-trains the Decision Tree on this "pruned" dataset, forcing the model to find novel, alternative rules and avoid patterns that are known to be unprofitable.

This makes the system smarter and more efficient over time.

---

## ⚙️ How It Works: The Machine Learning Logic

1.  **File Discovery & State Management:** The script first identifies which blueprints are new (for Phase 1) and which have new blacklist feedback (for Phase 2) by comparing various log and output files.
2.  **Parallel Processing:** The list of blueprints to process is divided into small batches, which are distributed to a pool of worker processes.
3.  **Worker Task:** Each worker performs the following for its assigned blueprints:
    a. **Load Data:** It loads the pre-computed performance data (the "target") for a specific blueprint and merges it with the master market context (the "features") from the Gold layer.
    b. **Prune Data:** It checks for any known bad rules (from previous discoveries or the blacklist) and prunes the dataset as described above.
    c. **Train Model:** It trains a `DecisionTreeRegressor` model on the data. The goal of the model is to predict `trade_count` (profitability) based on the market features.
    d. **Extract Rules:** The script then traverses the trained decision tree, converting each path to a leaf node into a human-readable rule (e.g., `` `RSI_14` <= -1.5 and `session_London` > 0.5 ``).
    e. **Filter for Quality:** Only rules that meet strict quality criteria are kept:
    _ `MIN_CANDLES_PER_RULE`: The rule must apply to a minimum number of historical situations.
    _ `DENSITY_LIFT_THRESHOLD`: The average profitability of candles matching the rule must be significantly higher (e.g., 1.5x) than the blueprint's overall average profitability.
4.  **Save Results:** The main process collects the results and saves the newly discovered, high-quality rules to the `/platinum_data/discovered_strategies` directory. Blueprints for which no good rules could be found are marked as "exhausted".

---

## 🛠️ Dependencies

This script requires several specialized libraries. Install them via pip:```bash
pip install pyarrow scikit-learn

````

---

## 🔧 Configuration

Key parameters for the ML model and rule quality can be tuned directly in the global configuration section at the top of the script:

-   `MIN_CANDLE_LIMIT`: Pre-filters and ignores any blueprint that didn't apply to at least this many candles in total.
-   `DECISION_TREE_MAX_DEPTH`: Controls the complexity of the rules the model can discover. A deeper tree can find more complex, multi-condition rules but risks overfitting.
-   `MIN_CANDLES_PER_RULE`: A key parameter to prevent overfitting. It forces any discovered rule to be statistically significant by ensuring it's based on a sufficient number of historical examples.
-   `DENSITY_LIFT_THRESHOLD`: Controls how much better a rule must be compared to the baseline. A higher value leads to fewer, but higher-quality, rules.

---

## 🚀 Usage

Execute the script from the root directory of the project.

**1. Interactive Mode (Recommended):**

The script will scan for instruments with processed combination files and present a menu.

```bash
python scripts/platinum_strategy_discoverer.py
````

**2. Targeted Mode:**

To process a specific instrument, pass its name as an argument.

```bash
python scripts/platinum_strategy_discoverer.py EURUSD15
```

---

## 📄 Output

This script modifies several files in the `/platinum_data/` directory:

- **`/discovered_strategies/{instrument}.parquet`:** The primary output. This file is appended with all the new, high-quality market rules discovered during the run.
- **`/exhausted_keys/{instrument}.parquet`:** Appended with the keys of any blueprints for which the model could not find any good rules.
- **`/discovery_log/{instrument}.processed.log`:** A log file that tracks which blueprints have been processed in Phase 1, making the script resumable.

import pandas as pd
import os
import re

# --- CONFIGURATION (Match these with your diamond_backtester.py) ---
CORE_DIR = os.path.dirname(os.path.abspath(__file__))
DIAMOND_RESULTS_DIR = os.path.abspath(os.path.join(CORE_DIR, '..', 'diamond_data/backtesting_results'))
ZIRCON_INPUT_DIR = os.path.abspath(os.path.join(CORE_DIR, '..', 'zircon_data/input'))

# "Mastery" Pass/Fail Thresholds - THESE ARE COPIED DIRECTLY FROM YOUR `diamond_backtester.py`
PROFIT_FACTOR_PASS_THRESHOLD = 1.5
MAX_DRAWDOWN_PASS_THRESHOLD = 25.0  # Maximum acceptable drawdown percentage.
MIN_TRADES_PASS_THRESHOLD = 50      # Minimum number of trades to be statistically valid.
# --- END CONFIGURATION ---

def rebuild_zircon_input():
    """
    Scans all diamond_report_*.csv files, extracts all unique "master" strategies,
    and consolidates them into the appropriate zircon_data/input files.
    """
    os.makedirs(ZIRCON_INPUT_DIR, exist_ok=True)
    all_master_strategies_by_market = {}

    print(f"Scanning for Diamond reports in: {DIAMOND_RESULTS_DIR}")
    report_files = [f for f in os.listdir(DIAMOND_RESULTS_DIR) if f.startswith('diamond_report_') and f.endswith('.csv')]

    if not report_files:
        print("❌ No Diamond report files found. Ensure 'diamond_data/backtesting_results' contains reports.")
        return

    for report_file in report_files:
        report_path = os.path.join(DIAMOND_RESULTS_DIR, report_file)
        # Extract the original market name from the filename (e.g., XAUUSD15 from diamond_report_XAUUSD15.csv)
        match = re.search(r'diamond_report_(.+?)\.csv', report_file)
        if not match:
            print(f"⚠️ Could not parse market name from {report_file}. Skipping.")
            continue
        origin_market_name = match.group(1)

        print(f"Processing report: {report_file} for market: {origin_market_name}")
        try:
            # Read with object dtype for sl_def/tp_def to avoid mixed type warnings
            # And read `strategy_id` as string to prevent issues with leading zeros if it's numeric
            df = pd.read_csv(report_path, dtype={'sl_def': object, 'tp_def': object, 'strategy_id': str})
            
            # Filter for strategies that passed all mastery thresholds.
            # These thresholds are matched with your diamond_backtester.py script.
            passed_strategies_df = df[
                (df['profit_factor'] > PROFIT_FACTOR_PASS_THRESHOLD) &
                (df['max_drawdown_pct'] < MAX_DRAWDOWN_PASS_THRESHOLD) &
                (df['total_trades'] >= MIN_TRADES_PASS_THRESHOLD)
            ].copy()

            if not passed_strategies_df.empty:
                # Store or append to existing strategies for this market
                if origin_market_name not in all_master_strategies_by_market:
                    all_master_strategies_by_market[origin_market_name] = passed_strategies_df
                else:
                    all_master_strategies_by_market[origin_market_name] = pd.concat([
                        all_master_strategies_by_market[origin_market_name],
                        passed_strategies_df
                    ]).drop_duplicates(subset=['strategy_id']) # Ensure unique strategies per market
                print(f"  -> Found {len(passed_strategies_df)} new master strategies (before de-duplication for this market).")
            else:
                print(f"  -> No master strategies found in {report_file} based on current thresholds.")

        except Exception as e:
            print(f"❌ Error processing {report_file}: {e}")

    if not all_master_strategies_by_market:
        print("\nNo master strategies were found across all reports. Zircon input will be empty.")
        return

    print("\nConsolidating and saving Zircon input files...")
    for market, strategies_df in all_master_strategies_by_market.items():
        zircon_output_path = os.path.join(ZIRCON_INPUT_DIR, f"master_strategies_{market}.csv")
        strategies_df.to_csv(zircon_output_path, index=False)
        print(f"✅ Saved {len(strategies_df)} unique master strategies for {market} to {zircon_output_path}")

    print("\n🎉 Zircon Layer input data successfully rebuilt!")

if __name__ == "__main__":
    rebuild_zircon_input()
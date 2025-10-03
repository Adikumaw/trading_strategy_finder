import os
import re # Import the regular expressions module
import pandas as pd
import numpy as np
from tqdm import tqdm

# --- Timeframe Presets for SL/TP ---
# Maps the timeframe identifier (e.g., "1m") to its specific processing parameters.
TIMEFRAME_PRESETS = {
    "1m": {
        "SL_RATIOS": np.arange(0.001, 0.0105, 0.001),   # 0.1% - 1.0%
        "TP_RATIOS": np.arange(0.001, 0.0205, 0.001),   # 0.1% - 2.0%
        "MAX_LOOKFORWARD": 200
    },
    "5m": {
        "SL_RATIOS": np.arange(0.002, 0.0155, 0.001),   # 0.2% - 1.5%
        "TP_RATIOS": np.arange(0.002, 0.0305, 0.001),   # 0.2% - 3.0%
        "MAX_LOOKFORWARD": 300
    },
    "15m": {
        "SL_RATIOS": np.arange(0.003, 0.0205, 0.001),   # 0.3% - 2.0%
        "TP_RATIOS": np.arange(0.003, 0.0405, 0.001),   # 0.3% - 4.0%
        "MAX_LOOKFORWARD": 400
    },
    "30m": {
        "SL_RATIOS": np.arange(0.005, 0.0255, 0.001),   # 0.5% - 2.5%
        "TP_RATIOS": np.arange(0.005, 0.0505, 0.001),   # 0.5% - 5.0%
        "MAX_LOOKFORWARD": 500
    },
    "60m": {
        "SL_RATIOS": np.arange(0.010, 0.0305, 0.001),   # 1.0% - 3.0%
        "TP_RATIOS": np.arange(0.010, 0.0705, 0.001),   # 1.0% - 7.0%
        "MAX_LOOKFORWARD": 600
    },
    "240m": {
        "SL_RATIOS": np.arange(0.020, 0.0505, 0.001),   # 2.0% - 5.0%
        "TP_RATIOS": np.arange(0.020, 0.1005, 0.001),   # 2.0% - 10.0%
        "MAX_LOOKFORWARD": 800
    }
}

def get_config_from_filename(filename):
    """
    Parses the filename to find a timeframe number and returns the corresponding config.
    Example: "EURUSD15.csv" -> extracts "15" -> returns config for "15m".
    """
    # This regex finds any numbers that come right before the '.csv' extension
    match = re.search(r'(\d+)\.csv$', filename)
    if match:
        timeframe_num = match.group(1)
        timeframe_key = f"{timeframe_num}m" # e.g., "15m"
        if timeframe_key in TIMEFRAME_PRESETS:
            print(f"✅ Timeframe '{timeframe_key}' detected for {filename}.")
            return TIMEFRAME_PRESETS[timeframe_key]
    
    print(f"⚠️ Could not determine a valid timeframe preset for {filename}. Skipping file.")
    return None

def load_and_process(csv_file: str, output_file: str, SL_RATIOS, TP_RATIOS, MAX_LOOKFORWARD):
    """
    Loads historical data, identifies all possible profitable trade combinations,
    and saves them to a new CSV file. This function's core logic is unchanged.
    """
    print(f"Loading data from {os.path.basename(csv_file)}...")
    try:
        df = pd.read_csv(csv_file, sep=None, engine="python")
        if df.shape[1] > 5:
            df = df.iloc[:, :5]
        df.columns = ["time", "open", "high", "low", "close"]
        df["time"] = pd.to_datetime(df["time"])
        df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].apply(pd.to_numeric)
    except Exception as e:
        print(f"❌ Error loading or parsing CSV file: {e}")
        return

    profitable_trades = []

    for i in tqdm(range(len(df) - 1), desc="Processing Candles"):
        entry_time = df.loc[i, "time"]
        entry_price = df.loc[i, "close"]

        buy_tps_to_check = {round(entry_price * (1 + r), 5): r for r in TP_RATIOS}
        buy_sls_to_check = {round(entry_price * (1 - r), 5): r for r in SL_RATIOS}
        
        sell_tps_to_check = {round(entry_price * (1 - r), 5): r for r in TP_RATIOS}
        sell_sls_to_check = {round(entry_price * (1 + r), 5): r for r in SL_RATIOS}

        lookforward_limit = min(i + 1 + MAX_LOOKFORWARD, len(df))
        for j in range(i + 1, lookforward_limit):
            future_high = df.loc[j, "high"]
            future_low = df.loc[j, "low"]
            exit_time = df.loc[j, "time"]

            # --- BUY Trades ---
            if buy_sls_to_check:
                hit_buy_tps = {p: r for p, r in buy_tps_to_check.items() if p <= future_high}
                if hit_buy_tps:
                    for tp_price, tp_ratio in hit_buy_tps.items():
                        for sl_price, sl_ratio in buy_sls_to_check.items():
                            profitable_trades.append({
                                "entry_time": entry_time, "trade_type": "buy", "entry_price": entry_price,
                                "sl_price": sl_price, "tp_price": tp_price, "sl_ratio": sl_ratio,
                                "tp_ratio": tp_ratio, "exit_time": exit_time, "outcome": "win",
                            })
                    buy_tps_to_check = {p: r for p, r in buy_tps_to_check.items() if p not in hit_buy_tps}

                hit_buy_sls = {p for p in buy_sls_to_check if p >= future_low}
                if hit_buy_sls:
                    buy_sls_to_check = {p: r for p, r in buy_sls_to_check.items() if p not in hit_buy_sls}
            
            # --- SELL Trades ---
            if sell_sls_to_check:
                hit_sell_tps = {p: r for p, r in sell_tps_to_check.items() if p >= future_low}
                if hit_sell_tps:
                    for tp_price, tp_ratio in hit_sell_tps.items():
                        for sl_price, sl_ratio in sell_sls_to_check.items():
                            profitable_trades.append({
                                "entry_time": entry_time, "trade_type": "sell", "entry_price": entry_price,
                                "sl_price": sl_price, "tp_price": tp_price, "sl_ratio": sl_ratio,
                                "tp_ratio": tp_ratio, "exit_time": exit_time, "outcome": "win",
                            })
                    sell_tps_to_check = {p: r for p, r in sell_tps_to_check.items() if p not in hit_sell_tps}

                hit_sell_sls = {p for p in sell_sls_to_check if p <= future_high}
                if hit_sell_sls:
                    sell_sls_to_check = {p: r for p, r in sell_sls_to_check.items() if p not in hit_sell_sls}

            if not buy_sls_to_check and not sell_sls_to_check:
                break
    
    if not profitable_trades:
        print("⚠️ No profitable trade combinations found with the current settings.")
        return

    print(f"\n✅ Found {len(profitable_trades)} profitable trades.")
    results_df = pd.DataFrame(profitable_trades)
    results_df.to_csv(output_file, index=False)
    print(f"SUCCESS: Data saved to {output_file}")


if __name__ == "__main__":
    core_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.abspath(os.path.join(core_dir, '..', 'raw_data'))
    bronze_data_dir = os.path.abspath(os.path.join(core_dir, '..', 'bronze_data'))
    os.makedirs(bronze_data_dir, exist_ok=True)

    try:
        raw_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.csv')]
    except FileNotFoundError:
        print(f"❌ Error: The directory '{raw_data_dir}' was not found.")
        raw_files = []

    if not raw_files: 
        print("❌ No CSV files found in 'raw_data'.")
    else: 
        print(f"Found {len(raw_files)} files to process...")

    for filename in raw_files:
        print("\n" + "="*50 + f"\nProcessing: {filename}")
        
        # 🔥 Automatically get config based on filename
        config = get_config_from_filename(filename)
        
        # If config is None, it means timeframe was not found, so we skip to the next file
        if config is None:
            continue
            
        try:
            input_path = os.path.join(raw_data_dir, filename)
            output_path = os.path.join(bronze_data_dir, filename)
            
            # Pass the specific parameters for this timeframe to the processing function
            load_and_process(
                input_path, 
                output_path, 
                SL_RATIOS=config["SL_RATIOS"], 
                TP_RATIOS=config["TP_RATIOS"], 
                MAX_LOOKFORWARD=config["MAX_LOOKFORWARD"]
            )
        except Exception as e:
            print(f"❌ FAILED to process {filename}. Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*50 + "\nBronze data generation complete.")
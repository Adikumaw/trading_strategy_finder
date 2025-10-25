# start_pipeline.py (V5 - Final & Fully Compatible)
import os
import sys
import subprocess
from time import sleep

# --- ANSI Color Codes for Better Terminal Output ---
class colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# --- Configuration ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(ROOT_DIR, 'scripts')
RAW_DATA_DIR = os.path.join(ROOT_DIR, 'raw_data')

def run_script(script_name, args=None):
    """
    Executes a script, passing command-line arguments for non-interactive execution.
    """
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    if not os.path.exists(script_path):
        print(f"{colors.RED}[ERROR] Script not found at {script_path}{colors.ENDC}")
        return False

    command = [sys.executable, script_path]
    if args:
        command.extend(args)

    print(f"\n{colors.HEADER}--- Executing: {' '.join(command)} ---{colors.ENDC}")
    
    try:
        # Stream the output directly to the console for real-time feedback.
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, text=True, encoding='utf-8', errors='replace') as process:
            for line in process.stdout:
                print(line, end='')
        
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)

        print(f"{colors.GREEN}[SUCCESS] {script_name} completed.{colors.ENDC}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{colors.RED}[FAILED] {script_name} exited with return code {e.returncode}.{colors.ENDC}")
        return False
    except Exception as e:
        print(f"{colors.RED}[ERROR] An unexpected error occurred while running {script_name}: {e}{colors.ENDC}")
        return False

def main():
    """The main function to orchestrate the entire pipeline."""
    print(f"{colors.BOLD}{colors.BLUE}===== Starting the Full Strategy Discovery Pipeline ====={colors.ENDC}")

    # --- Step 1: Select the Target Market for DISCOVERY ---
    try:
        raw_files = sorted([f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.csv')])
        if not raw_files:
            print(f"{colors.RED}[ERROR] No raw data files found in '{RAW_DATA_DIR}'. Exiting.{colors.ENDC}")
            return
    except FileNotFoundError:
        print(f"{colors.RED}[ERROR] Raw data directory not found at '{RAW_DATA_DIR}'. Exiting.{colors.ENDC}")
        return

    print("\n--- Select a Raw Market File for the Full Pipeline ---")
    for i, f in enumerate(raw_files):
        print(f"  [{i+1}] {f}")
    
    try:
        choice = int(input(f"Enter the number of the file to process (1-{len(raw_files)}): ")) - 1
        if not 0 <= choice < len(raw_files): raise ValueError
        target_market_file = raw_files[choice]
        print(f"{colors.CYAN}[INFO] You selected: {target_market_file}. The pipeline will now run non-interactively.{colors.ENDC}")
    except (ValueError, IndexError):
        print(f"{colors.RED}[ERROR] Invalid selection. Exiting.{colors.ENDC}")
        return
    
    # --- Define the full sequence of scripts and their arguments ---
    pipeline_stages = [
        {"name": "diamond_data_prepper.py", "args": [target_market_file]},
        {"name": "bronze_data_generator.py", "args": [target_market_file]},
        {"name": "silver_data_generator.py", "args": [target_market_file]},
        {"name": "gold_data_generator.py", "args": [target_market_file]},
        {"name": "platinum_combination_generator.py", "args": [target_market_file]},
        {"name": "platinum_target_extractor.py", "args": [target_market_file]},
        {"name": "platinum_strategy_discoverer.py", "args": [target_market_file]},
        {"name": "diamond_backtester.py", "args": [target_market_file]}, 
        {"name": "zircon_validator.py", "args": [target_market_file]},
    ]

    for stage in pipeline_stages:
        sleep(1)
        if not run_script(stage["name"], args=stage.get("args")):
            print(f"\n{colors.RED}{colors.BOLD}Pipeline halted due to an error in {stage['name']}. Please review the output above.{colors.ENDC}")
            return
            
    print(f"\n{colors.GREEN}{colors.BOLD}>>>>> Full pipeline completed successfully! <<<<<{colors.ENDC}")
    print(f"{colors.YELLOW}You can now launch the analyser to view the results:{colors.ENDC}")
    print(f"{colors.CYAN}streamlit run app.py{colors.ENDC}")

if __name__ == "__main__":
    main()
# platinum_chunk_maker.py

import pandas as pd
import os
from tqdm import tqdm

# --- CONFIGURATION ---
CHUNK_SIZE = 1_000_000 # Each output file will have 1 million rows

if __name__ == "__main__":
    core_dir = os.path.dirname(os.path.abspath(__file__))
    silver_outcomes_dir = os.path.abspath(os.path.join(core_dir, '..', 'silver_data', 'outcomes'))
    chunked_outcomes_dir = os.path.abspath(os.path.join(core_dir, '..', 'silver_data', 'chunked_outcomes'))
    
    os.makedirs(chunked_outcomes_dir, exist_ok=True)

    try:
        outcome_files = [f for f in os.listdir(silver_outcomes_dir) if f.endswith('.csv')]
    except FileNotFoundError:
        print(f"❌ Source directory not found: {silver_outcomes_dir}"); outcome_files = []

    if not outcome_files:
        print("❌ No silver outcome files found to chunk.")
    else:
        for fname in outcome_files:
            instrument_name = fname.replace('.csv', '')
            output_dir = os.path.join(chunked_outcomes_dir, instrument_name)
            
            if os.path.exists(output_dir):
                print(f"ℹ️ Chunked directory for {instrument_name} already exists. Skipping.")
                continue

            os.makedirs(output_dir, exist_ok=True)
            source_path = os.path.join(silver_outcomes_dir, fname)
            
            print(f"\nChunking {fname} into files of {CHUNK_SIZE} rows...")
            
            try:
                iterator = pd.read_csv(source_path, chunksize=CHUNK_SIZE)
                for i, chunk in enumerate(tqdm(iterator, desc=f"Writing chunks for {instrument_name}")):
                    chunk.to_csv(os.path.join(output_dir, f"chunk_{i+1}.csv"), index=False)
                
                print(f"✅ Successfully created chunks for {instrument_name} in {output_dir}")
            except Exception as e:
                print(f"❌ FAILED to chunk {fname}. Error: {e}")

    print("\n" + "="*50 + "\n✅ All chunking complete.")
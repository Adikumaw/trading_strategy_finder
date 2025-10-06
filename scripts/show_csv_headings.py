import pandas as pd
import os

def load_processed_csv(csv_file="../silver_data/EURUSD15.csv"):
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"❌ File {csv_file} not found in the current directory!")
        return None

    # Load CSV
    df = pd.read_csv(csv_file, parse_dates=["time"])

    print(f"✅ CSV loaded successfully! Shape: {df.shape}")
    # print("\n📌 First 10 rows:")
    # print(df.head(10))  

    # Print all column headings as a list
    print("\n📌 Column Headings:")
    print(df.columns.tolist())

    return df


if __name__ == "__main__":
    df = load_processed_csv()

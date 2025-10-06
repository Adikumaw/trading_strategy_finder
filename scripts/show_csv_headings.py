import pandas as pd
import os

def get_csv_headings(csv_file):
    """Fetch column names efficiently without loading the entire CSV."""
    try:
        with pd.read_csv(csv_file, iterator=True, chunksize=1, low_memory=True) as reader:
            columns = reader._engine.names
        return columns
    except Exception as e:
        print(f"❌ Error reading headings: {e}")
        return None


def show_csv_preview(csv_file, num_rows=5):
    """Load just the first few rows (num_rows) safely, with full display."""
    try:
        # Read only required rows
        df_preview = pd.read_csv(csv_file, nrows=num_rows, low_memory=True)

        # Configure pandas to show all columns & full width
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        pd.set_option("display.max_colwidth", None)
        pd.set_option("display.expand_frame_repr", False)

        print(f"\n✅ Showing first {num_rows} rows:\n")
        print(df_preview.to_string(index=False))  # full untruncated view

    except Exception as e:
        print(f"❌ Error reading preview rows: {e}")


def main():
    print("📁 Large CSV Explorer")
    csv_file = input("Enter CSV file path (default: ../silver_data/AUDUSD1.csv): ").strip() or "../silver_data/AUDUSD1.csv"

    if not os.path.exists(csv_file):
        print(f"❌ File '{csv_file}' not found!")
        return

    print("\nWhat do you want to do?")
    print("1️⃣  Show only column headings")
    print("2️⃣  Show column headings + first few rows")
    choice = input("Enter choice (1 or 2): ").strip()

    headings = get_csv_headings(csv_file)
    if headings:
        print("\n📌 Column Headings:")
        print(headings)

        if choice == "2":
            try:
                num_rows = int(input("How many rows to preview? (e.g., 5): ").strip())
            except ValueError:
                num_rows = 5
            show_csv_preview(csv_file, num_rows)
    else:
        print("⚠️ Could not extract headings from the file.")


if __name__ == "__main__":
    main()

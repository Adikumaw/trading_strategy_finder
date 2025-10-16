import pandas as pd
import os
import sys

def is_parquet_file(path):
    return os.path.splitext(path)[1].lower() in ('.parquet', '.pq', '.parq')

def get_csv_headings(csv_file):
    """Fetch column names efficiently without loading the entire CSV or Parquet."""
    # Support parquet transparently
    if is_parquet_file(csv_file):
        try:
            # Prefer pyarrow metadata extraction if available (most efficient)
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(csv_file)
            schema = pf.schema_arrow
            return list(schema.names)
        except Exception:
            # Fallback to pandas read_parquet (may load more data)
            try:
                df = pd.read_parquet(csv_file)
                return df.columns.tolist()
            except Exception as e:
                print(f"❌ Error reading parquet headings: {e}")
                return None

    # CSV path (original behavior)
    try:
        with pd.read_csv(csv_file, iterator=True, chunksize=1, low_memory=True) as reader:
            columns = reader._engine.names
        return columns
    except Exception as e:
        print(f"❌ Error reading headings: {e}")
        return None


def show_csv_preview(csv_file, num_rows=5):
    """Load just the first few rows (num_rows) safely, with full display. Supports CSV and Parquet."""
    try:
        if is_parquet_file(csv_file):
            # Use pandas to load a small preview for parquet
            try:
                df_preview = pd.read_parquet(csv_file)
                df_preview = df_preview.head(num_rows)
            except Exception as e:
                # If direct read_parquet fails, try pyarrow table -> pandas slice
                try:
                    import pyarrow.parquet as pq
                    table = pq.read_table(csv_file)
                    df_preview = table.to_pandas().head(num_rows)
                except Exception as e2:
                    raise RuntimeError(f"Failed to read parquet preview: {e} / {e2}")
        else:
            # Read only required rows for CSV
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
    print("📁 Large CSV/Parquet Explorer")
    default_path = "../silver_data/AUDUSD1.csv"
    csv_file = input(f"Enter file path (CSV or Parquet). default: {default_path}: ").strip() or default_path

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

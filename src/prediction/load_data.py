from pathlib import Path
import csv

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"
TARGET_COLUMN = "Machine failure"


def load_dataset(data_path: Path = DATA_PATH):
    """Load the raw AI4I dataset."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    if pd is None:
        with data_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            rows = list(reader)
        return rows

    df = pd.read_csv(data_path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    return df


def print_pandas_overview(df) -> None:
    print("Dataset loaded successfully")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    print("\nColumns:")
    for column in df.columns:
        print(f"- {column}")

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isna().sum())

    print(f"\nDuplicate rows: {df.duplicated().sum()}")

    if TARGET_COLUMN in df.columns:
        print(f"\nTarget distribution: {TARGET_COLUMN}")
        counts = df[TARGET_COLUMN].value_counts().sort_index()
        percentages = df[TARGET_COLUMN].value_counts(normalize=True).sort_index() * 100
        target_summary = pd.DataFrame(
            {
                "count": counts,
                "percentage": percentages.round(2),
            }
        )
        print(target_summary)


def print_csv_overview(rows: list[dict[str, str]]) -> None:
    print("Dataset loaded successfully")
    print("Note: pandas is not installed, so using standard-library CSV checks.")

    if not rows:
        print("Shape: 0 rows x 0 columns")
        return

    columns = list(rows[0].keys())
    print(f"Shape: {len(rows)} rows x {len(columns)} columns")

    print("\nColumns:")
    for column in columns:
        print(f"- {column}")

    print("\nMissing values:")
    for column in columns:
        missing_count = sum(row[column] == "" for row in rows)
        print(f"{column}: {missing_count}")

    duplicate_count = len(rows) - len({tuple(row.items()) for row in rows})
    print(f"\nDuplicate rows: {duplicate_count}")

    if TARGET_COLUMN in columns:
        print(f"\nTarget distribution: {TARGET_COLUMN}")
        counts = {}
        for row in rows:
            target_value = row[TARGET_COLUMN]
            counts[target_value] = counts.get(target_value, 0) + 1

        for target_value in sorted(counts):
            percentage = counts[target_value] / len(rows) * 100
            print(f"{target_value}: {counts[target_value]} rows ({percentage:.2f}%)")


def print_dataset_overview(dataset) -> None:
    if pd is not None:
        print_pandas_overview(dataset)
    else:
        print_csv_overview(dataset)


def main() -> None:
    dataset = load_dataset()
    print_dataset_overview(dataset)


if __name__ == "__main__":
    main()

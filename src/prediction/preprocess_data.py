from pathlib import Path

from load_data import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "model_input.csv"

ID_COLUMNS = ["UDI", "Product ID"]
LEAKAGE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
COLUMN_RENAME_MAP = {
    "Type": "product_quality",
    "Air temperature [K]": "air_temperature_k",
    "Process temperature [K]": "process_temperature_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
}
TARGET_COLUMN = "machine_failure"


def drop_unusable_columns(df):
    """Remove identifiers and failure-type leakage columns before modeling."""
    columns_to_drop = ID_COLUMNS + LEAKAGE_COLUMNS
    existing_columns = [column for column in columns_to_drop if column in df.columns]
    return df.drop(columns=existing_columns), existing_columns


def rename_columns(df):
    """Use code-friendly column names for the model input dataset."""
    return df.rename(columns=COLUMN_RENAME_MAP)


def save_model_input(df, output_path: Path = PROCESSED_DATA_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    df = load_dataset()
    cleaned_df, dropped_columns = drop_unusable_columns(df)
    cleaned_df = rename_columns(cleaned_df)
    save_model_input(cleaned_df)

    print("Preprocessing completed successfully")
    print(f"Dropped columns: {', '.join(dropped_columns)}")
    print(f"Output path: {PROCESSED_DATA_PATH}")
    print(f"Shape: {cleaned_df.shape[0]} rows x {cleaned_df.shape[1]} columns")

    print("\nRenamed columns:")
    for column in cleaned_df.columns:
        print(f"- {column}")

    if TARGET_COLUMN in cleaned_df.columns:
        print(f"\nTarget column preserved: {TARGET_COLUMN}")


if __name__ == "__main__":
    main()

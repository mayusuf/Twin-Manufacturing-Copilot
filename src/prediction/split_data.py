"""Create reproducible, stratified training and test datasets."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from feature_engineering import FEATURE_OUTPUT_PATH, PROJECT_ROOT, TARGET


TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
TEST_SIZE = 0.20
RANDOM_STATE = 42


def load_features(input_path: Path = FEATURE_OUTPUT_PATH) -> pd.DataFrame:
    """Load the engineered table before any scaling or resampling."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Features not found: {input_path}. Run feature_engineering.py first."
        )
    df = pd.read_csv(input_path)
    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}.")
    if df.empty or df.isna().any().any():
        raise ValueError("Splitting requires nonempty data without missing values.")
    if set(df[TARGET].unique()) != {0, 1}:
        raise ValueError("Expected both machine_failure classes: 0 and 1.")
    return df


def split_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Preserve failure proportions while assigning each row to one set."""
    return train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET],
    )


def main() -> None:
    features = load_features()
    train, test = split_features(features)
    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)

    print(f"Stratified split completed (random_state={RANDOM_STATE})")
    for name, df, path in (("Training", train, TRAIN_PATH), ("Test", test, TEST_PATH)):
        failures = int(df[TARGET].sum())
        print(
            f"{name}: {len(df):,} rows; "
            f"{failures:,} failures ({failures / len(df):.2%})"
        )
        print(f"Saved to: {path}")


if __name__ == "__main__":
    main()

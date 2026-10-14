"""Build leakage-safe engineered features for predictive maintenance modeling."""

from pathlib import Path
import math

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "model_input.csv"
FEATURE_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "model_features.csv"
REPORT_DIR = PROJECT_ROOT / "reports"
TABLE_DIR = REPORT_DIR / "tables"
FEATURE_DICTIONARY_PATH = TABLE_DIR / "feature_dictionary.csv"

TARGET = "machine_failure"
QUALITY_LEVELS = ["L", "M", "H"]
BASE_NUMERIC_FEATURES = [
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]
PREDICTOR_COLUMNS = ["product_quality", *BASE_NUMERIC_FEATURES]
REQUIRED_COLUMNS = [*PREDICTOR_COLUMNS, TARGET]
FEATURE_COLUMNS = [
    *BASE_NUMERIC_FEATURES,
    "temperature_difference_k",
    "temperature_ratio",
    "mechanical_power_kw",
    "torque_per_krpm",
    "tool_wear_x_torque",
    "tool_wear_x_power",
    "product_quality_L",
    "product_quality_M",
    "product_quality_H",
]


FEATURE_DESCRIPTIONS = {
    "air_temperature_k": "Original air temperature in kelvin.",
    "process_temperature_k": "Original process temperature in kelvin.",
    "rotational_speed_rpm": "Original rotational speed in revolutions per minute.",
    "torque_nm": "Original torque in newton-meters.",
    "tool_wear_min": "Original tool wear duration in minutes.",
    "temperature_difference_k": "Process temperature minus air temperature.",
    "temperature_ratio": "Process temperature divided by air temperature.",
    "mechanical_power_kw": "Mechanical power estimate from torque and rotational speed.",
    "torque_per_krpm": "Torque normalized by rotational speed in thousands of rpm.",
    "tool_wear_x_torque": "Interaction between accumulated tool wear and torque.",
    "tool_wear_x_power": "Interaction between accumulated tool wear and mechanical power.",
    "product_quality_L": "One-hot indicator for low-quality products.",
    "product_quality_M": "One-hot indicator for medium-quality products.",
    "product_quality_H": "One-hot indicator for high-quality products.",
    TARGET: "Binary target: 1 if a machine failure occurred, otherwise 0.",
}


def load_model_input(input_path: Path = MODEL_INPUT_PATH) -> pd.DataFrame:
    """Load the cleaned model input table produced by preprocess_data.py."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Model input not found: {input_path}. Run preprocess_data.py first."
        )

    df = pd.read_csv(input_path)
    validate_model_input(df)
    return df


def validate_model_input(df: pd.DataFrame) -> None:
    """Fail early if feature engineering receives unexpected input."""
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    validate_predictor_input(df)
    if df[TARGET].isna().any() or set(df[TARGET].unique()) - {0, 1}:
        raise ValueError("Expected binary machine_failure values of 0 and 1.")


def validate_predictor_input(df: pd.DataFrame) -> None:
    """Validate raw predictors without requiring a known failure label."""
    missing = set(PREDICTOR_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required predictor columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Feature engineering requires a nonempty dataset.")
    if df[PREDICTOR_COLUMNS].isna().any().any():
        raise ValueError("Feature engineering requires data without missing values.")
    invalid_quality = sorted(set(df["product_quality"]) - set(QUALITY_LEVELS))
    if invalid_quality:
        raise ValueError(f"Unexpected product quality levels: {invalid_quality}")
    for column in BASE_NUMERIC_FEATURES:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(f"Expected numeric values in {column}.")
    if not np.isfinite(df[BASE_NUMERIC_FEATURES].to_numpy()).all():
        raise ValueError("Numeric predictors must contain finite values.")
    if (df["air_temperature_k"] <= 0).any():
        raise ValueError("Air temperature must be positive for ratio features.")
    if (df["rotational_speed_rpm"] <= 0).any():
        raise ValueError("Rotational speed must be positive for normalized torque.")


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create physics-inspired features without using the target column."""
    features = df.copy()
    speed_rad_per_sec = features["rotational_speed_rpm"] * (2 * math.pi / 60)

    features["temperature_difference_k"] = (
        features["process_temperature_k"] - features["air_temperature_k"]
    )
    features["temperature_ratio"] = (
        features["process_temperature_k"] / features["air_temperature_k"]
    )
    features["mechanical_power_kw"] = features["torque_nm"] * speed_rad_per_sec / 1000
    features["torque_per_krpm"] = (
        features["torque_nm"] / (features["rotational_speed_rpm"] / 1000)
    )
    features["tool_wear_x_torque"] = (
        features["tool_wear_min"] * features["torque_nm"]
    )
    features["tool_wear_x_power"] = (
        features["tool_wear_min"] * features["mechanical_power_kw"]
    )
    return features


def encode_product_quality(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode product quality with a stable L, M, H column order."""
    encoded = df.copy()
    quality = pd.Categorical(encoded["product_quality"], categories=QUALITY_LEVELS)
    quality_dummies = pd.get_dummies(
        quality,
        prefix="product_quality",
        dtype=int,
    )
    quality_dummies.index = encoded.index
    encoded = pd.concat(
        [encoded.drop(columns=["product_quality"]), quality_dummies],
        axis=1,
    )
    return encoded


def build_predictor_table(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the training feature formulas to unlabeled prediction inputs."""
    validate_predictor_input(df)
    engineered = add_domain_features(df)
    engineered = encode_product_quality(engineered)
    features = engineered[FEATURE_COLUMNS]
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("Engineered features must contain finite values.")
    return features


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a model-ready table with engineered features and target last."""
    validate_model_input(df)
    features = build_predictor_table(df)
    return features.assign(**{TARGET: df[TARGET]})


def save_features(df: pd.DataFrame, output_path: Path = FEATURE_OUTPUT_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def save_feature_dictionary(df: pd.DataFrame) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for column in df.columns:
        rows.append(
            {
                "column": column,
                "role": "target" if column == TARGET else "feature",
                "description": FEATURE_DESCRIPTIONS[column],
            }
        )
    pd.DataFrame(rows).to_csv(FEATURE_DICTIONARY_PATH, index=False)


def main() -> None:
    model_input = load_model_input()
    feature_table = build_feature_table(model_input)
    save_features(feature_table)
    save_feature_dictionary(feature_table)

    added_features = [
        column
        for column in feature_table.columns
        if column not in BASE_NUMERIC_FEATURES and column != TARGET
    ]
    print("Feature engineering completed successfully")
    print(f"Input path: {MODEL_INPUT_PATH}")
    print(f"Output path: {FEATURE_OUTPUT_PATH}")
    print(f"Shape: {feature_table.shape[0]} rows x {feature_table.shape[1]} columns")
    print("\nEngineered/encoded features:")
    for column in added_features:
        print(f"- {column}")
    print(f"\nFeature dictionary: {FEATURE_DICTIONARY_PATH}")


if __name__ == "__main__":
    main()

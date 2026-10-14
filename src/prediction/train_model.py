"""Train and evaluate one weighted baseline on the saved training/test split."""

import argparse
import json
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from feature_engineering import PROJECT_ROOT, TARGET
from split_data import RANDOM_STATE, TEST_PATH, TRAIN_PATH


MODEL_DIR = PROJECT_ROOT / "models"
TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
MODEL_NAMES = ("random_forest", "xgboost")
THRESHOLD = 0.5


def load_training_data():
    """Use only predictor columns as model inputs, with identical schemas."""
    datasets = []
    for path in (TRAIN_PATH, TEST_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run split_data.py first.")
        df = pd.read_csv(path)
        if TARGET not in df.columns or set(df[TARGET].unique()) != {0, 1}:
            raise ValueError(f"Expected both binary target classes in {path}.")
        features = df.drop(columns=[TARGET])
        if features.empty or not all(
            pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes
        ):
            raise ValueError(f"Expected numeric feature columns in {path}.")
        if not np.isfinite(features.to_numpy()).all():
            raise ValueError(f"Features must be finite and nonmissing in {path}.")
        datasets.append((features, df[TARGET]))
    (X_train, y_train), (X_test, y_test) = datasets
    if set(X_train.columns) != set(X_test.columns):
        raise ValueError("Training and test feature columns do not match.")
    return X_train, X_test[X_train.columns], y_train, y_test


def create_model(name, y_train):
    """Calculate imbalance weights from training labels only."""
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=4,
        )

    from xgboost import XGBClassifier

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=negative_count / positive_count,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=4,
    )


def evaluate_model(model, X_test, y_test):
    """Measure failure detection at a fixed threshold on the held-out set."""
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def save_results(name, model, X_train, y_train, y_test, metrics):
    """Persist the model, reproducibility details, and comparison tables."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{name}.joblib"
    joblib.dump(model, model_path)
    metadata = {
        "model": name,
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
        "feature_columns": list(X_train.columns),
        "target": TARGET,
        "threshold": THRESHOLD,
        "random_state": RANDOM_STATE,
        "train_rows": len(y_train),
        "test_rows": len(y_test),
        "train_failures": int(y_train.sum()),
        "test_failures": int(y_test.sum()),
        "parameters": {
            key: "NaN" if isinstance(value, float) and np.isnan(value) else value
            for key, value in model.get_params().items()
        },
        "versions": {
            package: version(package)
            for package in ("pandas", "numpy", "scikit-learn", "joblib")
        },
        "test_metrics": metrics,
    }
    if name == "xgboost":
        metadata["versions"]["xgboost"] = version("xgboost")
    (MODEL_DIR / f"{name}_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    pd.DataFrame(
        {"feature": X_train.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False).to_csv(
        TABLE_DIR / f"{name}_feature_importance.csv", index=False
    )

    comparison = []
    for model_name in MODEL_NAMES:
        metadata_path = MODEL_DIR / f"{model_name}_metadata.json"
        if metadata_path.exists():
            result = json.loads(metadata_path.read_text(encoding="utf-8"))
            comparison.append({"model": model_name, **result["test_metrics"]})
    pd.DataFrame(comparison).to_csv(TABLE_DIR / "model_comparison.csv", index=False)
    return model_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODEL_NAMES)
    args = parser.parse_args()
    X_train, X_test, y_train, y_test = load_training_data()
    model = create_model(args.model, y_train)
    print(f"Training {args.model} on {len(y_train):,} rows", flush=True)
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    model_path = save_results(args.model, model, X_train, y_train, y_test, metrics)
    print(f"Test results ({len(y_test):,} rows; threshold={THRESHOLD}):")
    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"):
        print(f"{metric}: {metrics[metric]:.4f}")
    print(f"Confusion matrix [[TN, FP], [FN, TP]]: "
          f"[[{metrics['tn']}, {metrics['fp']}], [{metrics['fn']}, {metrics['tp']}]]")
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()

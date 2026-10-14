"""Export a test metric comparison of the two trained models."""

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".venv" / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


MODELS = {
    "random_forest": ("Random Forest (selected)", "#487B91"),
    "xgboost": ("XGBoost", "#C44848"),
}
METRICS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-score",
    "roc_auc": "ROC-AUC",
    "average_precision": "Average precision",
}
OUTPUT_PATH = PROJECT_ROOT / "reports" / "figures" / "model_comparison.png"


def main():
    results = {}
    for name in MODELS:
        path = PROJECT_ROOT / "models" / f"{name}_metadata.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Train {name} first.")
        results[name] = json.loads(path.read_text(encoding="utf-8"))
    first, second = results.values()
    for field in ("test_rows", "test_failures", "threshold"):
        if first[field] != second[field]:
            raise ValueError(f"Model evaluations differ in {field}.")

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.20, right=0.95, top=0.75, bottom=0.19)
    fig.text(0.06, 0.94, "Random Forest vs. XGBoost", fontsize=28, weight="bold", va="top")
    fig.text(
        0.06, 0.875,
        f"Held-out test set: {first['test_rows']:,} cases, {first['test_failures']} failures"
        f" | Classification threshold: {first['threshold']:.2f}",
        fontsize=16, color="#4B5563", va="top",
    )
    positions = np.arange(len(METRICS))
    for offset, (name, (label, color)) in zip((-0.18, 0.18), MODELS.items()):
        scores = [results[name]["test_metrics"][metric] for metric in METRICS]
        if not all(np.isfinite(score) and 0 <= score <= 1 for score in scores):
            raise ValueError(f"Invalid test metrics for {name}.")
        bars = ax.barh(positions + offset, scores, height=0.32, label=label, color=color)
        for metric, bar, score in zip(METRICS, bars, scores):
            text = f"{score:.4f}" if metric in ("roc_auc", "average_precision") else f"{score:.2%}"
            ax.text(
                score + 0.012, bar.get_y() + bar.get_height() / 2, text,
                va="center", fontsize=14, color="#252525",
            )
    ax.set_yticks(positions, list(METRICS.values()))
    ax.invert_yaxis()
    ax.set_xlim(0, 1.17)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.set_xlabel("Score (0 to 1; higher is better)", fontsize=15, labelpad=12)
    ax.tick_params(axis="both", labelsize=15, length=0, pad=10)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#E5E7EB")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(
        loc="lower left", bbox_to_anchor=(0, 1.035), ncol=2,
        frameon=False, fontsize=16, borderaxespad=0,
    )
    fig.text(
        0.06, 0.075,
        f"Random Forest: {first['test_metrics']['fn']} missed failures, "
        f"{first['test_metrics']['fp']} false alarms."
        f"    XGBoost: {second['test_metrics']['fn']} missed failures, "
        f"{second['test_metrics']['fp']} false alarms.",
        fontsize=14, color="#252525",
    )
    fig.text(
        0.06, 0.035,
        "Precision, recall, F1 and average precision refer to the failure class."
        " Source: saved test evaluation results.",
        fontsize=12, color="#60666B",
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

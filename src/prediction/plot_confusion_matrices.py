"""Export labeled test confusion matrices from saved model evaluation results."""

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".venv" / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


MODEL_DIR = PROJECT_ROOT / "models"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
MODEL_TITLES = {"random_forest": "Random Forest", "xgboost": "XGBoost"}


def plot_confusion_matrix(name, title):
    metadata_path = MODEL_DIR / f"{name}_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing {metadata_path}. Train {name} first.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metrics = metadata["test_metrics"]
    counts = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    if (counts < 0).any() or counts.sum() != metadata["test_rows"]:
        raise ValueError(f"Invalid confusion counts for {name}.")
    if counts[1].sum() != metadata["test_failures"] or (counts.sum(axis=1) == 0).any():
        raise ValueError(f"Invalid class totals for {name}.")
    row_rates = counts / counts.sum(axis=1, keepdims=True)
    if not np.isclose(row_rates[1, 1], metrics["recall"]):
        raise ValueError(f"Recall does not match confusion counts for {name}.")

    labels = np.array([["TN", "FP"], ["FN", "TP"]])
    annotations = np.array([
        [f"{labels[i, j]}\n{counts[i, j]:,}\n{row_rates[i, j]:.2%}" for j in range(2)]
        for i in range(2)
    ])
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.subplots_adjust(left=0.18, right=0.94, bottom=0.24, top=0.80)
    fig.text(0.08, 0.94, f"{title} confusion matrix", fontsize=21, weight="bold", va="top")
    fig.text(
        0.08, 0.875,
        f"{metadata['test_rows']:,} test cases | Threshold {metadata['threshold']:.2f}",
        fontsize=13, color="#4B5563", va="top",
    )
    sns.heatmap(
        row_rates, annot=annotations, fmt="", cmap="Blues", vmin=0, vmax=1,
        cbar=False, square=True, linewidths=3, linecolor="white",
        xticklabels=["No failure (0)", "Failure (1)"],
        yticklabels=["No failure (0)", "Failure (1)"],
        annot_kws={"fontsize": 17}, ax=ax,
    )
    ax.set_xlabel("Predicted class", fontsize=14, labelpad=12)
    ax.set_ylabel("Actual class", fontsize=14, labelpad=12)
    ax.tick_params(axis="both", labelsize=12, length=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.text(
        0.08, 0.11,
        f"Failure recall = TP / (TP + FN) = {counts[1, 1]} / {counts[1].sum()}"
        f" = {metrics['recall']:.2%}",
        fontsize=13, weight="bold",
    )
    fig.text(
        0.08, 0.065, "Counts shown in each cell; percentages and colors are within each actual class.",
        fontsize=10, color="#4B5563",
    )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURE_DIR / f"{name}_confusion_matrix.png"
    fig.savefig(output_path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    sns.set_theme(style="white", font="DejaVu Sans")
    for name, title in MODEL_TITLES.items():
        plot_confusion_matrix(name, title)


if __name__ == "__main__":
    main()

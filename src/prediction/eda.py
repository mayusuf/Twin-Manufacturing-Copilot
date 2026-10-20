"""Create four reproducible EDA figures and supporting slide notes."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".venv" / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, StrMethodFormatter
import pandas as pd
import seaborn as sns


DATA_PATH = PROJECT_ROOT / "data" / "processed" / "model_input.csv"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
TABLE_DIR = REPORT_DIR / "tables"
TARGET = "machine_failure"
LABELS = {0: "No failure", 1: "Failure"}
COLORS = {"No failure": "#487B91", "Failure": "#C44848"}
FEATURES = {
    "air_temperature_k": "Air temperature (K)",
    "process_temperature_k": "Process temperature (K)",
    "rotational_speed_rpm": "Rotational speed (rpm)",
    "torque_nm": "Torque (Nm)",
    "tool_wear_min": "Tool wear (min)",
}


def load_data():
    df = pd.read_csv(DATA_PATH)
    required = ["product_quality", *FEATURES, TARGET]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df.empty or df[required].isna().any().any():
        raise ValueError("EDA requires nonempty data without missing values.")
    if set(df[TARGET].unique()) != {0, 1}:
        raise ValueError("Expected both binary target classes: 0 and 1.")
    if not df["product_quality"].isin(["L", "M", "H"]).all():
        raise ValueError("Expected product quality levels L, M, or H.")
    if any(not pd.api.types.is_numeric_dtype(df[f]) for f in FEATURES):
        raise ValueError("Numeric feature columns must contain numbers.")
    return df.assign(failure_status=df[TARGET].map(LABELS))


def new_figure(title, subtitle):
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.10, right=0.96, bottom=0.16, top=0.79)
    add_heading(fig, title, subtitle)
    return fig, ax


def add_heading(fig, title, subtitle):
    fig.text(0.06, 0.94, title, fontsize=28, weight="bold", va="top")
    fig.text(0.06, 0.885, subtitle, fontsize=16, color="#4B5563", va="top")
    fig.text(
        0.06, 0.035,
        "Source: UCI AI4I 2020 | Full processed dataset | Synthetic manufacturing data",
        fontsize=12, color="#60666B",
    )


def save_figure(fig, filename):
    # Keep a fixed 16:9 canvas for slide placement; avoid tight bounding-box crops.
    for extension in ("png", "pdf"):
        fig.savefig(FIGURE_DIR / f"{filename}.{extension}", dpi=200, facecolor="white")
    plt.close(fig)


def plot_class_distribution(df, counts):
    failure_pct = counts.loc[1] / len(df) * 100
    fig, ax = new_figure(
        "Machine failure class distribution",
        f"{len(df):,} observations | Failures represent {failure_pct:.2f}% of the dataset",
    )
    bars = ax.bar([LABELS[c] for c in counts.index], counts.values,
                  color=list(COLORS.values()), width=0.50)
    for bar, count in zip(bars, counts.values):
        ax.annotate(f"{count:,}\n{count / len(df) * 100:.2f}%",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 12), textcoords="offset points", ha="center",
                    fontsize=20, weight="bold")
    ax.set_ylim(0, counts.max() * 1.22)
    ax.set_ylabel("Number of observations")
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax)
    save_figure(fig, "01_target_class_distribution")


def plot_quality_failure_rates(df, rates):
    baseline = df[TARGET].mean() * 100
    fig, ax = new_figure(
        "Failure rate by product quality",
        "L: low | M: medium | H: high | Rate = failures / observations within each level",
    )
    bars = ax.bar(rates.index, rates["failure_rate_pct"],
                  color=["#C44848", "#487B91", "#53846A"], width=0.55)
    for bar, row in zip(bars, rates.itertuples()):
        ax.annotate(f"{row.failure_rate_pct:.2f}%\n{row.failures:,} / {row.observations:,}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 12), textcoords="offset points", ha="center",
                    fontsize=18, weight="bold")
    ax.axhline(baseline, color="#656565", linestyle="--", linewidth=1.6,
               label=f"Overall rate: {baseline:.2f}%")
    ax.set_ylim(0, rates["failure_rate_pct"].max() * 1.42)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.set_ylabel("Failure rate")
    ax.set_xlabel("Product quality")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", frameon=False, fontsize=15)
    sns.despine(ax=ax)
    save_figure(fig, "02_failure_rate_by_product_quality")


def plot_feature_boxplots(df, counts):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.subplots_adjust(left=0.075, right=0.96, bottom=0.13, top=0.78,
                        hspace=0.55, wspace=0.37)
    add_heading(
        fig, "Numeric features by failure status",
        f"No failure: n={counts.loc[0]:,} | Failure: n={counts.loc[1]:,} | All outliers retained",
    )
    for ax, (feature, label) in zip(axes.flat, FEATURES.items()):
        sns.boxplot(data=df, x="failure_status", y=feature,
                    order=list(COLORS), hue="failure_status", hue_order=list(COLORS),
                    palette=COLORS, legend=False, width=0.55, linewidth=1.3,
                    fliersize=2.5, ax=ax)
        ax.set_title(label, fontsize=16, weight="bold", pad=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelsize=13)
        ax.grid(axis="x", visible=False)
        sns.despine(ax=ax)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(
        0.04, 0.78,
        "Box: middle 50%\nLine: median\nWhiskers: within 1.5 x IQR\nDots: values beyond whiskers",
        transform=axes.flat[-1].transAxes, fontsize=15, va="top", linespacing=1.7,
        color="#4B5563",
    )
    save_figure(fig, "03_numeric_features_by_failure")


def plot_torque_vs_rpm(df, correlation):
    fig, ax = new_figure(
        "Torque vs. rotational speed",
        f"All {len(df):,} observations | Pearson correlation: {correlation:.2f}",
    )
    for status, color in COLORS.items():
        subset = df.loc[df["failure_status"] == status]
        ax.scatter(subset["rotational_speed_rpm"], subset["torque_nm"],
                   color=color, s=18 if status == "No failure" else 38,
                   alpha=0.18 if status == "No failure" else 0.85,
                   edgecolors="none", label=f"{status} (n={len(subset):,})", rasterized=True)
    ax.set_xlabel("Rotational speed (rpm)")
    ax.set_ylabel("Torque (Nm)")
    ax.legend(loc="upper right", frameon=True, facecolor="white", fontsize=16)
    sns.despine(ax=ax)
    save_figure(fig, "04_torque_vs_rpm_by_failure")


def save_notes(df, counts, rates, medians, correlation):
    highest = rates["failure_rate_pct"].idxmax()
    notes = f"""# EDA slide takeaways

Input: `data/processed/model_input.csv` ({len(df):,} observations).
Source: UCI AI4I 2020 Predictive Maintenance Dataset, DOI `10.24432/C5HS5C`.
The dataset is synthetic. These are descriptive associations in the full dataset.

## 1. Machine failure class distribution

Failures: {counts.loc[1]:,} ({counts.loc[1] / len(df) * 100:.2f}%).
No failure: {counts.loc[0]:,} ({counts.loc[0] / len(df) * 100:.2f}%).

Slide takeaway: Strong class imbalance. A model that always predicts no failure
would have {counts.loc[0] / len(df) * 100:.2f}% accuracy and zero failure recall.
Use a stratified split and compare failure recall, precision, F1, and PR-AUC.

## 2. Failure rate by product quality

L = low, M = medium, H = high product quality.

"""
    for product_quality, row in rates.iterrows():
        notes += (f"- Quality {product_quality}: {row['failure_rate_pct']:.2f}% "
                  f"({int(row['failures']):,} failures / {int(row['observations']):,} observations).\n")
    notes += f"""
Slide takeaway: Quality {highest} has the highest observed failure rate.
Keep product quality as a categorical predictor. The feature builder uses a fixed,
predefined L/M/H one-hot vocabulary; it learns no encoding statistics from data.

## 3. Numeric features by failure status

Medians (failure vs. no failure):

"""
    for feature, label in FEATURES.items():
        notes += f"- {label}: {medians.loc[1, feature]:.1f} vs. {medians.loc[0, feature]:.1f}.\n"
    notes += f"""
Slide takeaway: Failure observations have higher median torque and tool wear,
and lower median rotational speed. The distributions still overlap.
Extreme values are retained because they may describe failure conditions.

## 4. Torque vs. rotational speed

Pearson correlation across all observations: {correlation:.3f}.

Slide takeaway: Torque and rotational speed have a strong inverse association.
Failure points also occur across multiple operating regions. Investigate their
interaction during feature engineering; this chart does not establish causation.

## Output and modeling notes

- Figures are 3200 x 1800 PNGs (16:9) with matching PDF exports.
- Counts, rates, and feature statistics are saved in `reports/tables`.
- EDA used the full dataset. Freeze modeling choices before evaluating the test set.
- The fixed L/M/H encoding is shared by training and twin inference.
- Fit any learned transformations or resampling using training data only.
"""
    (REPORT_DIR / "eda_slide_notes.md").write_text(notes, encoding="utf-8")


def main():
    df = load_data()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font="DejaVu Sans", font_scale=1.3,
                  rc={"axes.spines.top": False, "axes.spines.right": False,
                      "grid.color": "#E5E7EB", "axes.labelcolor": "#30363B"})

    counts = df[TARGET].value_counts().reindex([0, 1])
    distribution = pd.DataFrame({"failure_status": [LABELS[c] for c in counts.index],
                                 "observations": counts.values,
                                 "percentage": counts.values / len(df) * 100})
    distribution.to_csv(TABLE_DIR / "target_class_distribution.csv", index=False)
    rates = df.groupby("product_quality")[TARGET].agg(observations="count", failures="sum")
    rates = rates.reindex(["L", "M", "H"])
    rates["failure_rate_pct"] = rates["failures"] / rates["observations"] * 100
    rates.to_csv(TABLE_DIR / "failure_rate_by_product_quality.csv")
    summary = df.groupby("failure_status")[list(FEATURES)].agg(
        ["count", "mean", "median", "std", "min", "max"])
    summary.columns = [f"{feature}_{stat}" for feature, stat in summary.columns]
    summary.to_csv(TABLE_DIR / "feature_summary_by_failure.csv")
    medians = df.groupby(TARGET)[list(FEATURES)].median()
    correlation = df["torque_nm"].corr(df["rotational_speed_rpm"])

    plot_class_distribution(df, counts)
    plot_quality_failure_rates(df, rates)
    plot_feature_boxplots(df, counts)
    plot_torque_vs_rpm(df, correlation)
    save_notes(df, counts, rates, medians, correlation)

    print(f"EDA completed: {len(df):,} observations")
    print(distribution.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\nFailure rate by product quality:")
    print(rates.to_string(float_format=lambda x: f"{x:.2f}"))
    print(f"\nTorque/RPM correlation: {correlation:.3f}")
    print(f"Figures: {FIGURE_DIR} (4 PNGs and 4 PDFs)")
    print(f"Tables: {TABLE_DIR}")
    print(f"Slide takeaways: {REPORT_DIR / 'eda_slide_notes.md'}")


if __name__ == "__main__":
    main()

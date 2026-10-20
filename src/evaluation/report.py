"""Create a presentation table and chart from recorded live Copilot trials."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

from src.evaluation.build_suite import CATEGORIES


LABELS = {
    "current_state": "Current-state explanation",
    "what_if": "What-if simulation",
    "history": "History and follow-ups",
    "model_metrics": "Model performance questions",
    "ambiguous_invalid": "Ambiguous or invalid requests",
    "unsupported_claims": "Unsupported requests",
}


def report_rows(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["category"]].append(record)
    rows = []
    for category in CATEGORIES:
        turns = grouped.get(category, [])
        if not turns:
            continue
        trials = defaultdict(list)
        for turn in turns:
            trials[(turn["case_id"], turn["trial"])].append(turn)
        completed_turns = [turn for turn in turns if turn["error"] is None]
        passed = sum(all(turn["automatic_checks_pass"] for turn in trial) for trial in trials.values())
        rows.append({
            "category": LABELS[category], "unique_tasks": len({key[0] for key in trials}),
            "case_trials": len(trials), "passed_automatic_checks": passed,
            "automatic_pass_percent": round(100 * passed / len(trials), 2),
            "execution_errors": sum(turn["error"] is not None for turn in turns),
            "mean_completed_turn_seconds": round(statistics.mean(turn["latency_seconds"] for turn in completed_turns), 2) if completed_turns else None,
            "mean_completed_turn_tool_calls": round(statistics.mean(turn["tool_calls"] for turn in completed_turns), 2) if completed_turns else None,
            "human_evaluated_task_success": "Pending human review",
        })
    return rows


def generate_report(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    records = [json.loads(line) for line in (directory / "trials.jsonl").read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError("No trial records found.")
    rows = report_rows(records)
    planned = len(manifest["case_ids"]) * manifest["trials"]
    actual = {(record["case_id"], record["trial"]) for record in records}
    passed = sum(row["passed_automatic_checks"] for row in rows)
    errors = sum(row["execution_errors"] for row in rows)
    all_calls_failed = all(record["error"] for record in records)
    status = "CONNECTION/EXECUTION FAILURE — no completed answers" if all_calls_failed else ("Complete trial recording" if len(actual) == planned else "Partial trial recording")
    with (directory / "results_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table = [
        "| Category | Trials | Passed automatic checks | Pass rate | Errors | Mean answer time |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        latency = f"{row['mean_completed_turn_seconds']:.2f} s" if row["mean_completed_turn_seconds"] is not None else "Unavailable"
        table.append(f"| {row['category']} | {row['case_trials']} | {row['passed_automatic_checks']} | {row['automatic_pass_percent']:.2f}% | {row['execution_errors']} | {latency} |")
    report = [
        "# Copilot evaluation results", "", f"**Status: {status}.**", "",
        f"Model requested: `{manifest['model_requested']}`. Split: **{manifest['split']}**. Started: {manifest['started_at_utc']}.",
        f"Recorded **{len(actual)} / {planned} case-trials**, **{len(records)} conversation turns**, and **{errors} execution errors**.", "",
        "These rates measure tool execution, arguments, fresh state, and baseline preservation. They do not establish answer correctness, explanation quality, or absence of unsupported claims.", "",
        *table, "",
        f"Overall automatic check rate: **{passed}/{len(actual)} ({100 * passed / len(actual):.2f}%)**.", "",
        "**Human-evaluated task success: pending.** Have reviewers A and B independently complete `human_reviews.csv`, then document any adjudication. Do not report the automatic rate as overall Copilot accuracy.", "",
        "Errors count as failed trials; connection failures describe the test setup and do not establish the Copilot's reasoning quality. Mean answer time includes only completed turns and excludes agent initialization. The execution-error count is per recorded turn; a failed warm-up ends its task-trial.", "",
        "This report covers only the selected split and tasks. A development smoke test is not the 60-task final benchmark. Full research results should additionally report repeated-trial reliability and uncertainty after human grading.", "",
        "![Automatic tool and state checks](results_chart.png)", "",
        "The chart is generated from recorded trials. Full evidence is in `trials.jsonl`; run configuration is in `manifest.json`.", "",
    ]
    (directory / "results_report.md").write_text("\n".join(report))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    figure, axis = plt.subplots(figsize=(11, max(3.8, .65 * len(rows) + 1.8)))
    values = [row["automatic_pass_percent"] for row in rows]
    axis.barh([row["category"] for row in rows], values, color="#177E89", height=.55)
    axis.invert_yaxis()
    axis.set_xlim(0, 115)
    axis.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    axis.set_xlabel("Case-trials passing automatic tool and state checks")
    for index, row in enumerate(rows):
        axis.text(row["automatic_pass_percent"] + 1.5, index, f"{row['automatic_pass_percent']:.1f}% ({row['passed_automatic_checks']}/{row['case_trials']})", va="center", fontsize=10)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.set_axisbelow(True)
    axis.grid(axis="x", alpha=.18)
    figure.suptitle("Copilot execution checks" if not all_calls_failed else "Evaluation blocked: no completed answers", fontweight="bold", fontsize=16, x=.02, ha="left")
    figure.text(.02, .91, f"{manifest['split'].capitalize()} split · {len(actual)} case-trials · answer quality awaits human review", fontsize=10, color="#555555")
    figure.tight_layout(rect=(0, 0, 1, .87))
    figure.savefig(directory / "results_chart.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return directory / "results_report.md"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_directory", type=Path)
    args = parser.parse_args()
    try:
        print(generate_report(args.results_directory))
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

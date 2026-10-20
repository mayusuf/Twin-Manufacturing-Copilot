"""Record real Copilot trials, objective tool checks, and two independent review forms."""

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import statistics
import time

from src.copilot import CopilotAgent
from src.copilot.agent import DEFAULT_LLM_MODEL
from src.digital_twin import DigitalTwin, MachineState
from src.prediction.feature_engineering import PROJECT_ROOT
from src.evaluation.build_suite import SUITE_DIR, sha256, stable_snapshot


def equivalent(actual, expected, tolerance=1e-8):
    """Compare reference values with numeric tolerance, excluding runtime timestamps."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(key in actual and equivalent(actual[key], value, tolerance) for key, value in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(equivalent(a, e, tolerance) for a, e in zip(actual, expected))
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return isinstance(actual, (int, float)) and not isinstance(actual, bool) and math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance)
    return actual == expected


def normalized_inputs(evidence):
    inputs = {key: value for key, value in evidence["inputs"].items() if value is not None}
    if evidence["tool"] == "get_model_results":
        inputs.setdefault("model_name", "random_forest")
    return inputs


def check_reply(reply, expected, baseline, twin):
    """Necessary objective checks only; answer correctness remains a human review."""
    successful = [item for item in reply.evidence if item["status"] == "success"]
    checks = {
        "nonempty_answer": bool(reply.text.strip()),
        "baseline_preserved": twin.snapshot == baseline,
        "fresh_context_matches": equivalent(reply.context["current_snapshot"], expected["current_snapshot"]),
    }
    checks["required_tools_and_arguments"] = all(
        any(
            item["tool"] == required["tool"]
            and set(normalized_inputs(item)) == set(required["inputs"])
            and equivalent(normalized_inputs(item), required["inputs"])
            for item in successful
        ) for required in expected["required_tools"]
    )
    simulations = [item for item in successful if item["tool"] == "simulate_scenario"]
    if expected["allow_successful_simulation"]:
        required = next(item for item in expected["required_tools"] if item["tool"] == "simulate_scenario")
        checks["no_unrequested_simulations"] = len(simulations) == 1 and all(
            set(normalized_inputs(item)) == set(required["inputs"])
            and equivalent(normalized_inputs(item), required["inputs"]) for item in simulations
        )
        checks["simulation_matches_reference"] = len(reply.simulations) == 1 and equivalent(
            reply.simulations[0].to_dict(), expected["simulation"]
        )
    else:
        checks["no_unrequested_simulations"] = not simulations and not reply.simulations
    return checks


def validate_suite(path):
    suite = json.loads(path.read_text())
    if suite.get("schema_version") != 1:
        raise ValueError("Unsupported suite schema.")
    checksum_path = path.with_name("scenarios.sha256")
    if not checksum_path.exists() or checksum_path.read_text().split()[0] != sha256(path):
        raise ValueError("Suite checksum missing or changed; preserve the frozen suite.")
    # Code changes are allowed (the point of subsequent evaluations); record their hashes.
    # Changed model/data references would invalidate the numerical answer key.
    for relative, expected_hash in suite["provenance"]["artifacts_sha256"].items():
        if relative.startswith(("models/", "data/")) and sha256(PROJECT_ROOT / relative) != expected_hash:
            raise ValueError(f"Reference artifact changed: {relative}. Create a new suite version.")
    return suite


def check_references(cases):
    """Offline preflight: recompute the answer key without constructing any LLM client."""
    twin = DigitalTwin()
    for case in cases:
        twin.UpdateState(**case["initial_state"])
        for turn in case["turns"]:
            if turn["before_update"]:
                twin.UpdateState(**turn["before_update"])
            if not equivalent(stable_snapshot(twin.snapshot), turn["expected"]["current_snapshot"]):
                raise ValueError(f"Reference snapshot mismatch: {case['id']}")
            if "simulation" in turn["expected"]:
                changes = next(item["inputs"] for item in turn["expected"]["required_tools"] if item["tool"] == "simulate_scenario")
                if not equivalent(twin.Simulate(**changes).to_dict(), turn["expected"]["simulation"]):
                    raise ValueError(f"Reference simulation mismatch: {case['id']}")


REVIEW_FIELDS = [
    "case_id", "category", "trial", "turn", "reviewer", "question", "answer", "error",
    "reference", "review_criteria", "automatic_checks",
    "answer_correct_0_or_1", "appropriate_limitations_0_or_1", "unsupported_claim_present_0_or_1",
    "explanation_quality_1_to_5", "notes",
]


def run_trial(case, trial, model_name, agent_factory=None):
    """Every independent trial gets a new twin and conversation; history persists within it."""
    twin = DigitalTwin(MachineState(**case["initial_state"]))
    agent = (agent_factory or (lambda machine: CopilotAgent(machine, model_name=model_name)))(twin)
    history = [twin.snapshot]
    latest_simulation = None
    records = []
    for index, turn in enumerate(case["turns"], 1):
        if turn["before_update"]:
            history.append(twin.UpdateState(**turn["before_update"]))
            latest_simulation = None
        baseline = twin.snapshot
        start = time.perf_counter()
        try:
            reply = agent.ask(turn["prompt"], history=history, latest_simulation=latest_simulation)
            checks = check_reply(reply, turn["expected"], baseline, twin)
            response = reply.to_dict()
            if reply.simulations:
                latest_simulation = reply.simulations[-1]
            error = None
        except Exception as exception:
            # Do not include arbitrary provider errors, which could contain credentials or request data.
            error = type(exception).__name__
            checks = {"completed": False, "baseline_preserved": twin.snapshot == baseline}
            response = None
        records.append({
            "case_id": case["id"], "category": case["category"], "trial": trial, "turn": index,
            "question": turn["prompt"], "response": response, "error": error,
            "latency_seconds": time.perf_counter() - start,
            "tool_calls": len(response["evidence"]) if response else None,
            "automatic_checks": checks, "automatic_checks_pass": all(checks.values()),
            "task_success": None, "review_status": "pending_two_human_reviews",
            "expected": turn["expected"],
        })
        if error:
            # A failed warm-up is a failed task; later answers cannot conceal it.
            break
    return records


def summarize(records, expected_trials):
    by_trial = defaultdict(list)
    for record in records:
        by_trial[(record["case_id"], record["trial"])].append(record)
    outcomes = {key: all(item["automatic_checks_pass"] for item in items) for key, items in by_trial.items()}
    categories = defaultdict(list)
    for key, passed in outcomes.items():
        categories[by_trial[key][0]["category"]].append(passed)
    per_case = defaultdict(list)
    for (case_id, _), passed in outcomes.items():
        per_case[case_id].append(passed)
    successful_latencies = [record["latency_seconds"] for record in records if not record["error"]]
    return {
        "case_trials": len(outcomes), "recorded_turns": len(records),
        "automatic_checks_pass_rate": sum(outcomes.values()) / len(outcomes) if outcomes else None,
        "automatic_checks_pass_rate_by_category": {key: sum(values) / len(values) for key, values in categories.items()},
        "cases_passing_automatic_checks_in_all_trials": sum(len(values) == expected_trials and all(values) for values in per_case.values()),
        "cases": len(per_case), "errors": sum(bool(record["error"]) for record in records),
        "mean_successful_turn_latency_seconds": statistics.mean(successful_latencies) if successful_latencies else None,
        "mean_completed_turn_tool_calls": statistics.mean([record["tool_calls"] for record in records if record["tool_calls"] is not None]) if successful_latencies else None,
        "task_success_rate": None,
        "note": "Automatic checks cover execution and tool evidence only. Numerical claims in prose, unsupported claims, and task success require the two human reviews. Errors count as failed trials; failed calls may have unrecorded tool activity.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=SUITE_DIR / "scenarios.json")
    parser.add_argument("--split", choices=("development", "evaluation"), default="development")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Verify frozen references without API calls or credentials.")
    args = parser.parse_args()
    if args.trials < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Trials and limit must be positive.")
    try:
        suite = validate_suite(args.suite)
        cases = [case for case in suite["cases"] if case["split"] == args.split]
        if args.limit:
            cases = cases[:args.limit]
        if not cases:
            raise ValueError("No scenarios selected.")
        check_references(cases)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if args.dry_run:
        print(f"Verified {len(cases)} {args.split} cases and their references. Planned: {len(cases) * args.trials} case-trials; no API calls made.")
        return
    if not os.getenv("OPENAI_API_KEY", "").strip():
        parser.error("Set OPENAI_API_KEY in your environment. The runner does not read Streamlit secrets.")
    output = args.output or PROJECT_ROOT / "reports" / "copilot_evaluation" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("Output directory must be empty to avoid overwriting earlier trials.")
    manifest = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite_sha256": sha256(args.suite), "split": args.split,
        "model_requested": args.model, "trials": args.trials, "case_ids": [case["id"] for case in cases],
        "runtime_artifacts_sha256": {path: sha256(PROJECT_ROOT / path) for path in suite["provenance"]["artifacts_sha256"]},
        "runner_sha256": sha256(__file__),
        "packages": {name: version(name) for name in ("langgraph", "langchain-openai", "openai", "scikit-learn", "pandas")},
        "inference_settings": {"timeout_seconds": 30, "max_retries": 1, "max_tokens": 2000, "temperature": "provider default"},
        "note": "Frozen reference answers and selection labels are never supplied to the LLM. No token-cost accounting or exact provider model revision is captured.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    records = []
    # Keep partial results and review forms if a long evaluation is interrupted.
    with (output / "trials.jsonl").open("w") as trace, (output / "human_reviews.csv").open("w", newline="") as reviews:
        writer = csv.DictWriter(reviews, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for case in cases:
            for trial in range(1, args.trials + 1):
                batch = run_trial(case, trial, args.model)
                for record in batch:
                    records.append(record)
                    trace.write(json.dumps(record, allow_nan=False) + "\n")
                    for reviewer in ("A", "B"):
                        writer.writerow({
                            "case_id": case["id"], "category": case["category"], "trial": trial,
                            "turn": record["turn"], "reviewer": reviewer, "question": record["question"],
                            "answer": record["response"]["text"] if record["response"] else "",
                            "error": record["error"], "reference": json.dumps(record["expected"]),
                            "review_criteria": " | ".join(record["expected"]["review_criteria"]),
                            "automatic_checks": json.dumps(record["automatic_checks"]),
                        })
                trace.flush()
                reviews.flush()
                (output / "summary.json").write_text(json.dumps(summarize(records, args.trials), indent=2) + "\n")
                print(f"{case['id']} trial {trial}/{args.trials}: recorded; human review pending", flush=True)
    print(f"Saved transcripts, objective checks, and review forms to {output}")


if __name__ == "__main__":
    main()

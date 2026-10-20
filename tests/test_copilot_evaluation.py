"""Protect held-out separation, numerical references, and honest evaluation reporting."""

from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

from langchain_core.messages import AIMessage

from src.copilot import CopilotAgent
from src.digital_twin import DigitalTwin, MachineState
from src.evaluation.build_suite import CATEGORIES, SUITE_DIR, stable_snapshot
from src.evaluation.run import check_references, check_reply, run_trial, summarize, validate_suite
from src.evaluation.report import generate_report, report_rows
from test_copilot import ScriptedModel, request_tool


class CopilotEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suite = validate_suite(SUITE_DIR / "scenarios.json")

    def test_frozen_cases_have_disjoint_source_states_and_balanced_categories(self):
        cases = self.suite["cases"]
        self.assertEqual(len(cases), 72)
        self.assertEqual(len({case["id"] for case in cases}), 72)
        self.assertEqual(len({case["source"]["test_row_index_zero_based"] for case in cases}), 72)
        self.assertEqual(len({json.dumps(case["initial_state"], sort_keys=True) for case in cases}), 72)
        for split, count in (("development", 2), ("evaluation", 10)):
            self.assertEqual(Counter(case["category"] for case in cases if case["split"] == split), {category: count for category in CATEGORIES})

    def test_all_frozen_references_match_actual_twin(self):
        check_references(self.suite["cases"])

    def test_wrong_simulation_arguments_fail_even_when_tool_executes_successfully(self):
        case = next(case for case in self.suite["cases"] if case["category"] == "what_if")
        twin = DigitalTwin(MachineState(**case["initial_state"]))
        baseline = twin.snapshot
        model = ScriptedModel([
            request_tool("simulate_scenario", {"tool_wear_min": 50.0}),
            AIMessage(content="Simulation complete."),
        ])
        reply = CopilotAgent(twin, model=model).ask(case["turns"][0]["prompt"])
        checks = check_reply(reply, case["turns"][0]["expected"], baseline, twin)
        self.assertFalse(checks["required_tools_and_arguments"])
        self.assertFalse(checks["simulation_matches_reference"])
        self.assertTrue(checks["baseline_preserved"])

    def test_current_state_needs_no_tool_and_prose_is_never_automatically_certified(self):
        case = next(case for case in self.suite["cases"] if case["category"] == "current_state")
        agents = []
        def factory(twin):
            agent = CopilotAgent(twin, model=ScriptedModel([AIMessage(content="An unsupported statement needing human review.")]))
            agents.append(agent)
            return agent
        first = run_trial(case, 1, "scripted-test", agent_factory=factory)
        second = run_trial(case, 2, "scripted-test", agent_factory=factory)
        self.assertIsNot(agents[0].twin, agents[1].twin)
        self.assertEqual(len(agents[0]._turns), 1)
        self.assertTrue(first[0]["automatic_checks_pass"])
        self.assertIsNone(first[0]["task_success"])
        summary = summarize(first + second, 2)
        self.assertIsNone(summary["task_success_rate"])
        self.assertEqual(summary["case_trials"], 2)

    def test_modified_frozen_questions_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scenarios.json"
            modified = json.loads(json.dumps(self.suite))
            modified["cases"][0]["turns"][0]["prompt"] = "Changed question"
            path.write_text(json.dumps(modified))
            path.with_name("scenarios.sha256").write_text((SUITE_DIR / "scenarios.sha256").read_text())
            with self.assertRaisesRegex(ValueError, "checksum"):
                validate_suite(path)

    def test_report_counts_failed_history_task_once_and_keeps_human_results_pending(self):
        records = [
            {"case_id": "history-1", "trial": 1, "category": "history", "automatic_checks_pass": True, "error": None, "latency_seconds": 2.0, "tool_calls": 0},
            {"case_id": "history-1", "trial": 1, "category": "history", "automatic_checks_pass": False, "error": None, "latency_seconds": 4.0, "tool_calls": 1},
            {"case_id": "history-1", "trial": 2, "category": "history", "automatic_checks_pass": False, "error": "CopilotError", "latency_seconds": 10.0, "tool_calls": None},
        ]
        row = report_rows(records)[0]
        self.assertEqual(row["case_trials"], 2)
        self.assertEqual(row["passed_automatic_checks"], 0)
        self.assertEqual(row["execution_errors"], 1)
        self.assertEqual(row["mean_completed_turn_seconds"], 3.0)
        self.assertEqual(row["human_evaluated_task_success"], "Pending human review")

    def test_connection_failure_report_does_not_claim_final_benchmark_results(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "manifest.json").write_text(json.dumps({"case_ids": ["dev-1"], "trials": 1, "model_requested": "test-model", "split": "development", "started_at_utc": "2026-10-17T00:00:00Z"}))
            (path / "trials.jsonl").write_text(json.dumps({"case_id": "dev-1", "trial": 1, "category": "current_state", "automatic_checks_pass": False, "error": "CopilotError", "latency_seconds": 1, "tool_calls": None}) + "\n")
            report = generate_report(path)
            self.assertIn("no completed answers", report.read_text())
            self.assertIn("Human-evaluated task success: pending", report.read_text())
            self.assertTrue((path / "results_chart.png").exists())


if __name__ == "__main__":
    unittest.main()

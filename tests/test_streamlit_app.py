"""Verify the user workflows and session isolation in the Streamlit app."""

from pathlib import Path
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

from test_copilot import ScriptedModel, request_tool

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class StreamlitAppTests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
        self.assertEqual(len(self.app.exception), 0)

    def submit(self, label):
        next(button for button in self.app.button if button.label == label).click().run()
        self.assertEqual(len(self.app.exception), 0)

    def predict(self):
        self.submit("Predict and update")

    def test_initial_state_waits_for_user_prediction(self):
        self.assertEqual(len(self.app.number_input), 5)
        self.assertEqual(len(self.app.selectbox), 1)
        self.assertNotIn("snapshot", self.app.session_state)
        self.assertFalse(any(button.label == "Run simulation" for button in self.app.button))

    def test_prediction_uses_entered_values(self):
        self.app.selectbox(key="current_product_quality").set_value("H")
        self.app.number_input(key="current_tool_wear_min").set_value(150.0)
        self.predict()
        snapshot = self.app.session_state["snapshot"]
        self.assertEqual(snapshot.state.product_quality, "H")
        self.assertEqual(snapshot.state.tool_wear_min, 150)
        self.assertEqual(snapshot, self.app.session_state["twin"].snapshot)
        self.assertEqual(len(self.app.metric), 2)

    def test_simulation_preserves_original_and_updates_comparison(self):
        self.predict()
        original = self.app.session_state["snapshot"]
        self.app.number_input(key="scenario_1_rotational_speed_rpm").set_value(1200.0)
        self.app.number_input(key="scenario_1_torque_nm").set_value(60.0)
        self.app.number_input(key="scenario_1_tool_wear_min").set_value(220.0)
        self.submit("Run simulation")
        comparison = self.app.session_state["simulation"]
        self.assertEqual(comparison.baseline, original)
        self.assertEqual(self.app.session_state["twin"].snapshot, original)
        self.assertEqual(self.app.session_state["snapshot"], original)
        self.assertEqual(comparison.scenario.state.torque_nm, 60)
        self.assertAlmostEqual(comparison.risk_score_change, 47.0)
        self.assertEqual(len(self.app.metric), 4)

    def test_new_baseline_clears_old_simulation_and_resets_scenario_inputs(self):
        self.predict()
        self.app.number_input(key="scenario_1_torque_nm").set_value(60.0)
        self.submit("Run simulation")
        self.app.number_input(key="current_tool_wear_min").set_value(100.0)
        self.predict()
        self.assertIsNone(self.app.session_state["simulation"])
        self.assertEqual(self.app.session_state["revision"], 2)
        self.assertEqual(self.app.number_input(key="scenario_2_tool_wear_min").value, 100)
        self.assertEqual(self.app.number_input(key="scenario_2_torque_nm").value, 42.8)

    def test_reset_clears_state_and_restores_defaults(self):
        self.app.number_input(key="current_torque_nm").set_value(60.0)
        self.predict()
        self.app.button(key="reset_session").click().run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertNotIn("snapshot", self.app.session_state)
        self.assertNotIn("twin", self.app.session_state)
        self.assertEqual(self.app.number_input(key="current_torque_nm").value, 42.8)

    def test_browser_sessions_do_not_share_machine_state(self):
        self.predict()
        other = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
        self.assertEqual(len(other.exception), 0)
        self.assertNotIn("snapshot", other.session_state)
        self.assertNotIn("twin", other.session_state)
        self.assertIn("snapshot", self.app.session_state)

    def enable_copilot(self):
        self.predict()
        self.app.text_input(key="copilot_api_key").set_value("test-only-key").run()
        self.assertEqual(len(self.app.exception), 0)

    def ask(self, question):
        self.app.chat_input(key="copilot_prompt").set_value(question).run()
        self.assertEqual(len(self.app.exception), 0)

    def test_copilot_waits_for_evaluated_state_and_connection(self):
        self.assertTrue(self.app.chat_input(key="copilot_prompt").disabled)
        self.enable_copilot()
        self.assertFalse(self.app.chat_input(key="copilot_prompt").disabled)

    def test_copilot_simulation_updates_dashboard_and_exports_evidence(self):
        self.enable_copilot()
        original = self.app.session_state["snapshot"]
        model = ScriptedModel([
            request_tool("simulate_scenario", {"torque_nm": 60, "rotational_speed_rpm": 1200, "tool_wear_min": 220}),
            AIMessage(content="The scenario raises the model score by 47 points."),
        ])
        with patch("src.copilot.agent.ChatOpenAI", return_value=model):
            self.ask("Compare 1200 rpm, 60 Nm, and 220 minutes of wear.")
        self.assertEqual(len(self.app.chat_message), 2)
        self.assertEqual(self.app.session_state["twin"].snapshot, original)
        self.assertAlmostEqual(self.app.session_state["simulation"].risk_score_change, 47)
        transcript = self.app.session_state["copilot_messages"]
        self.assertEqual(transcript[-1]["reply"]["context"]["current_snapshot"], original.to_dict())
        self.assertEqual(transcript[-1]["reply"]["evidence"][0]["status"], "success")

    def test_copilot_fresh_context_and_clear_chat_preserve_history(self):
        self.enable_copilot()
        model = ScriptedModel([AIMessage(content="First answer."), AIMessage(content="Updated answer.")])
        with patch("src.copilot.agent.ChatOpenAI", return_value=model):
            self.ask("What is the current risk?")
            self.app.number_input(key="current_tool_wear_min").set_value(100)
            self.predict()
            self.ask("And now?")
        latest = self.app.session_state["copilot_messages"][-1]["reply"]["context"]["current_snapshot"]
        self.assertEqual(latest["state"]["tool_wear_min"], 100)
        self.assertEqual(len(self.app.session_state["machine_history"]), 2)
        self.app.button(key="copilot_clear_chat").click().run()
        self.assertEqual(len(self.app.chat_message), 0)
        self.assertNotIn("copilot_agent", self.app.session_state)
        self.assertEqual(len(self.app.session_state["machine_history"]), 2)

    def test_copilot_failure_keeps_machine_and_chat_intact(self):
        self.enable_copilot()
        original = self.app.session_state["snapshot"]
        model = ScriptedModel([AIMessage(content="First answer."), RuntimeError("Disconnected")])
        with patch("src.copilot.agent.ChatOpenAI", return_value=model):
            self.ask("Current risk?")
            self.ask("And now?")
        self.assertEqual(len(self.app.session_state["copilot_messages"]), 2)
        self.assertEqual(self.app.session_state["twin"].snapshot, original)
        self.assertIsNone(self.app.session_state["simulation"])
        self.assertTrue(any("could not complete" in error.value for error in self.app.error))

    def test_reset_removes_copilot_memory_and_history(self):
        self.enable_copilot()
        model = ScriptedModel([AIMessage(content="Current score.")])
        with patch("src.copilot.agent.ChatOpenAI", return_value=model):
            self.ask("Risk?")
        self.app.button(key="reset_session").click().run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertNotIn("copilot_agent", self.app.session_state)
        self.assertNotIn("machine_history", self.app.session_state)
        self.assertEqual(self.app.session_state["copilot_messages"], [])
        self.assertEqual(self.app.text_input(key="copilot_api_key").value, "")


if __name__ == "__main__":
    unittest.main()

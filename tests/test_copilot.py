"""Exercise the real graph and twin with scripted LLM responses, without API calls."""

import json
import os
import unittest
from unittest.mock import patch

import httpx
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

from src.copilot import CopilotAgent, CopilotError
from src.copilot.agent import MAX_CONVERSATION_TURNS, MAX_TOOL_ROUNDS
from src.digital_twin import DigitalTwin, MachineState


INITIAL_STATE = MachineState("M", 298.1, 308.6, 1551, 42.8, 0)


class ScriptedModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.tools = []

    def bind_tools(self, tools, **kwargs):
        self.tools = tools
        return self

    def invoke(self, messages):
        self.requests.append(messages)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response(messages) if callable(response) else response


def request_tool(name, args=None, call_id="call_1"):
    return AIMessage(content="", tool_calls=[{
        "name": name, "args": args or {}, "id": call_id, "type": "tool_call",
    }])


class CopilotTests(unittest.TestCase):
    def setUp(self):
        self.twin = DigitalTwin(INITIAL_STATE)

    def agent(self, *responses):
        model = ScriptedModel(responses)
        return CopilotAgent(self.twin, model=model), model

    def test_existing_prediction_is_injected_without_new_inference(self):
        agent, model = self.agent(AIMessage(content="Current risk comes from the evaluated snapshot."))
        with patch.object(self.twin._model, "predict_proba", side_effect=AssertionError("Unexpected inference")):
            reply = agent.ask("What is the current risk?")
        self.assertEqual(reply.context["current_snapshot"], self.twin.snapshot.to_dict())
        self.assertIn(json.dumps(reply.context), model.requests[0][0].content)
        self.assertEqual(reply.evidence, [])
        json.dumps(reply.to_dict(), allow_nan=False)

    def test_simulation_uses_real_model_and_preserves_baseline(self):
        original = self.twin.snapshot
        def explain(messages):
            comparison = json.loads(messages[-1].content)["comparison"]
            return AIMessage(content=f"Scenario risk: {comparison['scenario']['prediction']['risk_score']}.")
        agent, model = self.agent(
            request_tool("simulate_scenario", {"rotational_speed_rpm": 1200, "torque_nm": 60, "tool_wear_min": 220}),
            explain,
        )
        reply = agent.ask("Compare 1200 rpm, 60 Nm, and 220 minutes of wear.")
        self.assertIs(self.twin.snapshot, original)
        self.assertEqual(len(reply.simulations), 1)
        self.assertAlmostEqual(reply.simulations[0].risk_score_change, 47)
        self.assertEqual(reply.evidence[0]["status"], "success")
        self.assertEqual(reply.evidence[0]["inputs"]["torque_nm"], 60)
        self.assertEqual(len(model.requests), 2)
        json.dumps(reply.to_dict(), allow_nan=False)

    def test_replacement_sets_wear_to_zero_and_preserves_other_inputs(self):
        self.twin.UpdateState(tool_wear_min=200)
        agent, _ = self.agent(request_tool("simulate_scenario", {"tool_wear_min": 0}), AIMessage(content="Comparison complete."))
        reply = agent.ask("What if I replace the tool?")
        self.assertEqual(reply.simulations[0].scenario.state, INITIAL_STATE)
        self.assertEqual(self.twin.state.tool_wear_min, 200)

    def test_invalid_scenarios_return_tool_errors_without_inference(self):
        invalid = [
            {}, {"torque_nm": -1}, {"tool_wear_min": True}, {"torque_nm": "50"},
            {"rotational_speed_rpm": 0}, {"torque_nm": float("inf")},
            {"torque_nm": float("nan")}, {"product_quality": "X"}, {"unknown_input": 12},
        ]
        original = self.twin.snapshot
        for args in invalid:
            with self.subTest(args=args):
                agent, model = self.agent(request_tool("simulate_scenario", args), AIMessage(content="Please correct the input."))
                with patch.object(self.twin._model, "predict_proba", side_effect=AssertionError("Unexpected inference")):
                    reply = agent.ask("Compare this scenario.")
                self.assertEqual(reply.evidence[0]["status"], "error")
                self.assertEqual(reply.simulations, [])
                json.dumps(reply.to_dict(), allow_nan=False)
                self.assertIs(self.twin.snapshot, original)
                self.assertIsInstance(model.requests[1][-1], ToolMessage)

    def test_history_only_contains_supplied_evaluated_readings(self):
        first = self.twin.snapshot
        second = self.twin.UpdateState(tool_wear_min=100)
        agent, _ = self.agent(request_tool("get_machine_history"), AIMessage(content="Two readings are available."))
        reply = agent.ask("What changed?", history=[first, second])
        records = json.loads(reply.evidence[0]["result"])["snapshots"]
        self.assertEqual([record["state"]["tool_wear_min"] for record in records], [0, 100])
        self.assertEqual(reply.context["recorded_snapshot_count"], 2)

    def test_follow_up_retains_memory_and_uses_latest_state(self):
        agent, model = self.agent(AIMessage(content="First answer."), AIMessage(content="Second answer."))
        agent.ask("What is the risk?")
        latest = self.twin.UpdateState(torque_nm=60)
        reply = agent.ask("And now?")
        self.assertEqual(reply.context["current_snapshot"], latest.to_dict())
        self.assertIn(json.dumps(reply.context), model.requests[-1][0].content)
        self.assertEqual([msg.content for msg in model.requests[-1] if isinstance(msg, HumanMessage)], ["What is the risk?", "And now?"])

    def test_failed_turn_does_not_pollute_follow_up_memory(self):
        agent, model = self.agent(RuntimeError("Disconnected"), AIMessage(content="Recovered."))
        with self.assertRaises(RuntimeError):
            agent.ask("Failed question")
        agent.ask("Retry")
        self.assertEqual([msg.content for msg in model.requests[-1] if isinstance(msg, HumanMessage)], ["Retry"])

    def test_memory_is_bounded_and_keeps_complete_turns(self):
        responses = [AIMessage(content=f"Answer {index}") for index in range(MAX_CONVERSATION_TURNS + 2)]
        agent, model = self.agent(*responses)
        for index in range(len(responses)):
            agent.ask(f"Question {index}")
        self.assertEqual(len(agent._turns), MAX_CONVERSATION_TURNS)
        self.assertTrue(all(isinstance(turn[0], HumanMessage) for turn in agent._turns))
        self.assertNotIn("Question 0", [msg.content for msg in model.requests[-1]])

    def test_tool_loop_is_bounded(self):
        agent, model = self.agent(*[request_tool("get_current_state", call_id=f"call_{index}") for index in range(MAX_TOOL_ROUNDS + 1)])
        with self.assertRaisesRegex(CopilotError, "tool limit"):
            agent.ask("Keep looking")
        self.assertEqual(len(model.requests), MAX_TOOL_ROUNDS + 1)
        self.assertEqual(agent._turns, [])

    def test_model_results_are_saved_metrics_and_invalid_names_are_rejected(self):
        agent, _ = self.agent(request_tool("get_model_results", {"model_name": "random_forest"}), AIMessage(content="Saved test results."))
        reply = agent.ask("How accurate is the model?")
        result = json.loads(reply.evidence[0]["result"])
        self.assertTrue(result["selected_for_twin"])
        self.assertEqual(result["metadata"]["test_metrics"]["fn"], 13)
        agent, _ = self.agent(request_tool("get_model_results", {"model_name": "../../private"}), AIMessage(content="Unknown model."))
        self.assertEqual(agent.ask("Read this model").evidence[0]["status"], "error")

    def test_stale_dashboard_simulation_is_not_returned(self):
        comparison = self.twin.Simulate(tool_wear_min=100)
        self.twin.UpdateState(torque_nm=60)
        agent, _ = self.agent(request_tool("get_latest_simulation"), AIMessage(content="No current comparison."))
        reply = agent.ask("Explain the simulation", latest_simulation=comparison)
        self.assertIsNone(json.loads(reply.evidence[0]["result"])["comparison"])

    def test_agents_do_not_share_conversation_or_twin_state(self):
        first, _ = self.agent(AIMessage(content="First session."))
        first.ask("My machine")
        other_twin = DigitalTwin(INITIAL_STATE)
        model = ScriptedModel([AIMessage(content="Other session.")])
        other = CopilotAgent(other_twin, model=model)
        self.twin.UpdateState(torque_nm=60)
        other.ask("My question")
        self.assertEqual(len(model.requests[0]), 2)
        self.assertEqual(other.twin.state.torque_nm, 42.8)

    def test_missing_key_empty_prompt_and_missing_state_have_clear_errors(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            with self.assertRaisesRegex(CopilotError, "API key"):
                CopilotAgent(self.twin)
        agent, model = self.agent(AIMessage(content="Unused"))
        for prompt in ("  ", "x" * 4001):
            with self.assertRaises(CopilotError):
                agent.ask(prompt)
        empty = CopilotAgent(DigitalTwin(), model=model)
        with self.assertRaisesRegex(CopilotError, "Evaluate"):
            empty.ask("Risk?")
        self.assertEqual(model.requests, [])

    def test_empty_answer_is_not_added_to_memory(self):
        agent, _ = self.agent(AIMessage(content=""))
        with self.assertRaisesRegex(CopilotError, "no answer"):
            agent.ask("Risk?")
        self.assertEqual(agent._turns, [])

    def test_responses_api_adapter_sends_tools_and_returns_results(self):
        requests = []
        def respond(request):
            requests.append(json.loads(request.content))
            if len(requests) == 1:
                output = [{
                    "type": "function_call", "id": "fc_1", "call_id": "call_replacement",
                    "name": "simulate_scenario", "arguments": '{"tool_wear_min": 0}',
                    "status": "completed",
                }]
            else:
                output = [{
                    "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": "Replacement comparison complete.", "annotations": []}],
                }]
            return httpx.Response(200, json={
                "id": f"resp_{len(requests)}", "object": "response", "created_at": 1,
                "status": "completed", "model": "gpt-4.1-mini", "output": output,
                "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20},
            })
        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            llm = ChatOpenAI(
                model="gpt-4.1-mini", api_key="test-only-key", use_responses_api=True,
                http_client=client, max_retries=0,
            )
            agent = CopilotAgent(self.twin, model=llm)
            reply = agent.ask("What if I replace the tool?")
        self.assertEqual(reply.text, "Replacement comparison complete.")
        self.assertEqual(len(requests), 2)
        self.assertFalse(requests[0]["parallel_tool_calls"])
        self.assertIn("simulate_scenario", [entry["name"] for entry in requests[0]["tools"]])
        result = next(item for item in requests[1]["input"] if item.get("type") == "function_call_output")
        self.assertEqual(json.loads(result["output"])["comparison"], reply.simulations[0].to_dict())


if __name__ == "__main__":
    unittest.main()

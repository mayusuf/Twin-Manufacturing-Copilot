"""Answer operator questions using evaluated twin state and bounded tool calls."""

from dataclasses import dataclass
import json
import os
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from openai import APIConnectionError, APIError, AuthenticationError, BadRequestError, NotFoundError, RateLimitError
from pydantic import BaseModel, ConfigDict, Field

from src.digital_twin import DigitalTwin, SimulationResult, TwinSnapshot
from src.prediction.feature_engineering import PROJECT_ROOT


DEFAULT_LLM_MODEL = "gpt-4.1-mini"
MAX_TOOL_ROUNDS = 6
MAX_CONVERSATION_TURNS = 10
MAX_HISTORY_SNAPSHOTS = 20
SCORE_INTERPRETATION = (
    "Risk is an uncalibrated model score from 0 to 100, not a calibrated failure "
    "probability. Health is 100 minus risk, not measured health or remaining life. "
    "What-if comparisons are model responses, not physical or causal forecasts."
)
SYSTEM_PROMPT = """You are the Manufacturing Copilot for a single-machine digital twin.
Help the operator understand evaluated readings, model risk, changes in recorded
readings, and what-if comparisons, then suggest appropriate inspections.

Use the fresh context supplied for THIS turn as the current machine state.
Previous conversation and tool results may describe an older state. Obtain all
machine-specific numbers from that context or successful tools. Cite the evidence
in plain language (current reading, recorded history, simulation, or test metrics).
Never invent readings, trends, scores, simulation results, maintenance records,
manufacturer limits, failure causes, or remaining useful life.

Risk is an uncalibrated 0-100 model score, not a calibrated failure probability.
Health is the inverse of risk, not measured health. A failure label describes the
model classification, not a confirmed physical failure or a time-to-failure forecast.
There is no SHAP or other feature attribution here. You can describe readings and
scenario sensitivity, but cannot assert which feature caused the prediction.
History consists only of evaluated inputs recorded in this browser session; these
may be manually entered scenarios, not live sensor measurements. Do not infer a
trend if fewer than two relevant readings are available.

Use simulate_scenario for requested what-if changes. Values are absolute, with
temperatures in kelvin, speed in rpm, torque in Nm, and tool wear in minutes.
Convert explicit units correctly. Ask for missing values or unclear units instead
of choosing an operating setting. Replacing the tool means setting tool_wear_min
to zero while keeping other inputs fixed. Always report both the baseline and
scenario scores and their difference, and explain that this is a model comparison,
not proof that the intervention will prevent failure. A tool error is not a result;
ask for corrected inputs or explain the missing evidence.

Suggest inspections based on observed evidence and label general maintenance
advice as general guidance. Do not claim to operate, update, stop, or repair the
machine. Stay focused on manufacturing and this project. Keep answers concise,
with evidence first and practical next steps afterward.
"""


class ScenarioInputs(BaseModel):
    """Only explicitly requested input changes may be supplied to the twin."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    product_quality: Literal["L", "M", "H"] | None = Field(
        default=None, description="Product quality: L, M, or H; omit to preserve baseline."
    )
    air_temperature_k: float | None = Field(default=None, gt=0, description="Air temperature in K.")
    process_temperature_k: float | None = Field(default=None, gt=0, description="Process temperature in K.")
    rotational_speed_rpm: float | None = Field(default=None, gt=0, description="Rotational speed in rpm.")
    torque_nm: float | None = Field(default=None, ge=0, description="Torque in Nm.")
    tool_wear_min: float | None = Field(default=None, ge=0, description="Tool wear in minutes; 0 for a new tool.")


class AgentState(MessagesState):
    context: dict
    tool_rounds: int


class CopilotError(RuntimeError):
    """A recoverable Copilot connection, input, or execution error."""


@dataclass(frozen=True)
class CopilotReply:
    text: str
    context: dict
    evidence: list[dict]
    simulations: list[SimulationResult]

    def to_dict(self):
        return {
            "text": self.text,
            "context": self.context,
            "evidence": self.evidence,
            "simulations": [result.to_dict() for result in self.simulations],
        }


class CopilotAgent:
    """A session-owned agent; only successful, complete turns enter its memory."""

    def __init__(self, twin: DigitalTwin, *, model=None, api_key=None, model_name=DEFAULT_LLM_MODEL):
        self.twin = twin
        self.history: list[TwinSnapshot] = []
        self.latest_simulation: SimulationResult | None = None
        self._turns: list[list] = []
        if model is None:
            key = (api_key or os.getenv("OPENAI_API_KEY", "")).strip()
            if not key:
                raise CopilotError("Configure an OpenAI API key in Connection or OPENAI_API_KEY.")
            model = ChatOpenAI(
                model=model_name, api_key=key, use_responses_api=True,
                timeout=30, max_retries=1, max_tokens=2000,
            )
        self.tools = self._build_tools()
        self._model = model.bind_tools(self.tools, parallel_tool_calls=False)
        builder = StateGraph(AgentState)
        builder.add_node("load_context", self._load_context)
        builder.add_node("agent", self._reason)
        builder.add_node("tools", ToolNode(
            self.tools, handle_tool_errors=True,
        ))
        builder.add_edge(START, "load_context")
        builder.add_edge("load_context", "agent")
        builder.add_conditional_edges("agent", self._route, {"tools": "tools", "done": END})
        builder.add_edge("tools", "agent")
        self.graph = builder.compile()

    def _build_tools(self):
        @tool
        def get_current_state() -> dict:
            """Read the evaluated current machine inputs and existing prediction; no new inference."""
            return {"snapshot": self.twin.snapshot.to_dict(), "interpretation": SCORE_INTERPRETATION}

        @tool
        def get_machine_history() -> dict:
            """Read up to 20 evaluated machine snapshots from this browser session, oldest first."""
            return {
                "source": "Evaluated inputs in this browser session, not a live sensor feed.",
                "snapshots": [snapshot.to_dict() for snapshot in self.history],
                "interpretation": SCORE_INTERPRETATION,
            }

        @tool(args_schema=ScenarioInputs, response_format="content_and_artifact")
        def simulate_scenario(
            product_quality=None, air_temperature_k=None, process_temperature_k=None,
            rotational_speed_rpm=None, torque_nm=None, tool_wear_min=None,
        ) -> tuple:
            """Compare explicit absolute input changes against the current state, preserving the baseline."""
            changes = {
                "product_quality": product_quality, "air_temperature_k": air_temperature_k,
                "process_temperature_k": process_temperature_k, "rotational_speed_rpm": rotational_speed_rpm,
                "torque_nm": torque_nm, "tool_wear_min": tool_wear_min,
            }
            changes = {name: value for name, value in changes.items() if value is not None}
            if not changes:
                raise ValueError("Provide at least one explicit input change for the what-if scenario.")
            result = self.twin.Simulate(**changes)
            return {"comparison": result.to_dict(), "interpretation": SCORE_INTERPRETATION}, result

        @tool
        def get_latest_simulation() -> dict:
            """Read the latest dashboard comparison, if it belongs to the current evaluated baseline."""
            if self.latest_simulation is None or self.latest_simulation.baseline != self.twin.snapshot:
                return {"comparison": None, "note": "No comparison available for the current baseline."}
            return {"comparison": self.latest_simulation.to_dict(), "interpretation": SCORE_INTERPRETATION}

        @tool
        def get_model_results(model_name: Literal["random_forest", "xgboost"] = "random_forest") -> dict:
            """Read saved test performance and metadata; the active twin uses Random Forest."""
            metadata = json.loads((PROJECT_ROOT / "models" / f"{model_name}_metadata.json").read_text())
            return {"metadata": metadata, "selected_for_twin": model_name == "random_forest"}

        return [get_current_state, get_machine_history, simulate_scenario, get_latest_simulation, get_model_results]

    def _load_context(self, state):
        return {"context": {
            "current_snapshot": self.twin.snapshot.to_dict(),
            "recorded_snapshot_count": len(self.history),
            "interpretation": SCORE_INTERPRETATION,
        }}

    def _reason(self, state):
        if state["tool_rounds"] > MAX_TOOL_ROUNDS:
            raise CopilotError("The Copilot reached its tool limit. Try a more specific question.")
        context = json.dumps(state["context"], allow_nan=False)
        response = self._model.invoke([
            SystemMessage(content=SYSTEM_PROMPT + "\nFresh context for this turn:\n" + context),
            *state["messages"],
        ])
        if not isinstance(response, AIMessage):
            raise CopilotError("The language model did not return an assistant response.")
        if response.tool_calls and state["tool_rounds"] >= MAX_TOOL_ROUNDS:
            raise CopilotError("The Copilot reached its tool limit. Try a more specific question.")
        return {"messages": [response], "tool_rounds": state["tool_rounds"] + bool(response.tool_calls)}

    @staticmethod
    def _route(state):
        return "tools" if state["messages"][-1].tool_calls else "done"

    def ask(self, prompt: str, *, history=(), latest_simulation=None) -> CopilotReply:
        prompt = prompt.strip()
        if not prompt or len(prompt) > 4000:
            raise CopilotError("Enter a question between 1 and 4000 characters.")
        try:
            self.twin.snapshot
        except ValueError as error:
            raise CopilotError("Evaluate the machine state before asking the Copilot.") from error
        self.history = list(history)[-MAX_HISTORY_SNAPSHOTS:]
        self.latest_simulation = latest_simulation
        previous = [message for turn in self._turns for message in turn]
        try:
            result = self.graph.invoke(
                {"messages": [*previous, HumanMessage(content=prompt)], "tool_rounds": 0},
                config={"recursion_limit": MAX_TOOL_ROUNDS * 2 + 5},
            )
        except AuthenticationError as error:
            raise CopilotError("OpenAI rejected the API key. Check the key in Connection.") from error
        except RateLimitError as error:
            raise CopilotError("OpenAI quota or rate limit reached. Check API billing or try again later.") from error
        except APIConnectionError as error:
            raise CopilotError("Could not connect to OpenAI. Check the network and try again.") from error
        except (BadRequestError, NotFoundError) as error:
            raise CopilotError("OpenAI could not use this model or request. Check the model in Connection.") from error
        except APIError as error:
            raise CopilotError("OpenAI could not complete the request. Try again later.") from error
        turn = result["messages"][len(previous):]
        # Preserve complete tool-call/result pairs when limiting conversational memory.
        text = turn[-1].text.strip()
        if not text:
            raise CopilotError("The Copilot returned no answer. Try the question again.")
        calls = {call["id"]: call for message in turn if isinstance(message, AIMessage) for call in message.tool_calls}
        evidence, simulations = [], []
        for message in turn:
            if isinstance(message, ToolMessage):
                inputs = calls.get(message.tool_call_id, {}).get("args", {})
                try:
                    json.dumps(inputs, allow_nan=False)
                except (TypeError, ValueError):
                    inputs = {"invalid_inputs": str(inputs)}
                evidence.append({
                    "tool": message.name, "inputs": inputs,
                    "status": message.status, "result": message.content,
                })
                if isinstance(message.artifact, SimulationResult):
                    simulations.append(message.artifact)
        self._turns = [*self._turns, turn][-MAX_CONVERSATION_TURNS:]
        return CopilotReply(text, result["context"], evidence, simulations)

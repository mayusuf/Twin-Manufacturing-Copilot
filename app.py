"""Streamlit interface for the single-machine manufacturing Digital Twin."""

from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import os

import pandas as pd
import streamlit as st

from src.digital_twin import DigitalTwin
from src.copilot import CopilotAgent, CopilotError
from src.copilot.agent import DEFAULT_LLM_MODEL, MAX_HISTORY_SNAPSHOTS
from src.prediction.feature_engineering import PROJECT_ROOT


DEFAULT_INPUTS = {
    "product_quality": "M",
    "air_temperature_k": 298.1,
    "process_temperature_k": 308.6,
    "rotational_speed_rpm": 1551.0,
    "torque_nm": 42.8,
    "tool_wear_min": 0.0,
}
QUALITY_LABELS = {"L": "L - Low", "M": "M - Medium", "H": "H - High"}
INPUT_LABELS = {
    "product_quality": "Product quality",
    "air_temperature_k": "Air temperature (K)",
    "process_temperature_k": "Process temperature (K)",
    "rotational_speed_rpm": "Rotational speed (rpm)",
    "torque_nm": "Torque (Nm)",
    "tool_wear_min": "Tool wear (min)",
}
SCORE_NOTE = "Uncalibrated model score. Health indicator is 100 minus risk score, not measured machine health."


def reset_session():
    for key in list(st.session_state):
        if key in ("twin", "snapshot", "simulation", "revision", "machine_history") or key.startswith(("current_", "scenario_", "copilot_")):
            del st.session_state[key]


def machine_inputs(prefix, defaults):
    values = {}
    left, right = st.columns(2)
    with left:
        values["product_quality"] = st.selectbox(
            INPUT_LABELS["product_quality"], list(QUALITY_LABELS),
            index=list(QUALITY_LABELS).index(defaults["product_quality"]),
            format_func=QUALITY_LABELS.get, key=f"{prefix}_product_quality",
        )
        values["air_temperature_k"] = st.number_input(
            INPUT_LABELS["air_temperature_k"], min_value=0.1,
            value=float(defaults["air_temperature_k"]), step=0.1, format="%.1f",
            key=f"{prefix}_air_temperature_k",
        )
        values["rotational_speed_rpm"] = st.number_input(
            INPUT_LABELS["rotational_speed_rpm"], min_value=1.0,
            value=float(defaults["rotational_speed_rpm"]), step=10.0, format="%.0f",
            key=f"{prefix}_rotational_speed_rpm",
        )
    with right:
        values["tool_wear_min"] = st.number_input(
            INPUT_LABELS["tool_wear_min"], min_value=0.0,
            value=float(defaults["tool_wear_min"]), step=1.0, format="%.1f",
            key=f"{prefix}_tool_wear_min",
        )
        values["process_temperature_k"] = st.number_input(
            INPUT_LABELS["process_temperature_k"], min_value=0.1,
            value=float(defaults["process_temperature_k"]), step=0.1, format="%.1f",
            key=f"{prefix}_process_temperature_k",
        )
        values["torque_nm"] = st.number_input(
            INPUT_LABELS["torque_nm"], min_value=0.0,
            value=float(defaults["torque_nm"]), step=0.1, format="%.1f",
            key=f"{prefix}_torque_nm",
        )
    return values


def snapshot_download(snapshot):
    payload = {
        "model": "random_forest",
        "score_interpretation": SCORE_NOTE,
        "snapshot": snapshot.to_dict(),
    }
    st.download_button(
        "Download inputs and prediction", json.dumps(payload, indent=2, allow_nan=False),
        file_name="machine_prediction.json", mime="application/json",
        icon=":material/download:", on_click="ignore",
    )


def render_prediction(snapshot):
    st.subheader("Current prediction")
    if snapshot is None:
        st.info("No evaluated machine state.", icon=":material/pending:")
        return
    prediction = snapshot.prediction
    if prediction.machine_failure:
        st.error("Failure predicted", icon=":material/warning:")
    else:
        st.success("No failure predicted", icon=":material/check_circle:")
    risk, health = st.columns(2)
    risk.metric("Model risk score", f"{prediction.risk_score:.2f} / 100")
    health.metric("Health indicator", f"{prediction.health_indicator:.2f} / 100")
    st.progress(prediction.failure_model_score)
    st.caption(f"Classification threshold: {prediction.threshold * 100:.0f} / 100")
    with st.expander("Evaluated machine state"):
        st.table(pd.DataFrame({
            "Input": [INPUT_LABELS[key] for key in DEFAULT_INPUTS],
            "Value": [str(getattr(snapshot.state, key)) for key in DEFAULT_INPUTS],
        }).set_index("Input"))
        st.caption(f"Evaluated at {snapshot.timestamp_utc} (UTC)")
    snapshot_download(snapshot)


def render_current_state():
    input_column, output_column = st.columns([1.1, 1], gap="large")
    with input_column:
        st.subheader("Machine inputs")
        with st.form("current_state", border=False):
            values = machine_inputs("current", DEFAULT_INPUTS)
            submitted = st.form_submit_button(
                "Predict and update", type="primary", icon=":material/play_arrow:", width="stretch",
            )
        if submitted:
            try:
                twin = st.session_state.get("twin")
                if twin is None:
                    twin = DigitalTwin()
                snapshot = twin.UpdateState(**values)
            except (ValueError, TypeError, OSError, RuntimeError) as error:
                st.error(f"Prediction could not be completed: {error}")
            else:
                st.session_state["twin"] = twin
                st.session_state["snapshot"] = snapshot
                st.session_state["simulation"] = None
                st.session_state["revision"] = st.session_state.get("revision", 0) + 1
                history = st.session_state.get("machine_history", [])
                st.session_state["machine_history"] = [*history, snapshot][-MAX_HISTORY_SNAPSHOTS:]
    with output_column:
        render_prediction(st.session_state.get("snapshot"))


def render_simulation():
    snapshot = st.session_state.get("snapshot")
    if snapshot is None:
        st.info("No evaluated machine state available for simulation.", icon=":material/pending:")
        return
    input_column, output_column = st.columns([1.1, 1], gap="large")
    with input_column:
        st.subheader("Scenario inputs")
        revision = st.session_state["revision"]
        with st.form(f"scenario_form_{revision}", border=False):
            values = machine_inputs(f"scenario_{revision}", asdict(snapshot.state))
            submitted = st.form_submit_button(
                "Run simulation", type="primary", icon=":material/science:", width="stretch",
            )
        if submitted:
            try:
                comparison = st.session_state["twin"].Simulate(**values)
            except (ValueError, TypeError, OSError, RuntimeError) as error:
                st.error(f"Simulation could not be completed: {error}")
            else:
                st.session_state["simulation"] = comparison
        st.caption("Changed-input model comparison; not a physical or causal forecast.")
    with output_column:
        st.subheader("Original vs. what-if")
        comparison = st.session_state.get("simulation")
        if comparison is None:
            st.info("No scenario evaluated.", icon=":material/pending:")
            return
        baseline, scenario = comparison.baseline.prediction, comparison.scenario.prediction
        risk, change = st.columns(2)
        risk.metric("Scenario risk score", f"{scenario.risk_score:.2f} / 100")
        change.metric("Risk score change", f"{comparison.risk_score_change:+.2f} points")
        table = {
            "Prediction": [baseline.prediction_label, scenario.prediction_label],
            "Risk score / 100": [f"{baseline.risk_score:.2f}", f"{scenario.risk_score:.2f}"],
            "Health indicator / 100": [f"{baseline.health_indicator:.2f}", f"{scenario.health_indicator:.2f}"],
        }
        for key, label in INPUT_LABELS.items():
            table[label] = [str(getattr(comparison.baseline.state, key)), str(getattr(comparison.scenario.state, key))]
        st.table(pd.DataFrame(table, index=["Original", "What-if"]).T)
        st.caption("Original machine state unchanged.")
        payload = {
            "model": "random_forest", "score_interpretation": SCORE_NOTE,
            "simulation": comparison.to_dict(),
        }
        st.download_button(
            "Download comparison", json.dumps(payload, indent=2, allow_nan=False),
            file_name="what_if_comparison.json", mime="application/json",
            icon=":material/download:", on_click="ignore",
        )


def render_model_results():
    st.subheader("Test performance")
    records = []
    for name, title in (("random_forest", "Random Forest (selected)"), ("xgboost", "XGBoost")):
        path = PROJECT_ROOT / "models" / f"{name}_metadata.json"
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
            metrics = metadata["test_metrics"]
            records.append({
                "Model": title, "Accuracy": f"{metrics['accuracy']:.2%}",
                "Precision": f"{metrics['precision']:.2%}", "Recall": f"{metrics['recall']:.2%}",
                "F1": f"{metrics['f1']:.2%}", "ROC-AUC": f"{metrics['roc_auc']:.4f}",
                "Average precision": f"{metrics['average_precision']:.4f}",
                "Missed failures": metrics["fn"], "False alarms": metrics["fp"],
            })
        except (OSError, ValueError, KeyError) as error:
            st.warning(f"{title} test results unavailable: {error}")
    if records:
        st.dataframe(pd.DataFrame(records), hide_index=True, width="stretch")
    figure = PROJECT_ROOT / "reports" / "figures" / "model_comparison.png"
    if figure.exists():
        st.image(str(figure), width="stretch")


def clear_copilot_chat():
    for key in ("copilot_agent", "copilot_signature", "copilot_messages", "copilot_error"):
        st.session_state.pop(key, None)


def configured_api_key():
    key = os.getenv("OPENAI_API_KEY", "")
    if key:
        return key
    try:
        return st.secrets.get("OPENAI_API_KEY", "")
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return ""


def render_copilot_evidence(reply):
    with st.expander("Evidence"):
        st.caption("Evaluated machine state used for this answer")
        st.json(reply["context"])
        labels = {
            "get_current_state": "Current readings and prediction",
            "get_machine_history": "Recorded readings",
            "simulate_scenario": "What-if comparison",
            "get_latest_simulation": "Latest comparison",
            "get_model_results": "Model test results",
        }
        for evidence in reply["evidence"]:
            st.caption(labels.get(evidence["tool"], "Additional evidence"))
            result = evidence["result"]
            try:
                st.json(json.loads(result))
            except (TypeError, ValueError):
                st.write(result)
    for comparison in reply["simulations"]:
        baseline, scenario = comparison["baseline"], comparison["scenario"]
        st.table(pd.DataFrame({
            "Prediction": [baseline["prediction"]["prediction_label"], scenario["prediction"]["prediction_label"]],
            "Risk score / 100": [f"{baseline['prediction']['risk_score']:.2f}", f"{scenario['prediction']['risk_score']:.2f}"],
            **{INPUT_LABELS[key]: [str(baseline["state"][key]), str(scenario["state"][key])] for key in INPUT_LABELS},
        }, index=["Original", "What-if"]).T)


def render_copilot():
    heading, actions = st.columns([5, 1])
    with heading:
        st.subheader("Copilot")
    with actions:
        st.button("Clear chat", icon=":material/delete_sweep:", on_click=clear_copilot_chat, key="copilot_clear_chat")
    default_key = configured_api_key()
    with st.expander("Connection", expanded=not bool(default_key or st.session_state.get("copilot_api_key"))):
        connection, model = st.columns(2)
        with connection:
            entered_key = st.text_input("OpenAI API key", type="password", key="copilot_api_key")
            if default_key and not entered_key:
                st.caption("API key configured on the server.")
        with model:
            model_name = st.text_input(
                "LLM model", value=os.getenv("OPENAI_MODEL", DEFAULT_LLM_MODEL), key="copilot_model",
            ).strip()
    api_key = (entered_key or default_key).strip()
    signature = (model_name, hashlib.sha256(api_key.encode()).hexdigest())
    if "copilot_signature" in st.session_state and st.session_state["copilot_signature"] != signature:
        clear_copilot_chat()
    snapshot = st.session_state.get("snapshot")
    if snapshot is None:
        st.info("No evaluated machine state. Submit machine inputs before asking the Copilot.", icon=":material/pending:")
    else:
        evaluated_at = datetime.fromisoformat(snapshot.timestamp_utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        st.caption(f"Current risk: {snapshot.prediction.risk_score:.2f} / 100 | Evaluated at {evaluated_at}")
    if not api_key:
        st.info("OpenAI API key required. Set it in Connection or OPENAI_API_KEY.", icon=":material/key:")
    messages = st.session_state.setdefault("copilot_messages", [])
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_copilot_evidence(message["reply"])
    if st.session_state.get("copilot_error"):
        st.error(st.session_state["copilot_error"])
    prompt = st.chat_input(
        "Ask about machine risk or a what-if scenario", key="copilot_prompt",
        disabled=snapshot is None or not api_key or not model_name, max_chars=4000,
    )
    if prompt:
        st.session_state.pop("copilot_error", None)
        try:
            agent = st.session_state.get("copilot_agent")
            if agent is None:
                agent = CopilotAgent(st.session_state["twin"], api_key=api_key, model_name=model_name)
                st.session_state["copilot_agent"] = agent
                st.session_state["copilot_signature"] = signature
            with st.spinner("Checking machine evidence..."):
                reply = agent.ask(
                    prompt, history=st.session_state.get("machine_history", []),
                    latest_simulation=st.session_state.get("simulation"),
                )
        except CopilotError as error:
            st.session_state["copilot_error"] = str(error)
        except Exception:
            st.session_state["copilot_error"] = "Copilot could not complete the request. Check the connection and try again."
        else:
            messages.extend([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": reply.text, "reply": reply.to_dict()},
            ])
            if reply.simulations:
                st.session_state["simulation"] = reply.simulations[-1]
        st.rerun()
    if messages:
        st.download_button(
            "Download conversation", json.dumps(messages, indent=2, allow_nan=False),
            file_name="copilot_conversation.json", mime="application/json",
            icon=":material/download:", on_click="ignore",
        )


def main():
    st.set_page_config(page_title="Manufacturing Digital Twin", page_icon=":material/precision_manufacturing:", layout="wide")
    st.html("""<style>
        .stMainBlockContainer {max-width: 1280px; padding-top: 4rem; padding-bottom: 2rem;}
        h1 {font-size: 2rem !important; letter-spacing: 0 !important;}
        h2, h3 {font-size: 1.2rem !important; letter-spacing: 0 !important;}
        [data-testid="stMetricValue"] {font-size: 1.6rem;}
        @media (max-width: 640px) {
            .stMainBlockContainer {padding-left: 1rem; padding-right: 1rem;}
        }
    </style>""")
    title, actions = st.columns([5, 1])
    with title:
        st.title("Manufacturing Digital Twin")
        st.caption("Random Forest | Single machine")
    with actions:
        st.button("Reset", icon=":material/restart_alt:", help="Clear machine state, scenario, history, and chat", on_click=reset_session, key="reset_session")
    current, simulation, results, copilot = st.tabs(["Machine state", "What-if simulation", "Model results", "Copilot"])
    with current:
        render_current_state()
    with simulation:
        render_simulation()
    with results:
        render_model_results()
    with copilot:
        render_copilot()


if __name__ == "__main__":
    main()

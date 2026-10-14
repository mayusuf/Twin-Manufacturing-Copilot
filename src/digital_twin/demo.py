"""Run a single-machine Digital Twin and save a what-if comparison."""

import json

from src.digital_twin import DigitalTwin, MachineState
from src.prediction.feature_engineering import PROJECT_ROOT


def main():
    twin = DigitalTwin(MachineState(
        product_quality="M",
        air_temperature_k=298.1,
        process_temperature_k=308.6,
        rotational_speed_rpm=1551,
        torque_nm=42.8,
        tool_wear_min=0,
    ))
    simulation = twin.Simulate(rotational_speed_rpm=1200, torque_nm=60, tool_wear_min=220)
    report = {
        "model": "random_forest",
        "risk_score_definition": "100 * failure_model_score; uncalibrated model score.",
        "health_indicator_definition": "100 - risk_score; a heuristic inverse-risk indicator.",
        "simulation_interpretation": "Changed-input model comparison, not a physical or causal forecast.",
        "simulation": simulation.to_dict(),
        "original_state_preserved": twin.snapshot == simulation.baseline,
    }
    output_path = PROJECT_ROOT / "reports" / "digital_twin_demo.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for label, snapshot in (("Current", simulation.baseline), ("What-if", simulation.scenario)):
        prediction = snapshot.prediction
        print(
            f"{label}: {prediction.prediction_label}; "
            f"model risk score {prediction.risk_score:.2f}/100; "
            f"health indicator {prediction.health_indicator:.2f}/100"
        )
    print(f"Risk score change: {simulation.risk_score_change:+.2f} points")
    print(f"Original state preserved: {report['original_state_preserved']}")
    print(f"Saved demo: {output_path}")


if __name__ == "__main__":
    main()

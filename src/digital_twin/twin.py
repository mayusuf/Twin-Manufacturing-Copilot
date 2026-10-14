"""Maintain a virtual machine state and compare model-based operating scenarios."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from numbers import Real
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.prediction.feature_engineering import (
    BASE_NUMERIC_FEATURES,
    FEATURE_COLUMNS,
    PREDICTOR_COLUMNS,
    PROJECT_ROOT,
    QUALITY_LEVELS,
    TARGET,
    build_predictor_table,
)


MODEL_PATH = PROJECT_ROOT / "models" / "random_forest.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "random_forest_metadata.json"


@dataclass(frozen=True)
class MachineState:
    product_quality: str
    air_temperature_k: float
    process_temperature_k: float
    rotational_speed_rpm: float
    torque_nm: float
    tool_wear_min: float

    def __post_init__(self):
        if not isinstance(self.product_quality, str) or self.product_quality not in QUALITY_LEVELS:
            raise ValueError("product_quality must be L, M, or H.")
        for name in BASE_NUMERIC_FEATURES:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number.")
            if name in ("torque_nm", "tool_wear_min"):
                if value < 0:
                    raise ValueError(f"{name} cannot be negative.")
            elif value <= 0:
                raise ValueError(f"{name} must be positive.")
            object.__setattr__(self, name, float(value))


@dataclass(frozen=True)
class PredictionResult:
    machine_failure: int
    prediction_label: str
    failure_model_score: float
    risk_score: float
    health_indicator: float
    threshold: float


@dataclass(frozen=True)
class TwinSnapshot:
    state: MachineState
    prediction: PredictionResult
    timestamp_utc: str

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class SimulationResult:
    baseline: TwinSnapshot
    scenario: TwinSnapshot
    risk_score_change: float
    health_indicator_change: float

    def to_dict(self):
        return asdict(self)


class DigitalTwin:
    """A single virtual machine backed by the selected Random Forest model."""

    def __init__(
        self,
        initial_state: MachineState | None = None,
        *,
        model_path: Path = MODEL_PATH,
        metadata_path: Path = METADATA_PATH,
    ):
        model_path = Path(model_path)
        metadata_path = Path(metadata_path)
        for path in (model_path, metadata_path):
            if not path.exists():
                raise FileNotFoundError(f"Missing Random Forest artifact: {path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("model") != "random_forest" or metadata.get("target") != TARGET:
            raise ValueError("Expected Random Forest metadata for machine_failure.")
        if metadata.get("feature_columns") != FEATURE_COLUMNS:
            raise ValueError("Saved model metadata does not match the engineered features.")
        threshold = metadata.get("threshold")
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, Real)
            or not math.isfinite(threshold)
            or not 0 <= threshold <= 1
        ):
            raise ValueError("Model threshold must be a finite number between 0 and 1.")
        self._threshold = float(threshold)
        self._model = joblib.load(model_path)
        if not isinstance(self._model, RandomForestClassifier):
            raise ValueError("The selected model must be a fitted RandomForestClassifier.")
        if list(getattr(self._model, "feature_names_in_", [])) != FEATURE_COLUMNS:
            raise ValueError("Saved model feature order does not match the input pipeline.")
        if not np.array_equal(getattr(self._model, "classes_", []), [0, 1]):
            raise ValueError("Expected fitted model classes 0 and 1.")
        self._snapshot: TwinSnapshot | None = None
        if initial_state is not None:
            if not isinstance(initial_state, MachineState):
                raise TypeError("initial_state must be a MachineState.")
            self.UpdateState(**asdict(initial_state))

    @property
    def snapshot(self) -> TwinSnapshot:
        if self._snapshot is None:
            raise ValueError("No machine state set. Call UpdateState with all six inputs first.")
        return self._snapshot

    @property
    def state(self) -> MachineState:
        return self.snapshot.state

    def _candidate_state(self, changes) -> MachineState:
        unknown = set(changes) - set(PREDICTOR_COLUMNS)
        if unknown:
            raise ValueError(f"Unknown machine input fields: {sorted(unknown)}")
        values = asdict(self._snapshot.state) if self._snapshot is not None else {}
        values.update(changes)
        missing = set(PREDICTOR_COLUMNS) - set(values)
        if missing:
            raise ValueError(f"Missing machine input fields: {sorted(missing)}")
        return MachineState(**values)

    def _evaluate(self, state: MachineState) -> TwinSnapshot:
        features = build_predictor_table(pd.DataFrame([asdict(state)]))
        score = float(self._model.predict_proba(features)[0, 1])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("The model returned an invalid failure score.")
        predicted_class = int(score >= self._threshold)
        risk_score = score * 100
        prediction = PredictionResult(
            machine_failure=predicted_class,
            prediction_label="Failure" if predicted_class else "No failure",
            failure_model_score=score,
            risk_score=risk_score,
            health_indicator=100 - risk_score,
            threshold=self._threshold,
        )
        return TwinSnapshot(state, prediction, datetime.now(timezone.utc).isoformat())

    def UpdateState(self, **changes) -> TwinSnapshot:
        """Validate and predict before committing a complete or partial update."""
        candidate = self._candidate_state(changes)
        snapshot = self._evaluate(candidate)
        self._snapshot = snapshot
        return snapshot

    def PredictFailure(self) -> PredictionResult:
        """Return the current class and uncalibrated model risk score."""
        return self.snapshot.prediction

    def CalculateHealth(self) -> float:
        """Return an inverse-risk indicator, not a measured machine health score."""
        return self.PredictFailure().health_indicator

    def Simulate(self, **changes) -> SimulationResult:
        """Compare an altered input scenario without changing the current twin."""
        baseline = self.snapshot
        candidate = self._candidate_state(changes)
        scenario = self._evaluate(candidate)
        return SimulationResult(
            baseline=baseline,
            scenario=scenario,
            risk_score_change=scenario.prediction.risk_score - baseline.prediction.risk_score,
            health_indicator_change=(
                scenario.prediction.health_indicator - baseline.prediction.health_indicator
            ),
        )

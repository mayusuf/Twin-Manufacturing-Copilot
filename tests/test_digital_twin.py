"""Verify training/inference parity and Digital Twin state isolation."""

from dataclasses import FrozenInstanceError, asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from src.digital_twin import DigitalTwin, MachineState
from src.digital_twin.twin import METADATA_PATH, MODEL_PATH
from src.prediction.feature_engineering import (
    FEATURE_OUTPUT_PATH,
    MODEL_INPUT_PATH,
    PREDICTOR_COLUMNS,
    TARGET,
    build_feature_table,
    build_predictor_table,
)


class DigitalTwinTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = pd.read_csv(MODEL_INPUT_PATH)
        cls.saved_features = pd.read_csv(FEATURE_OUTPUT_PATH)
        cls.model = joblib.load(MODEL_PATH)
        cls.initial_state = MachineState(**cls.raw.loc[0, PREDICTOR_COLUMNS].to_dict())

    def setUp(self):
        self.twin = DigitalTwin(self.initial_state)

    def test_shared_feature_pipeline_preserves_training_table(self):
        pd.testing.assert_frame_equal(
            build_feature_table(self.raw), self.saved_features, rtol=1e-12, atol=1e-12,
        )

    def test_unlabeled_predictors_preserve_nondefault_row_indices(self):
        raw = self.raw.iloc[[1, 0, 17]][PREDICTOR_COLUMNS].copy()
        raw.index = [7, 25, 41]
        expected = self.saved_features.iloc[[1, 0, 17]].drop(columns=[TARGET]).copy()
        expected.index = raw.index
        actual = build_predictor_table(raw)
        pd.testing.assert_frame_equal(actual, expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(
            actual[["product_quality_L", "product_quality_M", "product_quality_H"]].sum(axis=1),
            np.ones(len(raw)),
        )

    def test_predictions_match_saved_model_and_feature_order(self):
        failure_index = self.raw.index[self.raw[TARGET] == 1][0]
        high_quality_index = self.raw.index[self.raw["product_quality"] == "H"][0]
        for index in (0, failure_index, high_quality_index):
            with self.subTest(index=index):
                snapshot = self.twin.UpdateState(**self.raw.loc[index, PREDICTOR_COLUMNS].to_dict())
                features = self.saved_features.loc[[index]].drop(columns=[TARGET])
                score = float(self.model.predict_proba(features)[0, 1])
                prediction = self.twin.PredictFailure()
                self.assertAlmostEqual(prediction.failure_model_score, score)
                self.assertEqual(prediction.machine_failure, int(score >= prediction.threshold))
                self.assertAlmostEqual(prediction.risk_score, score * 100)
                self.assertAlmostEqual(self.twin.CalculateHealth(), 100 - score * 100)
                self.assertEqual(datetime.fromisoformat(snapshot.timestamp_utc).tzinfo, timezone.utc)

    def test_partial_update_and_immutable_snapshots(self):
        original = self.twin.snapshot
        updated = self.twin.UpdateState(tool_wear_min=100)
        self.assertEqual(updated.state.tool_wear_min, 100)
        self.assertEqual(original.state.tool_wear_min, 0)
        self.assertEqual(updated.state.torque_nm, original.state.torque_nm)
        with self.assertRaises(FrozenInstanceError):
            self.twin.state.tool_wear_min = 500
        exported = updated.to_dict()
        exported["state"]["torque_nm"] = 999
        self.assertEqual(self.twin.state.torque_nm, original.state.torque_nm)
        json.dumps(exported, allow_nan=False)

    def test_invalid_updates_leave_current_state_intact(self):
        original = self.twin.snapshot
        invalid_changes = [
            {"product_quality": "X"}, {"air_temperature_k": 0},
            {"process_temperature_k": -1}, {"rotational_speed_rpm": 0},
            {"torque_nm": -1}, {"tool_wear_min": -1},
            {"torque_nm": True}, {"torque_nm": "42.8"}, {"torque_nm": None},
            {"tool_wear_min": float("nan")}, {"torque_nm": float("inf")},
            {TARGET: 1}, {"unknown_field": 5},
        ]
        for changes in invalid_changes:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    self.twin.UpdateState(**changes)
                self.assertIs(self.twin.snapshot, original)
        with patch.object(self.twin._model, "predict_proba", side_effect=RuntimeError("Inference failed")):
            with self.assertRaises(RuntimeError):
                self.twin.UpdateState(tool_wear_min=100)
        self.assertIs(self.twin.snapshot, original)

    def test_empty_twin_requires_complete_initial_inputs(self):
        twin = DigitalTwin()
        for action in (twin.PredictFailure, twin.CalculateHealth, twin.Simulate):
            with self.assertRaises(ValueError):
                action()
        with self.assertRaises(ValueError):
            twin.UpdateState(torque_nm=40)
        twin.UpdateState(**asdict(self.initial_state))
        self.assertEqual(twin.state, self.initial_state)

    def test_simulation_preserves_baseline_and_reports_score_changes(self):
        original = self.twin.snapshot
        comparison = self.twin.Simulate(rotational_speed_rpm=1200, torque_nm=60, tool_wear_min=220)
        self.assertIs(self.twin.snapshot, original)
        self.assertIs(comparison.baseline, original)
        self.assertEqual(comparison.scenario.state.rotational_speed_rpm, 1200)
        self.assertEqual(self.twin.state.rotational_speed_rpm, 1551)
        expected = build_predictor_table(pd.DataFrame([asdict(comparison.scenario.state)]))
        score = float(self.model.predict_proba(expected)[0, 1])
        self.assertAlmostEqual(comparison.scenario.prediction.failure_model_score, score)
        self.assertAlmostEqual(comparison.risk_score_change, score * 100 - original.prediction.risk_score)
        self.assertAlmostEqual(comparison.health_indicator_change, -comparison.risk_score_change)
        json.dumps(comparison.to_dict(), allow_nan=False)
        unchanged = self.twin.Simulate()
        self.assertEqual(unchanged.risk_score_change, 0)
        self.assertEqual(unchanged.scenario.state, original.state)
        with self.assertRaises(ValueError):
            self.twin.Simulate(rotational_speed_rpm=0)
        self.assertIs(self.twin.snapshot, original)

    def test_metadata_mismatch_is_rejected(self):
        metadata = json.loads(METADATA_PATH.read_text())
        with TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.json"
            for changes in (
                {"feature_columns": metadata["feature_columns"][::-1]},
                {"threshold": 1.5}, {"threshold": True}, {"model": "xgboost"},
                {"target": "wrong_target"},
            ):
                with self.subTest(changes=changes):
                    path.write_text(json.dumps({**metadata, **changes}), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        DigitalTwin(metadata_path=path)


if __name__ == "__main__":
    unittest.main()

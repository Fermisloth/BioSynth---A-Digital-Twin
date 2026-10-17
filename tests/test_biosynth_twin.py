"""Small, dependency-light checks for the user-facing Digital Twin runtime."""

import unittest

import pandas as pd

from src.biosynth_twin import BioSynthTwin


def sample_history() -> pd.DataFrame:
    """Return a short five-minute CGM history with one future reading."""
    timestamps = pd.date_range("2026-01-04 11:10:00", periods=14, freq="5min")
    glucose = [120, 122, 125, 130, 134, 140, 145, 151, 158, 164, 170, 176, 182, 240]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "glucose_value_mg_dl": glucose,
        }
    )


EHR_PROFILE = {
    "age_years": 48,
    "sex": "female",
    "bmi_kg_m2": 24.7,
    "hba1c_percent": 5.1,
    "fasting_glucose_mg_dl": 82,
    "family_history_diabetes": True,
    "hypertension_history": False,
    "activity_level": "high",
}


class BioSynthTwinTests(unittest.TestCase):
    def test_cgm_only_uses_only_history_at_prediction_time(self) -> None:
        twin = BioSynthTwin()
        history = sample_history()
        prediction_time = pd.Timestamp("2026-01-04 12:05:00")

        with_future = twin.predict_state(history, prediction_time)
        without_future = twin.predict_state(
            history[history["timestamp"] <= prediction_time], prediction_time
        )

        self.assertEqual(with_future["model_mode"], "CGM_ONLY")
        self.assertEqual(with_future["feature_count"], 18)
        self.assertEqual(with_future["history_observation_count"], 12)
        self.assertEqual(
            with_future["hyperglycemia_risk"],
            without_future["hyperglycemia_risk"],
        )

    def test_fusion_requires_an_explicit_ehr_profile(self) -> None:
        twin = BioSynthTwin(model_mode="EHR_CGM_FUSION")

        with self.assertRaisesRegex(ValueError, "requires an explicit static EHR profile"):
            twin.predict_state(sample_history(), "2026-01-04 12:05:00")

    def test_fusion_returns_29_features(self) -> None:
        twin = BioSynthTwin(model_mode="EHR_CGM_FUSION")
        state = twin.predict_state(
            sample_history(),
            "2026-01-04 12:05:00",
            ehr_profile=EHR_PROFILE,
        )

        self.assertEqual(state["model_mode"], "EHR_CGM_FUSION")
        self.assertEqual(state["feature_count"], 29)
        self.assertEqual(len(state["feature_order"]), 29)
        self.assertGreaterEqual(state["hyperglycemia_risk"], 0.0)
        self.assertLessEqual(state["hyperglycemia_risk"], 1.0)

    def test_invalid_model_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "model_mode must be either"):
            BioSynthTwin(model_mode="UNKNOWN")


if __name__ == "__main__":
    unittest.main()

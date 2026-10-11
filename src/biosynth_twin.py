from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from scripts.build_ehr_cgm_fusion_features import EHR_FEATURES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "biosynth_trajectory_model.joblib"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "models" / "biosynth_trajectory_model_metadata.json"
DEFAULT_FUSION_MODEL_PATH = PROJECT_ROOT / "models" / "biosynth_ehr_cgm_fusion_model.joblib"
DEFAULT_FUSION_METADATA_PATH = PROJECT_ROOT / "models" / "biosynth_ehr_cgm_fusion_metadata.json"

HIGH_THRESHOLD_MG_DL = 180.0
HISTORY_WINDOW_MINUTES = 60
PREDICTION_HORIZON_MINUTES = 120
MODEL_MODE_CGM_ONLY = "CGM_ONLY"
MODEL_MODE_EHR_CGM_FUSION = "EHR_CGM_FUSION"

EHR_PROFILE_FIELDS = [
    "age_years",
    "sex",
    "bmi_kg_m2",
    "hba1c_percent",
    "fasting_glucose_mg_dl",
    "family_history_diabetes",
    "hypertension_history",
    "activity_level",
]


class BioSynthTwin:
    """
    Hackathon POC runtime for a CGM-based BioSynth Digital Twin.

    The runtime uses only historical CGM observations at or before the
    prediction timestamp. Risk states and glucose states are engineering
    display states for a demo, not clinical diagnoses.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        metadata_path: str | Path = DEFAULT_METADATA_PATH,
        model_mode: str = MODEL_MODE_CGM_ONLY,
    ) -> None:
        self.model_mode = model_mode.upper()
        if self.model_mode == MODEL_MODE_EHR_CGM_FUSION and model_path == DEFAULT_MODEL_PATH:
            model_path = DEFAULT_FUSION_MODEL_PATH
            metadata_path = DEFAULT_FUSION_METADATA_PATH
        if self.model_mode not in {MODEL_MODE_CGM_ONLY, MODEL_MODE_EHR_CGM_FUSION}:
            raise ValueError(
                "model_mode must be either 'CGM_ONLY' or 'EHR_CGM_FUSION'."
            )

        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {self.model_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Model metadata not found: {self.metadata_path}")

        self.model = joblib.load(self.model_path)
        with self.metadata_path.open("r", encoding="utf-8") as handle:
            self.metadata = json.load(handle)

        self.feature_columns = list(self.metadata["feature_columns"])
        expected_feature_count = 29 if self.model_mode == MODEL_MODE_EHR_CGM_FUSION else 18
        if len(self.feature_columns) != expected_feature_count:
            raise ValueError(
                f"Expected exactly {expected_feature_count} model features for {self.model_mode}."
            )
        if self.model_mode == MODEL_MODE_EHR_CGM_FUSION:
            metadata_mode = self.metadata.get("model_mode")
            if metadata_mode != MODEL_MODE_EHR_CGM_FUSION:
                raise ValueError("Fusion runtime requires EHR_CGM_FUSION model metadata.")
            if list(self.metadata.get("ehr_feature_columns", [])) != EHR_FEATURES:
                raise ValueError("Fusion model EHR feature order does not match runtime encoder.")

    def predict_state(
        self,
        cgm_history: pd.DataFrame | list[dict[str, Any]],
        prediction_timestamp: str | pd.Timestamp | None = None,
        ehr_profile: dict[str, Any] | pd.Series | None = None,
    ) -> dict[str, Any]:
        history = self._prepare_history(cgm_history)
        if prediction_timestamp is None:
            prediction_time = history["timestamp"].max()
        else:
            prediction_time = pd.to_datetime(prediction_timestamp, errors="coerce")
            if pd.isna(prediction_time):
                raise ValueError("prediction_timestamp could not be parsed.")

        historical = history[history["timestamp"] <= prediction_time].copy()
        if historical.empty:
            raise ValueError("No CGM observations exist at or before prediction_timestamp.")

        features = self._calculate_features(historical, prediction_time)
        if self.model_mode == MODEL_MODE_EHR_CGM_FUSION:
            if ehr_profile is None:
                raise ValueError(
                    "EHR_CGM_FUSION mode requires an explicit static EHR profile. "
                    "Supply age_years, sex, bmi_kg_m2, hba1c_percent, "
                    "fasting_glucose_mg_dl, family_history_diabetes, "
                    "hypertension_history, and activity_level."
                )
            features.update(self._encode_ehr_profile(ehr_profile))

        feature_frame = pd.DataFrame([{column: features[column] for column in self.feature_columns}])
        risk = float(self.model.predict_proba(feature_frame)[:, 1][0])

        current_glucose = features["current_glucose_mg_dl"]
        return {
            "model_mode": self.model_mode,
            "prediction_timestamp": prediction_time.isoformat(sep=" "),
            "current_glucose_mg_dl": current_glucose,
            "glucose_state": self._glucose_state(current_glucose),
            "glucose_change_15m": features["glucose_change_15m"],
            "glucose_change_30m": features["glucose_change_30m"],
            "glucose_change_60m": features["glucose_change_60m"],
            "glucose_slope_per_hour": features["glucose_slope_per_hour"],
            "trend": self._trend_state(features["glucose_slope_per_hour"]),
            "hyperglycemia_risk": risk,
            "risk_state": self._risk_state(risk),
            "prediction_horizon_minutes": PREDICTION_HORIZON_MINUTES,
            "history_window_minutes": HISTORY_WINDOW_MINUTES,
            "history_observation_count": int(features["trajectory_numeric_count"]),
            "trajectory_actual_history_minutes": features["trajectory_actual_history_minutes"],
            "feature_count": len(self.feature_columns),
            "feature_order": list(self.feature_columns),
            "poc_notice": "Synthetic-data hackathon POC. Not clinically validated.",
        }

    @staticmethod
    def _prepare_history(cgm_history: pd.DataFrame | list[dict[str, Any]]) -> pd.DataFrame:
        if isinstance(cgm_history, pd.DataFrame):
            history = cgm_history.copy()
        else:
            history = pd.DataFrame(cgm_history)

        if history.empty:
            raise ValueError("CGM history is empty.")

        required = {"timestamp", "glucose_value_mg_dl"}
        missing = required - set(history.columns)
        if missing:
            raise ValueError(f"CGM history is missing columns: {sorted(missing)}")

        history["timestamp"] = pd.to_datetime(history["timestamp"], errors="coerce")
        history["glucose_numeric"] = pd.to_numeric(
            history["glucose_value_mg_dl"], errors="coerce"
        )
        history = history.dropna(subset=["timestamp"]).sort_values("timestamp")
        if history.empty:
            raise ValueError("No valid timestamps found in CGM history.")
        if history["glucose_numeric"].notna().sum() == 0:
            raise ValueError("No valid numeric glucose readings found in CGM history.")

        return history.reset_index(drop=True)

    @staticmethod
    def _calculate_features(history: pd.DataFrame, prediction_time: pd.Timestamp) -> dict[str, float]:
        window_start = prediction_time - pd.Timedelta(minutes=HISTORY_WINDOW_MINUTES)
        window = history[
            (history["timestamp"] >= window_start)
            & (history["timestamp"] <= prediction_time)
            & history["glucose_numeric"].notna()
        ].copy()

        if window.empty:
            raise ValueError("Insufficient history: no numeric readings in the 60-minute window.")

        values = window["glucose_numeric"]
        current_glucose = float(values.iloc[-1])
        mean_value = float(values.mean())
        std_value = float(values.std(ddof=1)) if len(values) >= 2 else np.nan
        min_value = float(values.min())
        max_value = float(values.max())
        high_mask = values > HIGH_THRESHOLD_MG_DL
        actual_history_minutes = float(
            (window["timestamp"].iloc[-1] - window["timestamp"].iloc[0]).total_seconds()
            / 60.0
        )

        intervals = window["timestamp"].diff().dt.total_seconds().div(60.0)
        minutes_above_180 = float(intervals.loc[(intervals > 0) & high_mask].sum())

        return {
            "current_glucose_mg_dl": current_glucose,
            "current_is_high": int(current_glucose > HIGH_THRESHOLD_MG_DL),
            "history_mean_glucose": mean_value,
            "history_median_glucose": float(values.median()),
            "history_min_glucose": min_value,
            "history_max_glucose": max_value,
            "history_range_glucose": max_value - min_value,
            "glucose_change_15m": BioSynthTwin._change_at(window, prediction_time, 15),
            "glucose_change_30m": BioSynthTwin._change_at(window, prediction_time, 30),
            "glucose_change_60m": BioSynthTwin._change_at(window, prediction_time, 60),
            "glucose_slope_per_hour": BioSynthTwin._slope_per_hour(window),
            "history_std_glucose": std_value,
            "history_cv_glucose": (
                float(std_value / mean_value)
                if pd.notna(std_value) and mean_value != 0
                else np.nan
            ),
            "history_high_count": int(high_mask.sum()),
            "history_high_fraction": float(high_mask.mean()),
            "history_minutes_above_180": minutes_above_180,
            "trajectory_numeric_count": int(len(window)),
            "trajectory_actual_history_minutes": actual_history_minutes,
        }

    @staticmethod
    def _encode_ehr_profile(ehr_profile: dict[str, Any] | pd.Series) -> dict[str, float | int]:
        if isinstance(ehr_profile, pd.Series):
            profile = ehr_profile.to_dict()
        else:
            profile = dict(ehr_profile)

        missing = [field for field in EHR_PROFILE_FIELDS if field not in profile or pd.isna(profile[field])]
        if missing:
            raise ValueError(
                "EHR_CGM_FUSION mode is missing required static EHR fields: "
                + ", ".join(missing)
            )

        sex = str(profile["sex"]).strip().lower()
        if sex not in {"female", "male"}:
            raise ValueError("EHR_CGM_FUSION sex must be 'female' or 'male'.")

        activity_level = str(profile["activity_level"]).strip().lower()
        if activity_level not in {"low", "moderate", "high"}:
            raise ValueError("EHR_CGM_FUSION activity_level must be 'low', 'moderate', or 'high'.")

        encoded = {
            "ehr_age_years": float(profile["age_years"]),
            "ehr_bmi_kg_m2": float(profile["bmi_kg_m2"]),
            "ehr_hba1c_percent": float(profile["hba1c_percent"]),
            "ehr_fasting_glucose_mg_dl": float(profile["fasting_glucose_mg_dl"]),
            "ehr_family_history_diabetes": BioSynthTwin._coerce_bool_feature(
                profile["family_history_diabetes"], "family_history_diabetes"
            ),
            "ehr_hypertension_history": BioSynthTwin._coerce_bool_feature(
                profile["hypertension_history"], "hypertension_history"
            ),
            "ehr_sex_female": int(sex == "female"),
            "ehr_sex_male": int(sex == "male"),
            "ehr_activity_low": int(activity_level == "low"),
            "ehr_activity_moderate": int(activity_level == "moderate"),
            "ehr_activity_high": int(activity_level == "high"),
        }

        if list(encoded) != EHR_FEATURES:
            raise ValueError("Runtime EHR encoding order does not match fusion feature builder.")
        return encoded

    @staticmethod
    def _coerce_bool_feature(value: Any, field_name: str) -> int:
        if isinstance(value, (bool, np.bool_)):
            return int(value)
        if isinstance(value, (int, np.integer)) and value in {0, 1}:
            return int(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "y"}:
                return 1
            if normalized in {"false", "0", "no", "n"}:
                return 0
        raise ValueError(f"EHR_CGM_FUSION {field_name} must be boolean-like.")

    @staticmethod
    def _change_at(window: pd.DataFrame, prediction_time: pd.Timestamp, minutes: int) -> float:
        current = window[window["timestamp"] <= prediction_time]
        prior = window[window["timestamp"] <= prediction_time - pd.Timedelta(minutes=minutes)]
        if current.empty or prior.empty:
            return np.nan
        return float(current.iloc[-1]["glucose_numeric"] - prior.iloc[-1]["glucose_numeric"])

    @staticmethod
    def _slope_per_hour(window: pd.DataFrame) -> float:
        if len(window) < 2:
            return np.nan
        elapsed_hours = (
            window["timestamp"] - window["timestamp"].iloc[0]
        ).dt.total_seconds().to_numpy() / 3600.0
        if elapsed_hours.max() <= 0:
            return np.nan
        return float(np.polyfit(elapsed_hours, window["glucose_numeric"].to_numpy(), 1)[0])

    @staticmethod
    def _risk_state(risk: float) -> str:
        # Demo display thresholds only; not clinically validated.
        if risk < 0.30:
            return "LOW"
        if risk < 0.60:
            return "ELEVATED"
        return "HIGH"

    @staticmethod
    def _glucose_state(glucose: float) -> str:
        # Engineering display states for the demo, not diagnostic categories.
        if glucose < 70:
            return "BELOW_RANGE"
        if glucose <= HIGH_THRESHOLD_MG_DL:
            return "IN_RANGE"
        return "ABOVE_180"

    @staticmethod
    def _trend_state(slope_per_hour: float) -> str:
        # Simple engineering trend display based on mg/dL per hour.
        if pd.isna(slope_per_hour):
            return "UNKNOWN"
        if slope_per_hour <= -30:
            return "RAPIDLY FALLING"
        if slope_per_hour <= -10:
            return "FALLING"
        if slope_per_hour < 10:
            return "STABLE"
        if slope_per_hour < 30:
            return "RISING"
        return "RAPIDLY RISING"

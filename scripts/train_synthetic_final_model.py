from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_PATH = Path("reports/synthetic_cgm_trajectory_features.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "biosynth_trajectory_model.joblib"
METADATA_PATH = MODEL_DIR / "biosynth_trajectory_model_metadata.json"

TARGET = "pre_event_target"
PERSON_ID = "person_id"

FEATURE_COLUMNS = [
    "current_glucose_mg_dl",
    "current_is_high",
    "history_mean_glucose",
    "history_median_glucose",
    "history_min_glucose",
    "history_max_glucose",
    "history_range_glucose",
    "glucose_change_15m",
    "glucose_change_30m",
    "glucose_change_60m",
    "glucose_slope_per_hour",
    "history_std_glucose",
    "history_cv_glucose",
    "history_high_count",
    "history_high_fraction",
    "history_minutes_above_180",
    "trajectory_numeric_count",
    "trajectory_actual_history_minutes",
]


def fail(message: str) -> None:
    raise RuntimeError(f"\nERROR: {message}")


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def load_training_data() -> pd.DataFrame:
    if not FEATURE_PATH.exists():
        fail(f"Synthetic trajectory feature dataset not found: {FEATURE_PATH}")

    df = pd.read_csv(FEATURE_PATH)
    required = {TARGET, PERSON_ID, *FEATURE_COLUMNS}
    missing = sorted(required - set(df.columns))
    if missing:
        fail("Training dataset is missing required columns:\n" + "\n".join(missing))

    df[TARGET] = df[TARGET].astype(bool).astype(int)
    return df


def main() -> None:
    df = load_training_data()
    model = build_model()
    model.fit(df[FEATURE_COLUMNS], df[TARGET])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    positive_rows = int(df[TARGET].sum())
    metadata = {
        "model_type": "scikit-learn Pipeline: SimpleImputer(median) -> StandardScaler -> LogisticRegression",
        "feature_columns": FEATURE_COLUMNS,
        "target": TARGET,
        "training_row_count": int(len(df)),
        "training_participant_count": int(df[PERSON_ID].nunique()),
        "positive_rows": positive_rows,
        "negative_rows": int(len(df) - positive_rows),
        "positive_rate": float(df[TARGET].mean()),
        "prediction_horizon_minutes": 120,
        "history_window_minutes": 60,
        "hyperglycemia_engineering_threshold": ">180 mg/dL",
        "model_source": "synthetic POC population",
        "reference_oof_pr_auc": 0.4433,
        "reference_oof_roc_auc": 0.8539,
        "reference_oof_note": (
            "Reference metrics came from participant-separated out-of-fold "
            "evaluation, not from this full-data fit."
        ),
        "risk_state_thresholds": {
            "LOW": "risk < 0.30",
            "ELEVATED": "0.30 <= risk < 0.60",
            "HIGH": "risk >= 0.60",
            "note": "Demo display thresholds only; not clinically validated or diagnostic.",
        },
        "clinical_validation_note": (
            "Hackathon POC model trained on synthetic data. Not clinically validated "
            "and not intended for medical decision-making."
        ),
    }

    with METADATA_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print()
    print("Synthetic final trajectory model trained")
    print("=======================================")
    print(f"Training rows: {len(df):,}")
    print(f"Training participants: {df[PERSON_ID].nunique():,}")
    print(f"Positive rows: {positive_rows:,}")
    print(f"Positive rate: {df[TARGET].mean():.2%}")
    print(f"Features: {len(FEATURE_COLUMNS)}")
    print(f"Model artifact: {MODEL_PATH}")
    print(f"Metadata: {METADATA_PATH}")
    print("Reference OOF PR-AUC: 0.4433")
    print("Reference OOF ROC-AUC: 0.8539")
    print("Note: reference metrics are participant-separated OOF results, not training-set performance.")


if __name__ == "__main__":
    main()

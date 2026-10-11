from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_PATH = Path("reports/synthetic_ehr_cgm_fusion_features.csv")
FEATURE_MANIFEST_PATH = Path("reports/synthetic_ehr_cgm_fusion_summary.json")
CV_SUMMARY_PATH = Path("reports/ehr_cgm_fusion_cross_validation_summary.json")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "biosynth_ehr_cgm_fusion_model.joblib"
METADATA_PATH = MODEL_DIR / "biosynth_ehr_cgm_fusion_metadata.json"

TARGET = "pre_event_target"
PERSON_ID = "person_id"


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


def load_feature_manifest() -> tuple[list[str], list[str], list[str]]:
    if not FEATURE_MANIFEST_PATH.exists():
        fail(f"Fusion feature manifest not found: {FEATURE_MANIFEST_PATH}")

    with FEATURE_MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    manifest = summary.get("feature_manifest", {})
    cgm_features = manifest.get("CGM_FEATURES")
    ehr_features = manifest.get("EHR_FEATURES")
    fusion_features = manifest.get("FUSION_FEATURES")
    if not isinstance(cgm_features, list):
        fail("CGM_FEATURES missing from fusion feature manifest.")
    if not isinstance(ehr_features, list):
        fail("EHR_FEATURES missing from fusion feature manifest.")
    if not isinstance(fusion_features, list):
        fail("FUSION_FEATURES missing from fusion feature manifest.")
    if fusion_features != [*cgm_features, *ehr_features]:
        fail("FUSION_FEATURES must equal ordered CGM_FEATURES + EHR_FEATURES.")
    if len(cgm_features) != 18 or len(ehr_features) != 11 or len(fusion_features) != 29:
        fail("Unexpected feature counts in fusion feature manifest.")

    return cgm_features, ehr_features, fusion_features


def load_validation_summary() -> dict[str, object]:
    if not CV_SUMMARY_PATH.exists():
        fail(f"Fusion cross-validation summary not found: {CV_SUMMARY_PATH}")
    with CV_SUMMARY_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_training_data(feature_columns: list[str]) -> pd.DataFrame:
    if not FEATURE_PATH.exists():
        fail(f"Fusion feature dataset not found: {FEATURE_PATH}")

    df = pd.read_csv(FEATURE_PATH)
    required = {TARGET, PERSON_ID, *feature_columns}
    missing = sorted(required - set(df.columns))
    if missing:
        fail("Training dataset is missing required columns:\n" + "\n".join(missing))

    if len(df) != 198000:
        fail(f"Expected 198,000 training rows; found {len(df)}")
    if df[PERSON_ID].nunique() != 100:
        fail(f"Expected 100 training participants; found {df[PERSON_ID].nunique()}")
    if df[feature_columns].isna().any().any():
        fail("Fusion training feature matrix contains missing values before model imputation.")

    df[TARGET] = df[TARGET].astype(bool).astype(int)
    return df


def main() -> None:
    cgm_features, ehr_features, fusion_features = load_feature_manifest()
    validation = load_validation_summary()
    df = load_training_data(fusion_features)

    model = build_model()
    model.fit(df[fusion_features], df[TARGET])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    positive_rows = int(df[TARGET].sum())
    fusion_metrics = validation["models"]["ehr_cgm_fusion"]["overall_oof_metrics"]
    cgm_metrics = validation["models"]["cgm_trajectory"]["overall_oof_metrics"]
    comparison = validation["comparison"]
    metadata = {
        "model_mode": "EHR_CGM_FUSION",
        "model_type": "scikit-learn Pipeline: SimpleImputer(median) -> StandardScaler -> LogisticRegression",
        "feature_columns": fusion_features,
        "cgm_feature_columns": cgm_features,
        "ehr_feature_columns": ehr_features,
        "target": TARGET,
        "target_definition": (
            "pre_event_target: whether a >180 mg/dL hyperglycemic event begins within "
            "the next 120 minutes, using only historical CGM features at prediction time."
        ),
        "training_row_count": int(len(df)),
        "training_participant_count": int(df[PERSON_ID].nunique()),
        "positive_rows": positive_rows,
        "negative_rows": int(len(df) - positive_rows),
        "positive_rate": float(df[TARGET].mean()),
        "prediction_horizon_minutes": 120,
        "history_window_minutes": 60,
        "hyperglycemia_engineering_threshold": ">180 mg/dL",
        "model_source": "synthetic BioSynth POC population with static synthetic EHR plus synthetic CGM",
        "reference_oof_metrics": {
            "ehr_cgm_fusion": fusion_metrics,
            "cgm_trajectory": cgm_metrics,
            "fusion_minus_trajectory_pr_auc": comparison["fusion_minus_trajectory_pr_auc"],
            "fusion_minus_trajectory_roc_auc": comparison["fusion_minus_trajectory_roc_auc"],
            "result_interpretation": validation["status"],
            "note": (
                "Reference metrics came from participant-separated out-of-fold evaluation, "
                "not from this full-data fit."
            ),
        },
        "risk_state_thresholds": {
            "LOW": "risk < 0.30",
            "ELEVATED": "0.30 <= risk < 0.60",
            "HIGH": "risk >= 0.60",
            "note": "Demo display thresholds only; not clinically validated or diagnostic.",
        },
        "runtime_requirements": {
            "cgm_history_columns": ["timestamp", "glucose_value_mg_dl"],
            "static_ehr_profile_fields": [
                "age_years",
                "sex",
                "bmi_kg_m2",
                "hba1c_percent",
                "fasting_glucose_mg_dl",
                "family_history_diabetes",
                "hypertension_history",
                "activity_level",
            ],
            "ehr_supplied_explicitly": True,
            "person_id_not_used_as_predictive_feature": True,
            "simulation_phenotype_not_used_as_predictive_feature": True,
        },
        "clinical_validation_note": (
            "Hackathon POC model trained on synthetic data. Synthetic EHR values were "
            "phenotype-aligned and are not genuine clinical records. This model is not "
            "clinically validated and is not intended for medical decision-making."
        ),
    }

    with METADATA_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print()
    print("Synthetic final EHR + CGM fusion model trained")
    print("==============================================")
    print(f"Training rows: {len(df):,}")
    print(f"Training participants: {df[PERSON_ID].nunique():,}")
    print(f"Positive rows: {positive_rows:,}")
    print(f"Positive rate: {df[TARGET].mean():.2%}")
    print(f"CGM features: {len(cgm_features)}")
    print(f"EHR features: {len(ehr_features)}")
    print(f"Fusion features: {len(fusion_features)}")
    print(f"Model artifact: {MODEL_PATH}")
    print(f"Metadata: {METADATA_PATH}")
    print(f"Reference OOF PR-AUC: {fusion_metrics['pr_auc']:.4f}")
    print(f"Reference OOF ROC-AUC: {fusion_metrics['roc_auc']:.4f}")
    print(f"Fusion vs CGM-only PR-AUC delta: {comparison['fusion_minus_trajectory_pr_auc']:+.4f}")
    print(f"Fusion vs CGM-only ROC-AUC delta: {comparison['fusion_minus_trajectory_roc_auc']:+.4f}")
    print(f"Interpretation: {validation['status']}")
    print("Note: reference metrics are participant-separated OOF results, not training-set performance.")


if __name__ == "__main__":
    main()

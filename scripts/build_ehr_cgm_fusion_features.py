from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CGM_FEATURE_PATH = PROJECT_ROOT / "reports" / "synthetic_cgm_trajectory_features.csv"
EHR_PATH = PROJECT_ROOT / "data" / "02_synthetic" / "ehr" / "synthetic_ehr.csv"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "synthetic_ehr_cgm_fusion_features.csv"
SUMMARY_PATH = PROJECT_ROOT / "reports" / "synthetic_ehr_cgm_fusion_summary.json"

PERSON_ID = "person_id"
TIMESTAMP = "prediction_timestamp"
TARGET = "pre_event_target"
PHENOTYPE = "simulation_phenotype"

CGM_FEATURES = [
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

ORIGINAL_EHR_COLUMNS = [
    "age_years",
    "sex",
    "bmi_kg_m2",
    "hba1c_percent",
    "fasting_glucose_mg_dl",
    "family_history_diabetes",
    "hypertension_history",
    "activity_level",
]

EHR_FEATURES = [
    "ehr_age_years",
    "ehr_bmi_kg_m2",
    "ehr_hba1c_percent",
    "ehr_fasting_glucose_mg_dl",
    "ehr_family_history_diabetes",
    "ehr_hypertension_history",
    "ehr_sex_female",
    "ehr_sex_male",
    "ehr_activity_low",
    "ehr_activity_moderate",
    "ehr_activity_high",
]

FUSION_FEATURES = [*CGM_FEATURES, *EHR_FEATURES]

FORBIDDEN_FEATURES = {
    PERSON_ID,
    TIMESTAMP,
    TARGET,
    PHENOTYPE,
    "synthetic_data_note",
    "baseline_glucose_mg_dl",
    "circadian_amplitude",
    "noise_scale",
    "meal_response_scale",
    "recovery_parameter",
    "random_seed",
    "future_max_glucose_mg_dl",
    "future_hyperglycemia",
    "future_observation_count",
    "event_status",
    "target_event_id",
    "time_to_start_time_minutes",
    "time_from_start_time_minutes",
    "prediction_class",
    "event_id",
    "event_timestamp",
    "event_type",
    "event_magnitude",
    "event_duration_parameter",
}


def fail(message: str) -> None:
    print(f"\nERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not CGM_FEATURE_PATH.exists():
        fail(f"CGM trajectory feature dataset not found: {CGM_FEATURE_PATH}")
    if not EHR_PATH.exists():
        fail(f"Synthetic EHR dataset not found: {EHR_PATH}")

    cgm = pd.read_csv(CGM_FEATURE_PATH)
    ehr = pd.read_csv(EHR_PATH)

    required_cgm = {PERSON_ID, TIMESTAMP, PHENOTYPE, TARGET, *CGM_FEATURES}
    missing_cgm = sorted(required_cgm - set(cgm.columns))
    if missing_cgm:
        fail("CGM feature dataset is missing columns:\n" + "\n".join(missing_cgm))

    required_ehr = {PERSON_ID, PHENOTYPE, *ORIGINAL_EHR_COLUMNS, "synthetic_data_note"}
    missing_ehr = sorted(required_ehr - set(ehr.columns))
    if missing_ehr:
        fail("Synthetic EHR dataset is missing columns:\n" + "\n".join(missing_ehr))

    cgm[PERSON_ID] = cgm[PERSON_ID].astype(str)
    ehr[PERSON_ID] = ehr[PERSON_ID].astype(str)

    if ehr[PERSON_ID].duplicated().any():
        fail("Synthetic EHR dataset contains duplicate person_id values.")

    return cgm.reset_index(drop=True), ehr.reset_index(drop=True)


def encode_ehr(ehr: pd.DataFrame) -> pd.DataFrame:
    encoded = ehr.copy()

    encoded["ehr_age_years"] = pd.to_numeric(encoded["age_years"], errors="coerce")
    encoded["ehr_bmi_kg_m2"] = pd.to_numeric(encoded["bmi_kg_m2"], errors="coerce")
    encoded["ehr_hba1c_percent"] = pd.to_numeric(encoded["hba1c_percent"], errors="coerce")
    encoded["ehr_fasting_glucose_mg_dl"] = pd.to_numeric(
        encoded["fasting_glucose_mg_dl"], errors="coerce"
    )
    encoded["ehr_family_history_diabetes"] = encoded["family_history_diabetes"].astype(bool).astype(int)
    encoded["ehr_hypertension_history"] = encoded["hypertension_history"].astype(bool).astype(int)

    encoded["ehr_sex_female"] = (encoded["sex"] == "female").astype(int)
    encoded["ehr_sex_male"] = (encoded["sex"] == "male").astype(int)

    encoded["ehr_activity_low"] = (encoded["activity_level"] == "low").astype(int)
    encoded["ehr_activity_moderate"] = (encoded["activity_level"] == "moderate").astype(int)
    encoded["ehr_activity_high"] = (encoded["activity_level"] == "high").astype(int)

    return encoded[[PERSON_ID, PHENOTYPE, *ORIGINAL_EHR_COLUMNS, "synthetic_data_note", *EHR_FEATURES]]


def build_fusion_dataset(cgm: pd.DataFrame, ehr: pd.DataFrame) -> pd.DataFrame:
    encoded_ehr = encode_ehr(ehr)
    fused = cgm.merge(
        encoded_ehr,
        on=PERSON_ID,
        how="left",
        suffixes=("", "_ehr"),
        validate="many_to_one",
    )

    if f"{PHENOTYPE}_ehr" in fused.columns:
        phenotype_mismatch = fused[PHENOTYPE] != fused[f"{PHENOTYPE}_ehr"]
        if phenotype_mismatch.any():
            fail("simulation_phenotype mismatch between CGM features and EHR records.")
        fused = fused.drop(columns=[f"{PHENOTYPE}_ehr"])

    output_columns = [
        PERSON_ID,
        TIMESTAMP,
        PHENOTYPE,
        TARGET,
        *CGM_FEATURES,
        *ORIGINAL_EHR_COLUMNS,
        "synthetic_data_note",
        *EHR_FEATURES,
    ]
    return fused[output_columns]


def verify(cgm: pd.DataFrame, ehr: pd.DataFrame, fused: pd.DataFrame) -> dict[str, object]:
    errors = []

    row_count_matches = len(fused) == len(cgm) == 198000
    if not row_count_matches:
        errors.append(f"Expected 198,000 rows and no row loss; got {len(fused)} fused rows")

    participant_count_matches = fused[PERSON_ID].nunique() == 100
    if not participant_count_matches:
        errors.append(f"Expected 100 participants; got {fused[PERSON_ID].nunique()}")

    no_duplicate_prediction_keys = not fused[[PERSON_ID, TIMESTAMP]].duplicated().any()
    if not no_duplicate_prediction_keys:
        errors.append("Duplicate person_id + prediction_timestamp rows found after join")

    all_rows_have_ehr = fused[EHR_FEATURES].notna().all().all()
    if not all_rows_have_ehr:
        errors.append("Some fused rows are missing encoded EHR feature values")

    target_unchanged = cgm[TARGET].reset_index(drop=True).equals(fused[TARGET].reset_index(drop=True))
    if not target_unchanged:
        errors.append("pre_event_target changed during fusion join")

    cgm_features_unchanged = all(
        np.allclose(
            pd.to_numeric(cgm[column], errors="coerce"),
            pd.to_numeric(fused[column], errors="coerce"),
            equal_nan=True,
        )
        for column in CGM_FEATURES
    )
    if not cgm_features_unchanged:
        errors.append("One or more CGM feature columns changed during fusion join")

    missing_required_ehr_values = int(fused[ORIGINAL_EHR_COLUMNS].isna().sum().sum())
    if missing_required_ehr_values:
        errors.append(f"Missing required original EHR values: {missing_required_ehr_values}")

    forbidden_in_features = sorted(set(FUSION_FEATURES) & FORBIDDEN_FEATURES)
    if forbidden_in_features:
        errors.append(f"Forbidden columns appear in FUSION_FEATURES: {forbidden_in_features}")

    non_numeric_features = [
        column for column in FUSION_FEATURES
        if not pd.api.types.is_numeric_dtype(fused[column])
    ]
    if non_numeric_features:
        errors.append(f"Non-numeric FUSION_FEATURES found: {non_numeric_features}")

    participant_sets_match = set(cgm[PERSON_ID]) == set(ehr[PERSON_ID]) == set(fused[PERSON_ID])
    if not participant_sets_match:
        errors.append("Participant ID sets do not match across CGM, EHR, and fused dataset")

    return {
        "valid": not errors,
        "errors": errors,
        "row_count_matches_expected": bool(row_count_matches),
        "participant_count_matches_expected": bool(participant_count_matches),
        "no_prediction_rows_lost_or_duplicated": bool(row_count_matches and no_duplicate_prediction_keys),
        "every_prediction_row_has_ehr": bool(all_rows_have_ehr),
        "pre_event_target_unchanged": bool(target_unchanged),
        "cgm_features_unchanged": bool(cgm_features_unchanged),
        "missing_required_ehr_values": missing_required_ehr_values,
        "forbidden_columns_in_fusion_features": forbidden_in_features,
        "fusion_features_all_numeric": not non_numeric_features,
        "non_numeric_fusion_features": non_numeric_features,
        "participant_sets_match": bool(participant_sets_match),
        "join_key": PERSON_ID,
        "join_uses_future_or_event_outcomes": False,
    }


def build_summary(fused: pd.DataFrame, validation: dict[str, object]) -> dict[str, object]:
    positive_rows = int(fused[TARGET].astype(bool).sum())
    return {
        "source_files": {
            "cgm_trajectory_features": str(CGM_FEATURE_PATH.relative_to(PROJECT_ROOT)),
            "synthetic_ehr": str(EHR_PATH.relative_to(PROJECT_ROOT)),
        },
        "output_path": str(OUTPUT_PATH.relative_to(PROJECT_ROOT)),
        "row_count": int(len(fused)),
        "participant_count": int(fused[PERSON_ID].nunique()),
        "target": {
            "name": TARGET,
            "positive_rows": positive_rows,
            "negative_rows": int(len(fused) - positive_rows),
            "positive_rate": float(fused[TARGET].astype(bool).mean()),
        },
        "feature_manifest": {
            "CGM_FEATURES": CGM_FEATURES,
            "EHR_FEATURES": EHR_FEATURES,
            "FUSION_FEATURES": FUSION_FEATURES,
            "cgm_feature_count": len(CGM_FEATURES),
            "ehr_feature_count": len(EHR_FEATURES),
            "fusion_feature_count": len(FUSION_FEATURES),
        },
        "ehr_encoding_strategy": {
            "numeric_fields": {
                "age_years": "copied to ehr_age_years",
                "bmi_kg_m2": "copied to ehr_bmi_kg_m2",
                "hba1c_percent": "copied to ehr_hba1c_percent",
                "fasting_glucose_mg_dl": "copied to ehr_fasting_glucose_mg_dl",
            },
            "boolean_fields": {
                "family_history_diabetes": "encoded as 0/1 in ehr_family_history_diabetes",
                "hypertension_history": "encoded as 0/1 in ehr_hypertension_history",
            },
            "categorical_fields": {
                "sex": "one-hot encoded as ehr_sex_female and ehr_sex_male",
                "activity_level": "one-hot encoded as ehr_activity_low, ehr_activity_moderate, ehr_activity_high",
            },
        },
        "excluded_from_predictive_features": [
            PERSON_ID,
            TIMESTAMP,
            PHENOTYPE,
            TARGET,
            "synthetic_data_note",
            "future/event-derived columns",
            "simulator ground-truth parameters",
        ],
        "validation": validation,
        "scientific_limitation": (
            "Synthetic EHR values were generated in alignment with simulator phenotypes. "
            "Any future gain from EHR+CGM fusion may reflect simulator structure rather "
            "than genuine clinical value. This dataset does not yet show that EHR fusion "
            "improves prediction."
        ),
    }


def main() -> int:
    cgm, ehr = load_inputs()
    fused = build_fusion_dataset(cgm, ehr)
    validation = verify(cgm, ehr, fused)

    if not validation["valid"]:
        fail("Fusion verification failed: " + "; ".join(validation["errors"]))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fused.to_csv(OUTPUT_PATH, index=False)

    summary = build_summary(fused, validation)
    with SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print()
    print("Synthetic EHR + CGM fusion feature dataset complete")
    print("===================================================")
    print(f"Rows: {len(fused):,}")
    print(f"Participants: {fused[PERSON_ID].nunique():,}")
    print(f"CGM features: {len(CGM_FEATURES)}")
    print(f"EHR features: {len(EHR_FEATURES)}")
    print(f"Fusion features: {len(FUSION_FEATURES)}")
    print(f"Participant match: {validation['participant_sets_match']}")
    print(f"Rows lost or duplicated: {not validation['no_prediction_rows_lost_or_duplicated']}")
    print(f"Target unchanged: {validation['pre_event_target_unchanged']}")
    print(f"CGM features unchanged: {validation['cgm_features_unchanged']}")
    print(f"All fusion features numeric: {validation['fusion_features_all_numeric']}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print("Note: EHR values are synthetic and phenotype-aligned; no fusion model has been trained yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

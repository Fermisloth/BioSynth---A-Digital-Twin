from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARTICIPANT_METADATA_PATH = (
    PROJECT_ROOT / "data" / "02_synthetic" / "metadata" / "synthetic_participants.csv"
)
EHR_DIR = PROJECT_ROOT / "data" / "02_synthetic" / "ehr"
EHR_PATH = EHR_DIR / "synthetic_ehr.csv"
REPORT_DIR = PROJECT_ROOT / "reports"
SUMMARY_PATH = REPORT_DIR / "synthetic_ehr_generation_summary.json"

MASTER_RANDOM_SEED = 20261011

REQUIRED_COLUMNS = [
    "person_id",
    "simulation_phenotype",
    "age_years",
    "sex",
    "bmi_kg_m2",
    "hba1c_percent",
    "fasting_glucose_mg_dl",
    "family_history_diabetes",
    "hypertension_history",
    "activity_level",
    "synthetic_data_note",
]


PHENOTYPE_PROFILES = {
    # Simulation phenotype alignment only. These are not clinical diagnoses.
    "No diabetes-like": {
        "age": (32, 9, 18, 62),
        "bmi": (23.8, 3.0, 18.0, 33.0),
        "hba1c": (5.3, 0.25, 4.6, 5.8),
        "fasting_glucose": (90, 8, 70, 110),
        "family_history_probability": 0.18,
        "hypertension_probability": 0.10,
        "activity_weights": [0.15, 0.45, 0.40],
    },
    "Prediabetes-like": {
        "age": (45, 10, 24, 72),
        "bmi": (29.2, 4.0, 21.0, 42.0),
        "hba1c": (6.0, 0.25, 5.7, 6.4),
        "fasting_glucose": (111, 9, 96, 128),
        "family_history_probability": 0.38,
        "hypertension_probability": 0.28,
        "activity_weights": [0.35, 0.45, 0.20],
    },
    "T2D-like": {
        "age": (56, 11, 32, 82),
        "bmi": (33.0, 5.0, 23.0, 50.0),
        "hba1c": (7.3, 0.75, 6.5, 10.2),
        "fasting_glucose": (148, 28, 126, 260),
        "family_history_probability": 0.55,
        "hypertension_probability": 0.55,
        "activity_weights": [0.55, 0.35, 0.10],
    },
}

SEX_OPTIONS = ["female", "male"]
ACTIVITY_LEVELS = ["low", "moderate", "high"]
SYNTHETIC_NOTE = (
    "Synthetic static EHR profile for BioSynth POC; not a real clinical record."
)


def fail(message: str) -> None:
    print(f"\nERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def clipped_normal(
    rng: np.random.Generator,
    mean: float,
    sd: float,
    lower: float,
    upper: float,
) -> float:
    value = rng.normal(mean, sd)
    return float(np.clip(value, lower, upper))


def load_participants() -> pd.DataFrame:
    if not PARTICIPANT_METADATA_PATH.exists():
        fail(f"Synthetic participant metadata not found: {PARTICIPANT_METADATA_PATH}")

    participants = pd.read_csv(PARTICIPANT_METADATA_PATH)
    required = {"person_id", "simulation_phenotype"}
    missing = required - set(participants.columns)
    if missing:
        fail(f"Synthetic participant metadata is missing columns: {sorted(missing)}")

    participants["person_id"] = participants["person_id"].astype(str)
    if participants["person_id"].duplicated().any():
        fail("Synthetic participant metadata contains duplicate person_id values.")

    unknown = sorted(set(participants["simulation_phenotype"]) - set(PHENOTYPE_PROFILES))
    if unknown:
        fail(f"Unknown simulation phenotypes found: {unknown}")

    return participants.sort_values("person_id").reset_index(drop=True)


def generate_profile(row: pd.Series, rng: np.random.Generator) -> dict[str, object]:
    phenotype = row["simulation_phenotype"]
    profile = PHENOTYPE_PROFILES[phenotype]

    age = round(clipped_normal(rng, *profile["age"]))
    bmi = round(clipped_normal(rng, *profile["bmi"]), 1)
    hba1c = round(clipped_normal(rng, *profile["hba1c"]), 1)
    fasting_glucose = round(clipped_normal(rng, *profile["fasting_glucose"]))

    return {
        "person_id": row["person_id"],
        "simulation_phenotype": phenotype,
        "age_years": int(age),
        "sex": str(rng.choice(SEX_OPTIONS)),
        "bmi_kg_m2": bmi,
        "hba1c_percent": hba1c,
        "fasting_glucose_mg_dl": int(fasting_glucose),
        "family_history_diabetes": bool(
            rng.random() < profile["family_history_probability"]
        ),
        "hypertension_history": bool(
            rng.random() < profile["hypertension_probability"]
        ),
        "activity_level": str(
            rng.choice(ACTIVITY_LEVELS, p=profile["activity_weights"])
        ),
        "synthetic_data_note": SYNTHETIC_NOTE,
    }


def validate_ehr(ehr: pd.DataFrame, participants: pd.DataFrame) -> dict[str, object]:
    errors = []

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(ehr.columns))
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")

    if len(ehr) != len(participants):
        errors.append(f"Expected {len(participants)} records, found {len(ehr)}")

    if ehr["person_id"].duplicated().any():
        errors.append("Duplicate person_id values found")

    participant_ids = set(participants["person_id"])
    ehr_ids = set(ehr["person_id"])
    missing_ids = sorted(participant_ids - ehr_ids)
    extra_ids = sorted(ehr_ids - participant_ids)
    if missing_ids:
        errors.append(f"Missing EHR records for participants: {missing_ids[:5]}")
    if extra_ids:
        errors.append(f"Unexpected EHR records: {extra_ids[:5]}")

    if ehr[REQUIRED_COLUMNS].isna().any().any():
        errors.append("Missing values found in required EHR fields")

    range_checks = {
        "age_years": ehr["age_years"].between(18, 90).all(),
        "bmi_kg_m2": ehr["bmi_kg_m2"].between(15, 60).all(),
        "hba1c_percent": ehr["hba1c_percent"].between(4.0, 12.0).all(),
        "fasting_glucose_mg_dl": ehr["fasting_glucose_mg_dl"].between(60, 300).all(),
        "sex": ehr["sex"].isin(SEX_OPTIONS).all(),
        "activity_level": ehr["activity_level"].isin(ACTIVITY_LEVELS).all(),
    }

    failed_ranges = sorted(name for name, passed in range_checks.items() if not passed)
    if failed_ranges:
        errors.append(f"Plausible range checks failed: {failed_ranges}")

    merged = participants[["person_id", "simulation_phenotype"]].merge(
        ehr[["person_id", "simulation_phenotype"]],
        on="person_id",
        suffixes=("_participant", "_ehr"),
        how="outer",
    )
    phenotype_match = (
        merged["simulation_phenotype_participant"]
        == merged["simulation_phenotype_ehr"]
    ).all()
    if not phenotype_match:
        errors.append("simulation_phenotype values do not match participant metadata")

    return {
        "valid": not errors,
        "errors": errors,
        "record_count": int(len(ehr)),
        "unique_person_ids": int(ehr["person_id"].nunique()),
        "participant_matching": {
            "expected_participants": int(len(participants)),
            "missing_ids": missing_ids,
            "extra_ids": extra_ids,
            "exact_match": not missing_ids and not extra_ids,
            "phenotype_match": bool(phenotype_match),
        },
        "plausible_range_checks": {k: bool(v) for k, v in range_checks.items()},
        "missing_required_values": int(ehr[REQUIRED_COLUMNS].isna().sum().sum()),
    }


def build_summary(ehr: pd.DataFrame, validation: dict[str, object]) -> dict[str, object]:
    return {
        "random_seed": MASTER_RANDOM_SEED,
        "source_participant_metadata": str(
            PARTICIPANT_METADATA_PATH.relative_to(PROJECT_ROOT)
        ),
        "output_path": str(EHR_PATH.relative_to(PROJECT_ROOT)),
        "record_count": int(len(ehr)),
        "columns": REQUIRED_COLUMNS,
        "phenotype_counts": ehr["simulation_phenotype"].value_counts().to_dict(),
        "sex_counts": ehr["sex"].value_counts().to_dict(),
        "activity_level_counts": ehr["activity_level"].value_counts().to_dict(),
        "field_ranges": {
            "age_years": {
                "min": int(ehr["age_years"].min()),
                "max": int(ehr["age_years"].max()),
                "mean": float(ehr["age_years"].mean()),
            },
            "bmi_kg_m2": {
                "min": float(ehr["bmi_kg_m2"].min()),
                "max": float(ehr["bmi_kg_m2"].max()),
                "mean": float(ehr["bmi_kg_m2"].mean()),
            },
            "hba1c_percent": {
                "min": float(ehr["hba1c_percent"].min()),
                "max": float(ehr["hba1c_percent"].max()),
                "mean": float(ehr["hba1c_percent"].mean()),
            },
            "fasting_glucose_mg_dl": {
                "min": int(ehr["fasting_glucose_mg_dl"].min()),
                "max": int(ehr["fasting_glucose_mg_dl"].max()),
                "mean": float(ehr["fasting_glucose_mg_dl"].mean()),
            },
        },
        "validation": validation,
        "generation_assumptions": [
            "One static EHR-like profile is generated for each existing synthetic participant.",
            "Values are aligned with simulation_phenotype to be internally plausible.",
            "Fields are generated from fixed random seed and phenotype profiles only.",
            "No future CGM readings, future events, labels, or model outputs are used.",
            "No EHR value is a genuine clinical record.",
        ],
        "clinical_validation_note": (
            "Synthetic static EHR data for BioSynth POC only; not real patient data "
            "and not clinically validated."
        ),
    }


def main() -> int:
    participants = load_participants()
    rng = np.random.default_rng(MASTER_RANDOM_SEED)

    ehr = pd.DataFrame(
        [generate_profile(row, rng) for _, row in participants.iterrows()]
    )[REQUIRED_COLUMNS]

    validation = validate_ehr(ehr, participants)
    if not validation["valid"]:
        fail("Synthetic EHR validation failed: " + "; ".join(validation["errors"]))

    EHR_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ehr.to_csv(EHR_PATH, index=False)

    summary = build_summary(ehr, validation)
    with SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print()
    print("Synthetic static EHR generation complete")
    print("========================================")
    print(f"Records: {len(ehr)}")
    print(f"Unique person_id values: {ehr['person_id'].nunique()}")
    print(f"Participant match: {validation['participant_matching']['exact_match']}")
    print(f"Phenotype distribution: {ehr['simulation_phenotype'].value_counts().to_dict()}")
    print(f"Age range: {ehr['age_years'].min()}-{ehr['age_years'].max()}")
    print(f"BMI range: {ehr['bmi_kg_m2'].min():.1f}-{ehr['bmi_kg_m2'].max():.1f}")
    print(f"HbA1c range: {ehr['hba1c_percent'].min():.1f}-{ehr['hba1c_percent'].max():.1f}")
    print(
        "Fasting glucose range: "
        f"{ehr['fasting_glucose_mg_dl'].min()}-{ehr['fasting_glucose_mg_dl'].max()}"
    )
    print(f"Output: {EHR_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print("Clinical note: synthetic EHR profiles only; not real patient data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

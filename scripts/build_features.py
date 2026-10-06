"""
Hall 2018 — Leakage-Safe Feature Engineering

Purpose
-------
Build an interpretable feature matrix from the supervised prediction dataset.

Primary target
--------------
pre_event_target:
    1 = a new hyperglycemic event (>180 mg/dL) begins within the next
        120 minutes.
    0 = otherwise.

Important
---------
This script ONLY uses information available at prediction time. It does not
use future glucose, event timing, or future-derived fields as predictors.

Input
-----
reports/hall2018_prediction_dataset.csv

Outputs
-------
reports/hall2018_feature_dataset.csv
reports/hall2018_feature_summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_PATH = Path("reports/hall2018_prediction_dataset.csv")
OUTPUT_PATH = Path("reports/hall2018_feature_dataset.csv")
SUMMARY_PATH = Path("reports/hall2018_feature_summary.json")

TARGET_COLUMN = "pre_event_target"

# Columns that are metadata/targets or contain future information.
# None of these may enter the model feature matrix.
FORBIDDEN_FEATURE_COLUMNS = {
    # Direct future measurements / target definitions
    "future_max_glucose_mg_dl",
    "future_hyperglycemia",
    "future_observation_count",
    "actual_future_minutes",

    # Event information derived from the future
    "event_status",
    "target_event_id",
    "time_to_start_time_minutes",
    "time_from_start_time_minutes",
    "prediction_class",

    # Configuration/context fields that should not be learned as physiology
    "lookback_minutes",
    "horizon_minutes",
    "segment_id",
}

# Required source columns.
REQUIRED_COLUMNS = {
    "person_id",
    "diabetes_type",
    "prediction_timestamp",
    "current_glucose_mg_dl",
    "current_is_high",
    TARGET_COLUMN,
    "history_observation_count",
    "actual_history_minutes",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fail(message: str) -> None:
    print(f"\nERROR: {message}")
    sys.exit(1)


def safe_numeric(series: pd.Series) -> pd.Series:
    """Convert a series to numeric without coercing categorical values into
    invented physiological values."""
    return pd.to_numeric(series, errors="coerce")


def calculate_slope(values: pd.Series, times: pd.Series) -> float:
    """
    Estimate glucose slope in mg/dL per hour using all valid observations
    available in the history.

    The supervised dataset stores summary-level windows rather than the
    complete historical glucose sequence. Therefore this function is only
    used if a compatible history representation is available. The current
    prediction CSV does not contain the complete history, so the initial
    feature set uses a conservative proxy from current-state information.
    """
    valid = values.notna() & times.notna()
    if valid.sum() < 2:
        return np.nan

    x = times[valid].astype("int64") / 3_600_000_000_000
    y = values[valid].astype(float)

    if x.max() == x.min():
        return np.nan

    return float(np.polyfit(x.to_numpy(), y.to_numpy(), 1)[0])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("Hall 2018 — Leakage-Safe Feature Engineering")
    print("=" * 72)

    if not INPUT_PATH.exists():
        fail(
            f"Input dataset not found: {INPUT_PATH}\n"
            "Run build_prediction_dataset.py first."
        )

    print(f"\nLoading: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)

    print(f"Prediction rows loaded: {len(df):,}")
    print(f"Columns loaded:         {len(df.columns)}")

    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        fail(
            "Prediction dataset is missing required columns:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )

    # -----------------------------------------------------------------------
    # Basic validation
    # -----------------------------------------------------------------------

    print("\n" + "-" * 72)
    print("SOURCE VALIDATION")
    print("-" * 72)

    df["prediction_timestamp"] = pd.to_datetime(
        df["prediction_timestamp"], errors="coerce"
    )

    if df["prediction_timestamp"].isna().any():
        fail(
            f"{df['prediction_timestamp'].isna().sum():,} prediction timestamps "
            "could not be parsed."
        )

    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")

    if df[TARGET_COLUMN].isna().any():
        fail(
            f"{df[TARGET_COLUMN].isna().sum():,} rows have missing "
            f"{TARGET_COLUMN} values."
        )

    invalid_targets = ~df[TARGET_COLUMN].isin([0, 1])
    if invalid_targets.any():
        fail(
            f"{invalid_targets.sum():,} rows contain values other than 0/1 "
            f"in {TARGET_COLUMN}."
        )

    current_glucose = safe_numeric(df["current_glucose_mg_dl"])
    history_count = safe_numeric(df["history_observation_count"])
    actual_history = safe_numeric(df["actual_history_minutes"])

    print(f"Missing current glucose:      {current_glucose.isna().sum():,}")
    print(f"Missing history count:        {history_count.isna().sum():,}")
    print(f"Missing actual history time:  {actual_history.isna().sum():,}")

    # -----------------------------------------------------------------------
    # Leakage audit
    # -----------------------------------------------------------------------

    print("\n" + "-" * 72)
    print("LEAKAGE AUDIT")
    print("-" * 72)

    present_forbidden = sorted(
        set(df.columns).intersection(FORBIDDEN_FEATURE_COLUMNS)
    )

    print("Future/event-derived columns detected in source dataset:")
    for col in present_forbidden:
        print(f"  BLOCKED: {col}")

    # These remain in the output only as audit/metadata columns if explicitly
    # retained below. They are NEVER included in feature_columns.
    print(
        "\nThese columns will not be included in the model feature matrix."
    )

    # -----------------------------------------------------------------------
    # Feature construction
    # -----------------------------------------------------------------------

    print("\n" + "-" * 72)
    print("BUILDING FEATURES")
    print("-" * 72)

    features = pd.DataFrame(index=df.index)

    # Current-state features.
    features["current_glucose_mg_dl"] = current_glucose
    features["current_is_high"] = pd.to_numeric(
        df["current_is_high"], errors="coerce"
    )

    # History availability/context.
    features["history_observation_count"] = history_count
    features["actual_history_minutes"] = actual_history

    # Conservative interpretable state features available directly from the
    # prediction dataset.
    #
    # The current supervised CSV contains the current glucose and summary
    # metadata, but not the full 60-minute glucose sequence. We therefore do
    # NOT invent mean/std/slope/min/max/range from unavailable observations.
    #
    # Those trajectory features will be added in a subsequent version by
    # deriving them directly from the raw timestamped CGM files.

    # Simple transformations of current glucose.
    features["current_glucose_squared"] = current_glucose ** 2
    features["current_glucose_above_140"] = (
        current_glucose > 140
    ).astype("int8")
    features["current_glucose_above_180"] = (
        current_glucose > 180
    ).astype("int8")

    # Preserve identifiers/labels outside the model matrix.
    output = pd.DataFrame({
        "person_id": df["person_id"].astype(str),
        "diabetes_type": df["diabetes_type"].astype(str),
        "prediction_timestamp": df["prediction_timestamp"],
        TARGET_COLUMN: df[TARGET_COLUMN].astype("int8"),
    })

    for column in features.columns:
        output[column] = features[column]

    # -----------------------------------------------------------------------
    # Feature validation
    # -----------------------------------------------------------------------

    feature_columns = list(features.columns)

    accidental_forbidden = sorted(
        set(feature_columns).intersection(FORBIDDEN_FEATURE_COLUMNS)
    )

    if accidental_forbidden:
        fail(
            "LEAKAGE DETECTED: forbidden columns entered the feature matrix:\n"
            + "\n".join(f"  - {c}" for c in accidental_forbidden)
        )

    # Check that no feature is literally copied from a known future field.
    for feature_name in feature_columns:
        if feature_name in df.columns and feature_name in FORBIDDEN_FEATURE_COLUMNS:
            fail(f"LEAKAGE DETECTED: {feature_name}")

    if output[feature_columns].shape[1] == 0:
        fail("No features were generated.")

    # -----------------------------------------------------------------------
    # Missingness and ranges
    # -----------------------------------------------------------------------

    missingness = {}
    numeric_ranges = {}

    for column in feature_columns:
        series = pd.to_numeric(output[column], errors="coerce")

        missingness[column] = int(series.isna().sum())

        valid = series.dropna()
        if len(valid):
            numeric_ranges[column] = {
                "min": float(valid.min()),
                "max": float(valid.max()),
                "mean": float(valid.mean()),
                "median": float(valid.median()),
            }
        else:
            numeric_ranges[column] = None

    # -----------------------------------------------------------------------
    # Target summary
    # -----------------------------------------------------------------------

    target_counts = (
        output[TARGET_COLUMN]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    positive_count = int(target_counts.get(1, 0))
    negative_count = int(target_counts.get(0, 0))
    total_count = len(output)

    positive_rate = (
        positive_count / total_count if total_count else 0.0
    )

    # -----------------------------------------------------------------------
    # Participant coverage
    # -----------------------------------------------------------------------

    participant_counts = (
        output.groupby("person_id")
        .size()
        .sort_values(ascending=False)
    )

    participant_positive = (
        output.groupby("person_id")[TARGET_COLUMN]
        .sum()
        .sort_values(ascending=False)
    )

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    output.to_csv(OUTPUT_PATH, index=False)

    summary = {
        "input": str(INPUT_PATH),
        "output": str(OUTPUT_PATH),
        "rows": int(len(output)),
        "participants": int(output["person_id"].nunique()),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "target": {
            "name": TARGET_COLUMN,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "positive_rate": positive_rate,
        },
        "missingness": missingness,
        "numeric_ranges": numeric_ranges,
        "participant_coverage": {
            "min_rows": int(participant_counts.min()),
            "median_rows": float(participant_counts.median()),
            "max_rows": int(participant_counts.max()),
        },
        "participant_positive_rows": {
            "min": int(participant_positive.min()),
            "median": float(participant_positive.median()),
            "max": int(participant_positive.max()),
        },
        "leakage_audit": {
            "forbidden_columns_present_in_source": present_forbidden,
            "forbidden_columns_in_feature_matrix": accidental_forbidden,
            "status": "PASS" if not accidental_forbidden else "FAIL",
        },
        "methodological_note": (
            "The current supervised prediction CSV does not contain the "
            "complete historical glucose sequence for each prediction row. "
            "Therefore this first feature set deliberately avoids inventing "
            "history statistics. Trajectory features such as mean, standard "
            "deviation, range, and slope should be derived directly from the "
            "raw timestamped CGM files in a subsequent feature-building "
            "stage."
        ),
    }

    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # -----------------------------------------------------------------------
    # Status report
    # -----------------------------------------------------------------------

    print("\n" + "-" * 72)
    print("FEATURE DATASET")
    print("-" * 72)

    print(f"Rows:                    {len(output):,}")
    print(f"Participants:            {output['person_id'].nunique():,}")
    print(f"Features:                {len(feature_columns)}")
    print(f"Positive target rows:    {positive_count:,}")
    print(f"Negative target rows:    {negative_count:,}")
    print(f"Positive target rate:    {positive_rate:.2%}")

    print("\nFeature columns:")
    for column in feature_columns:
        print(f"  - {column}")

    print("\n" + "-" * 72)
    print("LEAKAGE RESULT")
    print("-" * 72)
    print("PASS — no future/event-derived columns entered the feature matrix.")

    print("\n" + "-" * 72)
    print("OUTPUT FILES")
    print("-" * 72)
    print(OUTPUT_PATH.resolve())
    print(SUMMARY_PATH.resolve())

    print("\n" + "=" * 72)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()

"""
Build the supervised prediction dataset for Hall 2018.

Prediction task
----------------
Use the previous 60 minutes of CGM data to predict whether glucose will
exceed 180 mg/dL during the following 2 hours.

Important methodological distinction
------------------------------------
A future-positive window can occur in two different situations:

1. PRE-EVENT
   The prediction timestamp is before a hyperglycemic event begins.
   This is the more interesting case for an early-warning model.

2. EVENT-UNDERWAY
   A hyperglycemic event has already started at the prediction timestamp.
   A model performing well here may be detecting an existing event rather
   than predicting a new one.

This script preserves both cases rather than silently removing one.

The script does NOT train a model.

Outputs
-------
reports/
    hall2018_prediction_dataset.csv
    hall2018_prediction_dataset_summary.json

The raw Hall 2018 data is never modified.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "01_raw"
    / "Hall_2018"
)

GLUCOSE_DIR = DATA_DIR / "Hall_2018-extracted-glucose-files"

METADATA_FILE = DATA_DIR / "Hall_2018-metadata.csv"

EVENT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "hall2018_hyperglycemic_events.csv"
)

REPORT_DIR = PROJECT_ROOT / "reports"

OUTPUT_DATASET = (
    REPORT_DIR
    / "hall2018_prediction_dataset.csv"
)

OUTPUT_SUMMARY = (
    REPORT_DIR
    / "hall2018_prediction_dataset_summary.json"
)


LOOKBACK_MINUTES = 60
HORIZON_MINUTES = 120
HYPERGLYCEMIA_THRESHOLD = 180.0
MAX_SEGMENT_GAP_MINUTES = 15.0


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def fail(message: str) -> None:
    """Print an error message and terminate cleanly."""
    print(f"\nERROR: {message}")
    sys.exit(1)


def load_metadata() -> pd.DataFrame:
    """Load and validate participant metadata."""

    if not METADATA_FILE.exists():
        fail(f"Metadata file not found: {METADATA_FILE}")

    try:
        metadata = pd.read_csv(METADATA_FILE)
    except Exception as exc:
        fail(f"Could not read metadata: {exc}")

    required_columns = {
        "person_id",
        "diabetes_type",
    }

    missing = required_columns - set(metadata.columns)

    if missing:
        fail(
            "Metadata is missing required columns: "
            + ", ".join(sorted(missing))
        )

    metadata["person_id"] = metadata["person_id"].astype(str)

    return metadata


def load_events() -> pd.DataFrame:
    """
    Load previously detected hyperglycemic events.

    The event-analysis script is intentionally treated as the source of
    event annotations so that this script does not silently implement a
    different event definition.
    """

    if not EVENT_FILE.exists():
        fail(
            "Hyperglycemic event file was not found:\n"
            f"  {EVENT_FILE}\n\n"
            "Run analyze_hyperglycemic_events.py first."
        )

    try:
        events = pd.read_csv(EVENT_FILE)
    except Exception as exc:
        fail(f"Could not read event file: {exc}")

    required = {
        "person_id",
        "event_id",
        "start_time",
        "end_time",
        "peak_time",
        "peak_glucose_mg_dl",
    }

    missing = required - set(events.columns)

    if missing:
        fail(
            "Event file is missing required columns: "
            + ", ".join(sorted(missing))
        )

    events["person_id"] = events["person_id"].astype(str)

    events["start_time"] = pd.to_datetime(
    events["start_time"],
    errors="coerce",
)

    events["start_time"] = pd.to_datetime(
    events["start_time"],
    errors="coerce",
)

    events["end_time"] = pd.to_datetime(
        events["end_time"],
        errors="coerce",
    )

    if events["start_time"].isna().any():
        fail("Event file contains invalid start_time timestamps.")

    if events["end_time"].isna().any():
        fail("Event file contains invalid end_time timestamps.")   
    
    return events


def load_glucose_file(path: Path) -> pd.DataFrame:
    """
    Load one participant's CGM file.

    Non-numeric glucose values such as 'Low' are preserved in the raw
    loaded data but are converted to NaN in the analysis-only numeric
    column.
    """

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Could not read {path.name}: {exc}") from exc

    required = {
        "timestamp",
        "glucose_value_mg_dl",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path.name} is missing columns: "
            + ", ".join(sorted(missing))
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    if df["timestamp"].isna().any():
        raise ValueError(
            f"{path.name} contains invalid timestamps."
        )

    # IMPORTANT:
    # We do not overwrite the original glucose column.
    #
    # Values such as "Low" remain present in the original data.
    # The numeric version is used only for calculations.
    df["glucose_numeric"] = pd.to_numeric(
        df["glucose_value_mg_dl"],
        errors="coerce",
    )

    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def find_event_relationship(
    prediction_time: pd.Timestamp,
    current_glucose: Optional[float],
    person_events: pd.DataFrame,
) -> dict:
    """
    Determine the relationship between a prediction timestamp and
    detected hyperglycemic events.

    Returns:
        event_status:
            no_event
            pre_event
            event_underway

        target_event_id:
            Event responsible for the future positive label, when known.

        time_to_start_time_minutes:
            Minutes until the event begins, for pre-event predictions.

        time_from_start_time_minutes:
            Minutes since event began, for event-underway predictions.
    """

    result = {
        "event_status": "no_event",
        "target_event_id": None,
        "time_to_start_time_minutes": None,
        "time_from_start_time_minutes": None,
    }

    if person_events.empty:
        return result

    # Event already underway at prediction time.
    underway = person_events[
    (person_events["start_time"] <= prediction_time)
    & (prediction_time <= person_events["end_time"])
    ]   

    if not underway.empty:
        event = underway.sort_values("start_time").iloc[0]

        result["event_status"] = "event_underway"
        result["target_event_id"] = str(event["event_id"])

        result["time_from_start_time_minutes"] = (
        prediction_time - event["start_time"]
        ).total_seconds() / 60.0

        return result

    # Future event begins after prediction time.
    future_events = person_events[
    person_events["start_time"] > prediction_time
    ].sort_values("start_time")

    if not future_events.empty:
        event = future_events.iloc[0]

        result["event_status"] = "pre_event"
        result["target_event_id"] = str(event["event_id"])

        result["time_to_start_time_minutes"] = (
            event["start_time"] - prediction_time
        ).total_seconds() / 60.0

    return result


# ---------------------------------------------------------------------
# Main dataset construction
# ---------------------------------------------------------------------

def build_dataset() -> tuple[pd.DataFrame, dict]:
    """Build the complete supervised prediction dataset."""

    metadata = load_metadata()
    events = load_events()

    print(f"Participants in metadata: {len(metadata)}")
    print(f"Detected hyperglycemic events: {len(events)}")
    print()

    metadata_lookup = metadata.set_index("person_id")

    rows = []

    processed_files = 0
    failed_files = []

    for glucose_path in sorted(GLUCOSE_DIR.glob("*.csv")):

        person_id = glucose_path.stem

        if person_id not in metadata_lookup.index:
            print(
                f"WARNING: {person_id} has a glucose file but "
                "no metadata record. Skipping."
            )
            continue

        try:
            df = load_glucose_file(glucose_path)
        except Exception as exc:
            failed_files.append(
                {
                    "person_id": person_id,
                    "error": str(exc),
                }
            )

            print(
                f"WARNING: Failed to process {person_id}: {exc}"
            )

            continue

        processed_files += 1

        person_events = events[
            events["person_id"] == person_id
        ].copy()

        # Precompute timestamp gaps.
        df["gap_minutes"] = (
            df["timestamp"]
            .diff()
            .dt.total_seconds()
            / 60.0
        )

        # A new continuous segment begins after a gap > 15 minutes.
        df["segment_id"] = (
            df["gap_minutes"]
            .gt(MAX_SEGMENT_GAP_MINUTES)
            .fillna(False)
            .cumsum()
        )

        # Only numeric observations can be used for prediction targets.
        numeric_mask = df["glucose_numeric"].notna()

        numeric_df = df[numeric_mask].copy()

        if numeric_df.empty:
            continue

        timestamps = df["timestamp"]

        # Use each observed timestamp as a possible prediction point.
        for idx, current_row in df.iterrows():

            prediction_time = current_row["timestamp"]

            current_glucose = current_row["glucose_numeric"]

            current_segment = current_row["segment_id"]

            # ---------------------------------------------------------
            # Historical window
            # ---------------------------------------------------------

            history_start = (
                prediction_time
                - pd.Timedelta(minutes=LOOKBACK_MINUTES)
            )

            history = df[
                (df["timestamp"] >= history_start)
                & (df["timestamp"] <= prediction_time)
            ]

            # History must remain inside one continuous segment.
            if history.empty:
                continue

            if history["segment_id"].nunique() != 1:
                continue

            # Require the observed prediction timestamp to actually
            # have approximately one hour of preceding data.
            actual_history_minutes = (
                prediction_time
                - history["timestamp"].min()
            ).total_seconds() / 60.0

            if actual_history_minutes < LOOKBACK_MINUTES:
                continue

            # ---------------------------------------------------------
            # Future window
            # ---------------------------------------------------------

            future_end = (
                prediction_time
                + pd.Timedelta(minutes=HORIZON_MINUTES)
            )

            future = df[
                (df["timestamp"] > prediction_time)
                & (df["timestamp"] <= future_end)
            ]

            if future.empty:
                continue

            # Future must remain within the same continuous segment.
            if future["segment_id"].nunique() != 1:
                continue

            actual_future_minutes = (
                future["timestamp"].max()
                - prediction_time
            ).total_seconds() / 60.0

            if actual_future_minutes < HORIZON_MINUTES:
                continue

            # Only genuinely numeric glucose values can determine
            # whether the threshold is exceeded.
            future_numeric = future[
                future["glucose_numeric"].notna()
            ]

            if future_numeric.empty:
                continue

            future_max = future_numeric[
                "glucose_numeric"
            ].max()

            future_hyperglycemia = bool(
                future_max > HYPERGLYCEMIA_THRESHOLD
            )

            # ---------------------------------------------------------
            # Current state
            # ---------------------------------------------------------

            current_is_high = (
                pd.notna(current_glucose)
                and current_glucose > HYPERGLYCEMIA_THRESHOLD
            )

            # ---------------------------------------------------------
            # Event relationship
            # ---------------------------------------------------------

            relationship = find_event_relationship(
                prediction_time=prediction_time,
                current_glucose=(
                    float(current_glucose)
                    if pd.notna(current_glucose)
                    else None
                ),
                person_events=person_events,
            )

            # ---------------------------------------------------------
            # More precise predictive target
            # ---------------------------------------------------------
            #
            # A "pre_event_target" is positive only if:
            #
            #   - the prediction point is NOT already inside an event
            #   - a detected event starts within the next 2 hours
            #
            # This is deliberately stricter than future_hyperglycemia.
            # It represents an early-warning interpretation.

            pre_event_target = bool(
                future_hyperglycemia
                and relationship["event_status"] == "pre_event"
                and relationship["time_to_start_time_minutes"] <= HORIZON_MINUTES
            )

            if relationship["event_status"] == "event_underway":
                prediction_class = "event_underway"

            elif pre_event_target:
                prediction_class = "pre_event"

            elif future_hyperglycemia:
                # This should be uncommon and is retained explicitly
                # rather than silently forcing an event relationship.
                prediction_class = "future_positive_unmatched_event"

            else:
                prediction_class = "negative"

            person_meta = metadata_lookup.loc[person_id]

            rows.append(
                {
                    "person_id": person_id,
                    "diabetes_type": person_meta["diabetes_type"],
                    "prediction_timestamp": prediction_time,

                    "lookback_minutes": LOOKBACK_MINUTES,
                    "horizon_minutes": HORIZON_MINUTES,

                    "current_glucose_mg_dl": (
                        float(current_glucose)
                        if pd.notna(current_glucose)
                        else None
                    ),

                    "current_is_high": current_is_high,

                    "future_max_glucose_mg_dl": float(
                        future_max
                    ),

                    "future_hyperglycemia": future_hyperglycemia,

                    "event_status": relationship[
                        "event_status"
                    ],

                    "target_event_id": relationship[
                        "target_event_id"
                    ],

                    "time_to_start_time_minutes": relationship[
                        "time_to_start_time_minutes"
                    ],

                    "time_from_start_time_minutes": relationship[
                        "time_from_start_time_minutes"
                    ],

                    "pre_event_target": pre_event_target,

                    "prediction_class": prediction_class,

                    "history_observation_count": int(
                        len(history)
                    ),

                    "future_observation_count": int(
                        len(future_numeric)
                    ),

                    "actual_history_minutes": actual_history_minutes,

                    "actual_future_minutes": actual_future_minutes,

                    "segment_id": int(current_segment),
                }
            )

    dataset = pd.DataFrame(rows)

    if dataset.empty:
        fail(
            "No prediction windows were generated. "
            "Check the dataset paths and timestamp coverage."
        )

    # Consistent ordering makes the generated CSV easier to inspect.
    dataset = dataset.sort_values(
        ["person_id", "prediction_timestamp"]
    ).reset_index(drop=True)

    # -------------------------------------------------------------
    # Summary statistics
    # -------------------------------------------------------------

    summary = {
        "configuration": {
            "lookback_minutes": LOOKBACK_MINUTES,
            "prediction_horizon_minutes": HORIZON_MINUTES,
            "hyperglycemia_threshold_mg_dl": HYPERGLYCEMIA_THRESHOLD,
            "max_segment_gap_minutes": MAX_SEGMENT_GAP_MINUTES,
        },

        "processing": {
            "metadata_participants": int(len(metadata)),
            "glucose_files_found": int(
                len(list(GLUCOSE_DIR.glob("*.csv")))
            ),
            "glucose_files_processed": int(processed_files),
            "glucose_files_failed": int(len(failed_files)),
        },

        "dataset": {
            "rows": int(len(dataset)),
            "participants": int(
                dataset["person_id"].nunique()
            ),
        },

        "target_distribution": {
            "future_hyperglycemia": {
                "positive": int(
                    dataset["future_hyperglycemia"].sum()
                ),
                "negative": int(
                    (~dataset["future_hyperglycemia"]).sum()
                ),
                "positive_rate": float(
                    dataset["future_hyperglycemia"].mean()
                ),
            },

            "pre_event_target": {
                "positive": int(
                    dataset["pre_event_target"].sum()
                ),
                "negative": int(
                    (~dataset["pre_event_target"]).sum()
                ),
                "positive_rate": float(
                    dataset["pre_event_target"].mean()
                ),
            },
        },

        "prediction_class_counts": {
            str(k): int(v)
            for k, v in dataset[
                "prediction_class"
            ].value_counts().items()
        },

        "event_relationship_counts": {
            str(k): int(v)
            for k, v in dataset[
                "event_status"
            ].value_counts().items()
        },

        "rows_by_diabetes_type": {
            str(k): int(v)
            for k, v in dataset[
                "diabetes_type"
            ].value_counts().items()
        },

        "future_positive_rows_by_participant": {
            str(person_id): int(count)
            for person_id, count in dataset[
                dataset["future_hyperglycemia"]
            ]["person_id"].value_counts().items()
        },

        "pre_event_positive_rows_by_participant": {
            str(person_id): int(count)
            for person_id, count in dataset[
                dataset["pre_event_target"]
            ]["person_id"].value_counts().items()
        },

        "rows_per_target_event": {
            str(event_id): int(count)
            for event_id, count in dataset[
                dataset["target_event_id"].notna()
            ]["target_event_id"].value_counts().items()
        },

        "failed_files": failed_files,
    }

    return dataset, summary


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    print("=" * 72)
    print("Hall 2018 — Building supervised prediction dataset")
    print("=" * 72)
    print()

    if not GLUCOSE_DIR.exists():
        fail(f"Glucose directory not found: {GLUCOSE_DIR}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        dataset, summary = build_dataset()
    except KeyboardInterrupt:
        fail("Process interrupted by user.")
    except Exception as exc:
        fail(f"Unexpected failure: {exc}")

    # Save dataset.
    try:
        dataset.to_csv(
            OUTPUT_DATASET,
            index=False,
        )
    except Exception as exc:
        fail(f"Could not write prediction dataset: {exc}")

    # Save summary.
    try:
        with OUTPUT_SUMMARY.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                summary,
                f,
                indent=2,
            )
    except Exception as exc:
        fail(f"Could not write summary report: {exc}")

    # -------------------------------------------------------------
    # Human-readable status report
    # -------------------------------------------------------------

    print()
    print("-" * 72)
    print("DATASET BUILD COMPLETE")
    print("-" * 72)

    print(
        f"Prediction rows:              "
        f"{len(dataset):,}"
    )

    print(
        f"Participants represented:     "
        f"{dataset['person_id'].nunique():,}"
    )

    future_positive = int(
        dataset["future_hyperglycemia"].sum()
    )

    pre_event_positive = int(
        dataset["pre_event_target"].sum()
    )

    print()
    print("Future hyperglycemia target")
    print(
        f"  Positive:                   "
        f"{future_positive:,}"
    )
    print(
        f"  Negative:                   "
        f"{len(dataset) - future_positive:,}"
    )
    print(
        f"  Positive rate:              "
        f"{dataset['future_hyperglycemia'].mean():.2%}"
    )

    print()
    print("Pre-event early-warning target")
    print(
        f"  Positive:                   "
        f"{pre_event_positive:,}"
    )
    print(
        f"  Negative:                   "
        f"{len(dataset) - pre_event_positive:,}"
    )
    print(
        f"  Positive rate:              "
        f"{dataset['pre_event_target'].mean():.2%}"
    )

    print()
    print("Prediction classes:")

    for label, count in (
        dataset["prediction_class"]
        .value_counts()
        .items()
    ):
        print(
            f"  {label:<32} {count:,}"
        )

    print()
    print("Event relationships:")

    for label, count in (
        dataset["event_status"]
        .value_counts()
        .items()
    ):
        print(
            f"  {label:<32} {count:,}"
        )

    print()
    print("Rows by diabetes type:")

    diabetes_counts = (
        dataset.groupby("diabetes_type")
        .size()
        .sort_values(ascending=False)
    )

    for label, count in diabetes_counts.items():
        print(
            f"  {str(label):<25} {count:,}"
        )

    print()
    print("Output files:")
    print(f"  {OUTPUT_DATASET}")
    print(f"  {OUTPUT_SUMMARY}")

    print()
    print("=" * 72)
    print(
        "NEXT STEP: inspect the generated dataset before feature engineering "
        "or model training."
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
"""
Hall 2018 — Raw-CGM Trajectory Feature Engineering

Purpose
-------
Reconstruct the actual 60-minute CGM history for every prediction timestamp
and calculate leakage-safe trajectory features.

Primary target
--------------
pre_event_target

Input
-----
reports/hall2018_prediction_dataset.csv

Raw CGM directory
-----------------
data/01_raw/Hall_2018/Hall_2018-extracted-glucose-files

Outputs
-------
reports/hall2018_cgm_trajectory_features.csv
reports/hall2018_cgm_trajectory_features_summary.json

Important
---------
- Only observations at or before the prediction timestamp are used.
- A history window may not cross a >15-minute recording gap.
- No future/event-derived columns are used as features.
- Raw files are never modified.
- "Low" is treated as nonnumeric and is not converted to an invented
  physiological glucose value.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# Configuration
# ============================================================================

PREDICTION_PATH = Path("reports/hall2018_prediction_dataset.csv")

RAW_DIR = Path(
    "data/01_raw/Hall_2018/Hall_2018-extracted-glucose-files"
)

OUTPUT_PATH = Path(
    "reports/hall2018_cgm_trajectory_features.csv"
)

SUMMARY_PATH = Path(
    "reports/hall2018_cgm_trajectory_features_summary.json"
)

LOOKBACK_MINUTES = 60
MAX_GAP_MINUTES = 15
HIGH_THRESHOLD = 180.0

TARGET_COLUMN = "pre_event_target"

REQUIRED_PREDICTION_COLUMNS = {
    "person_id",
    "diabetes_type",
    "prediction_timestamp",
    "current_glucose_mg_dl",
    "current_is_high",
    TARGET_COLUMN,
}


# ============================================================================
# Utility functions
# ============================================================================

def fail(message: str) -> None:
    print(f"\nERROR: {message}")
    sys.exit(1)


def safe_numeric(series: pd.Series) -> pd.Series:
    """
    Convert numeric-looking glucose values to numeric.

    Values such as "Low" become NaN rather than being mapped to an
    invented glucose concentration.
    """
    return pd.to_numeric(series, errors="coerce")


def find_person_file(
    person_id: str,
    files_by_stem: dict[str, Path],
) -> Path | None:
    """
    Locate the raw CGM file corresponding to a participant.
    """

    person_id = str(person_id)

    # Exact filename stem match.
    if person_id in files_by_stem:
        return files_by_stem[person_id]

    # Controlled fallback for filenames with suffixes.
    matches = [
        path
        for stem, path in files_by_stem.items()
        if stem.startswith(person_id)
    ]

    if len(matches) == 1:
        return matches[0]

    return None


def contiguous_history(
    history: pd.DataFrame,
    prediction_time: pd.Timestamp,
) -> pd.DataFrame:
    """
    Return the valid 60-minute history immediately preceding a prediction.

    The history is truncated after the most recent gap > 15 minutes.
    """

    if history.empty:
        return history.copy()

    # Never allow observations after prediction time.
    history = history.loc[
        history["timestamp"] <= prediction_time
    ].copy()

    # Apply 60-minute lookback.
    lower_bound = (
        prediction_time
        - pd.Timedelta(minutes=LOOKBACK_MINUTES)
    )

    history = history.loc[
        history["timestamp"] >= lower_bound
    ].copy()

    if history.empty:
        return history

    history = (
        history
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # Calculate gaps between observations.
    gaps = (
        history["timestamp"]
        .diff()
        .dt.total_seconds()
        .div(60)
    )

    # A gap > 15 minutes starts a new valid segment.
    break_positions = np.where(
        gaps > MAX_GAP_MINUTES
    )[0]

    if len(break_positions):
        # Keep only the observations after the most recent large gap.
        history = history.iloc[
            break_positions[-1]:
        ].copy()

    return history.reset_index(drop=True)


def glucose_at_or_before(
    history: pd.DataFrame,
    target_time: pd.Timestamp,
) -> float:
    """
    Return the latest numeric glucose observation at or before target_time.
    """

    valid = history.loc[
        (history["timestamp"] <= target_time)
        & history["glucose_numeric"].notna()
    ]

    if valid.empty:
        return np.nan

    return float(
        valid.iloc[-1]["glucose_numeric"]
    )


def calculate_slope(
    timestamps: pd.Series,
    glucose: pd.Series,
) -> float:
    """
    Calculate glucose slope in mg/dL per hour using least squares.
    """

    valid = (
        timestamps.notna()
        & glucose.notna()
    )

    if valid.sum() < 2:
        return np.nan

    t = timestamps.loc[valid]
    y = glucose.loc[valid].astype(float)

    elapsed_hours = (
        t - t.iloc[0]
    ).dt.total_seconds() / 3600.0

    if elapsed_hours.max() <= 0:
        return np.nan

    return float(
        np.polyfit(
            elapsed_hours.to_numpy(),
            y.to_numpy(),
            1,
        )[0]
    )


def calculate_features(
    history: pd.DataFrame,
    prediction_time: pd.Timestamp,
) -> dict:
    """
    Calculate trajectory features from the valid historical CGM window.
    """

    numeric = history.loc[
        history["glucose_numeric"].notna()
    ].copy()

    # No usable numeric glucose observations.
    if numeric.empty:
        return {
            "history_mean_glucose": np.nan,
            "history_median_glucose": np.nan,
            "history_min_glucose": np.nan,
            "history_max_glucose": np.nan,
            "history_range_glucose": np.nan,
            "glucose_change_15m": np.nan,
            "glucose_change_30m": np.nan,
            "glucose_change_60m": np.nan,
            "glucose_slope_per_hour": np.nan,
            "history_std_glucose": np.nan,
            "history_cv_glucose": np.nan,
            "history_high_count": 0,
            "history_high_fraction": np.nan,
            "history_minutes_above_180": np.nan,
            "trajectory_numeric_count": 0,
            "trajectory_actual_history_minutes": 0.0,
        }

    values = numeric["glucose_numeric"]

    # ------------------------------------------------------------------------
    # Level
    # ------------------------------------------------------------------------

    mean_value = float(values.mean())
    median_value = float(values.median())
    min_value = float(values.min())
    max_value = float(values.max())
    range_value = max_value - min_value

    # ------------------------------------------------------------------------
    # Variability
    # ------------------------------------------------------------------------

    if len(values) >= 2:
        std_value = float(values.std(ddof=1))
    else:
        std_value = np.nan

    if (
        pd.notna(std_value)
        and mean_value != 0
    ):
        coefficient_variation = (
            std_value / mean_value
        )
    else:
        coefficient_variation = np.nan

    # ------------------------------------------------------------------------
    # Hyperglycemic exposure
    # ------------------------------------------------------------------------

    high_mask = (
        values > HIGH_THRESHOLD
    )

    high_count = int(
        high_mask.sum()
    )

    high_fraction = float(
        high_count / len(values)
    )

    # Approximate observed time above threshold.
    #
    # For each interval, the glucose value at the beginning of the interval
    # determines whether that interval contributes to time above threshold.
    #
    # Because large gaps were removed before this calculation, a gap >15 min
    # cannot contribute a large artificial duration.
    if len(numeric) >= 2:

        intervals = (
            numeric["timestamp"]
            .diff()
            .dt.total_seconds()
            .div(60)
        )

        valid_intervals = (
            intervals > 0
        )

        high_intervals = (
            valid_intervals
            & high_mask
        )

        minutes_above = float(
            intervals.loc[
                high_intervals
            ].sum()
        )

    else:
        minutes_above = 0.0

    # ------------------------------------------------------------------------
    # Trajectory changes
    # ------------------------------------------------------------------------

    latest_glucose = float(
        values.iloc[-1]
    )

    changes = {}

    for minutes in (
        15,
        30,
        60,
    ):

        target_time = (
            prediction_time
            - pd.Timedelta(minutes=minutes)
        )

        earlier_glucose = glucose_at_or_before(
            numeric,
            target_time,
        )

        if pd.isna(earlier_glucose):

            changes[
                f"glucose_change_{minutes}m"
            ] = np.nan

        else:

            changes[
                f"glucose_change_{minutes}m"
            ] = (
                latest_glucose
                - earlier_glucose
            )

    # ------------------------------------------------------------------------
    # Actual usable history duration
    # ------------------------------------------------------------------------

    actual_history_minutes = float(
        (
            numeric["timestamp"].iloc[-1]
            - numeric["timestamp"].iloc[0]
        ).total_seconds()
        / 60.0
    )

    # ------------------------------------------------------------------------
    # Return
    # ------------------------------------------------------------------------

    return {
        "history_mean_glucose": mean_value,
        "history_median_glucose": median_value,
        "history_min_glucose": min_value,
        "history_max_glucose": max_value,
        "history_range_glucose": range_value,

        **changes,

        "glucose_slope_per_hour": calculate_slope(
            numeric["timestamp"],
            values,
        ),

        "history_std_glucose": std_value,
        "history_cv_glucose": coefficient_variation,

        "history_high_count": high_count,
        "history_high_fraction": high_fraction,
        "history_minutes_above_180": minutes_above,

        "trajectory_numeric_count": int(
            len(numeric)
        ),

        "trajectory_actual_history_minutes":
            actual_history_minutes,
    }


# ============================================================================
# Main
# ============================================================================

def main() -> None:

    print("=" * 76)
    print(
        "Hall 2018 — Raw-CGM Trajectory Feature Engineering"
    )
    print("=" * 76)

    # ------------------------------------------------------------------------
    # Check input files
    # ------------------------------------------------------------------------

    if not PREDICTION_PATH.exists():

        fail(
            f"Prediction dataset not found:\n"
            f"{PREDICTION_PATH}"
        )

    if not RAW_DIR.exists():

        fail(
            f"Raw CGM directory not found:\n"
            f"{RAW_DIR}"
        )

    # ------------------------------------------------------------------------
    # Load prediction dataset
    # ------------------------------------------------------------------------

    prediction = pd.read_csv(
        PREDICTION_PATH
    )

    print(
        f"\nPrediction rows loaded: "
        f"{len(prediction):,}"
    )

    missing_columns = sorted(
        REQUIRED_PREDICTION_COLUMNS
        - set(prediction.columns)
    )

    if missing_columns:

        fail(
            "Prediction dataset is missing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing_columns
            )
        )

    # ------------------------------------------------------------------------
    # Validate timestamps
    # ------------------------------------------------------------------------

    prediction[
        "prediction_timestamp"
    ] = pd.to_datetime(
        prediction[
            "prediction_timestamp"
        ],
        errors="coerce",
    )

    if prediction[
        "prediction_timestamp"
    ].isna().any():

        fail(
            "One or more prediction timestamps "
            "could not be parsed."
        )

    # ------------------------------------------------------------------------
    # Validate target
    # ------------------------------------------------------------------------

    prediction[
        TARGET_COLUMN
    ] = pd.to_numeric(
        prediction[TARGET_COLUMN],
        errors="coerce",
    )

    if prediction[
        TARGET_COLUMN
    ].isna().any():

        fail(
            f"Missing {TARGET_COLUMN} values found."
        )

    if not prediction[
        TARGET_COLUMN
    ].isin([0, 1]).all():

        fail(
            f"{TARGET_COLUMN} contains values other than 0/1."
        )

    # ------------------------------------------------------------------------
    # Discover raw participant files
    # ------------------------------------------------------------------------

    print("\n" + "-" * 76)
    print("RAW CGM FILE DISCOVERY")
    print("-" * 76)

    raw_files = sorted(
        RAW_DIR.glob("*.csv")
    )

    if not raw_files:

        fail(
            f"No CSV files found in:\n{RAW_DIR}"
        )

    files_by_stem = {
        path.stem: path
        for path in raw_files
    }

    participant_ids = (
        prediction[
            "person_id"
        ]
        .astype(str)
        .unique()
    )

    missing_files = []

    for person_id in participant_ids:

        path = find_person_file(
            person_id,
            files_by_stem,
        )

        if path is None:
            missing_files.append(
                person_id
            )

    print(
        f"Raw CSV files found:             "
        f"{len(raw_files):,}"
    )

    print(
        f"Participants in prediction set: "
        f"{len(participant_ids):,}"
    )

    print(
        f"Missing participant files:       "
        f"{len(missing_files):,}"
    )

    if missing_files:

        print(
            "\nMissing participant IDs:"
        )

        for person_id in missing_files:
            print(
                f"  - {person_id}"
            )

        fail(
            "Every prediction participant "
            "must have a raw CGM file."
        )

    # ------------------------------------------------------------------------
    # Load raw CGM data into memory
    # ------------------------------------------------------------------------

    print("\n" + "-" * 76)
    print("LOADING RAW CGM DATA")
    print("-" * 76)

    cgm_cache = {}

    for index, person_id in enumerate(
        participant_ids,
        start=1,
    ):

        path = find_person_file(
            person_id,
            files_by_stem,
        )

        try:

            raw = pd.read_csv(
                path
            )

        except Exception as exc:

            fail(
                f"Could not read {path}:\n"
                f"{exc}"
            )

        required_raw_columns = {
            "timestamp",
            "glucose_value_mg_dl",
        }

        missing_raw_columns = (
            required_raw_columns
            - set(raw.columns)
        )

        if missing_raw_columns:

            fail(
                f"{path.name} is missing required columns:\n"
                + "\n".join(
                    f"  - {column}"
                    for column in sorted(
                        missing_raw_columns
                    )
                )
            )

        # Parse timestamps.
        raw[
            "timestamp"
        ] = pd.to_datetime(
            raw["timestamp"],
            errors="coerce",
        )

        if raw[
            "timestamp"
        ].isna().any():

            fail(
                f"{path.name} contains invalid timestamps."
            )

        # Preserve original glucose values and create a numeric-only copy.
        raw[
            "glucose_numeric"
        ] = safe_numeric(
            raw[
                "glucose_value_mg_dl"
            ]
        )

        raw = (
            raw
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        # Defensive duplicate check.
        if raw[
            "timestamp"
        ].duplicated().any():

            fail(
                f"{path.name} contains duplicate timestamps."
            )

        cgm_cache[
            str(person_id)
        ] = raw

        if (
            index == 1
            or index % 10 == 0
            or index == len(participant_ids)
        ):

            print(
                f"Loaded {index:>2}/"
                f"{len(participant_ids)} "
                f"participants..."
            )

    # ------------------------------------------------------------------------
    # Build trajectory features
    # ------------------------------------------------------------------------

    print("\n" + "-" * 76)
    print("BUILDING TRAJECTORY FEATURES")
    print("-" * 76)

    feature_rows = []

    missing_history_rows = 0
    numeric_empty_rows = 0
    leakage_violations = 0

    total_rows = len(
        prediction
    )

    # Process participant by participant.
    for person_id, group in prediction.groupby(
        prediction[
            "person_id"
        ].astype(str),
        sort=False,
    ):

        raw = cgm_cache[
            str(person_id)
        ]

        group = group.sort_values(
            "prediction_timestamp"
        )

        for row in group.itertuples(
            index=False
        ):

            prediction_time = (
                row.prediction_timestamp
            )

            # Candidate history.
            history = raw.loc[
                raw["timestamp"]
                <= prediction_time
            ].copy()

            # Apply lookback and gap rule.
            history = contiguous_history(
                history=history,
                prediction_time=prediction_time,
            )

            if history.empty:

                missing_history_rows += 1

            numeric_history = history.loc[
                history[
                    "glucose_numeric"
                ].notna()
            ]

            if numeric_history.empty:

                numeric_empty_rows += 1

            # Explicit future-timestamp leakage check.
            if not history.empty:

                if (
                    history["timestamp"]
                    > prediction_time
                ).any():

                    leakage_violations += 1

                    fail(
                        "LEAKAGE DETECTED: "
                        "historical window contains "
                        "a timestamp after "
                        "prediction_timestamp."
                    )

            features = calculate_features(
                history=history,
                prediction_time=prediction_time,
            )

            output_row = {

                "person_id":
                    str(row.person_id),

                "diabetes_type":
                    str(row.diabetes_type),

                "prediction_timestamp":
                    prediction_time,

                TARGET_COLUMN:
                    int(
                        getattr(
                            row,
                            TARGET_COLUMN,
                        )
                    ),

                "current_glucose_mg_dl":
                    pd.to_numeric(
                        getattr(
                            row,
                            "current_glucose_mg_dl",
                        ),
                        errors="coerce",
                    ),

                "current_is_high":
                    pd.to_numeric(
                        getattr(
                            row,
                            "current_is_high",
                        ),
                        errors="coerce",
                    ),

                **features,
            }

            feature_rows.append(
                output_row
            )

        processed = len(
            feature_rows
        )

        if (
            processed == total_rows
            or processed % 10000 < len(group)
        ):

            print(
                f"Processed "
                f"{processed:,}/"
                f"{total_rows:,} "
                f"prediction rows..."
            )

    # ------------------------------------------------------------------------
    # Create output dataframe
    # ------------------------------------------------------------------------

    output = pd.DataFrame(
        feature_rows
    )

    if len(output) != total_rows:

        fail(
            f"Output row count mismatch: "
            f"expected {total_rows:,}, "
            f"got {len(output):,}"
        )

    # ------------------------------------------------------------------------
    # Feature list
    # ------------------------------------------------------------------------

    feature_columns = [

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

    # ------------------------------------------------------------------------
    # Leakage audit
    # ------------------------------------------------------------------------

    accidental_forbidden = sorted(
        set(feature_columns)
        & {
            "future_max_glucose_mg_dl",
            "future_hyperglycemia",
            "future_observation_count",
            "actual_future_minutes",
            "event_status",
            "target_event_id",
            "time_to_start_time_minutes",
            "time_from_start_time_minutes",
            "prediction_class",
        }
    )

    if accidental_forbidden:

        fail(
            "LEAKAGE DETECTED: "
            "forbidden columns entered "
            "the feature matrix:\n"
            + "\n".join(
                f"  - {column}"
                for column in accidental_forbidden
            )
        )

    # ------------------------------------------------------------------------
    # Current glucose consistency check
    # ------------------------------------------------------------------------

    current_disagreement_count = 0

    for person_id, group in output.groupby(
        "person_id"
    ):

        raw = cgm_cache[
            person_id
        ]

        for row in group.itertuples(
            index=False
        ):

            latest = raw.loc[
                (
                    raw["timestamp"]
                    <= row.prediction_timestamp
                )
                &
                raw[
                    "glucose_numeric"
                ].notna()
            ]

            if latest.empty:
                continue

            latest_value = float(
                latest.iloc[-1][
                    "glucose_numeric"
                ]
            )

            current_value = (
                row.current_glucose_mg_dl
            )

            if pd.notna(
                current_value
            ):

                if not np.isclose(
                    float(current_value),
                    latest_value,
                    atol=1e-9,
                ):

                    current_disagreement_count += 1

    # ------------------------------------------------------------------------
    # Missingness and ranges
    # ------------------------------------------------------------------------

    missingness = {}
    numeric_ranges = {}

    for column in feature_columns:

        numeric = pd.to_numeric(
            output[column],
            errors="coerce",
        )

        missingness[
            column
        ] = int(
            numeric.isna().sum()
        )

        valid = numeric.dropna()

        if len(valid):

            numeric_ranges[
                column
            ] = {

                "min":
                    float(valid.min()),

                "max":
                    float(valid.max()),

                "mean":
                    float(valid.mean()),

                "median":
                    float(valid.median()),
            }

        else:

            numeric_ranges[
                column
            ] = None

    # ------------------------------------------------------------------------
    # Impossible-value checks
    # ------------------------------------------------------------------------

    invalid_checks = {}

    invalid_checks[
        "negative_current_glucose"
    ] = int(
        (
            output[
                "current_glucose_mg_dl"
            ] < 0
        )
        .fillna(False)
        .sum()
    )

    invalid_checks[
        "negative_history_std"
    ] = int(
        (
            output[
                "history_std_glucose"
            ] < 0
        )
        .fillna(False)
        .sum()
    )

    invalid_checks[
        "negative_history_range"
    ] = int(
        (
            output[
                "history_range_glucose"
            ] < 0
        )
        .fillna(False)
        .sum()
    )

    invalid_checks[
        "negative_numeric_count"
    ] = int(
        (
            output[
                "trajectory_numeric_count"
            ] < 0
        )
        .fillna(False)
        .sum()
    )

    invalid_checks[
        "negative_high_count"
    ] = int(
        (
            output[
                "history_high_count"
            ] < 0
        )
        .fillna(False)
        .sum()
    )

    invalid_checks[
        "high_fraction_outside_0_1"
    ] = int(
        (
            (
                output[
                    "history_high_fraction"
                ] < 0
            )
            |
            (
                output[
                    "history_high_fraction"
                ] > 1
            )
        )
        .fillna(False)
        .sum()
    )

    # ------------------------------------------------------------------------
    # Target summary
    # ------------------------------------------------------------------------

    positive = int(
        output[
            TARGET_COLUMN
        ].sum()
    )

    negative = int(
        len(output)
        - positive
    )

    positive_rate = (
        positive / len(output)
        if len(output)
        else 0.0
    )

    # ------------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    summary = {

        "input_prediction_dataset":
            str(PREDICTION_PATH),

        "raw_cgm_directory":
            str(RAW_DIR),

        "output":
            str(OUTPUT_PATH),

        "rows":
            int(len(output)),

        "participants":
            int(
                output[
                    "person_id"
                ].nunique()
            ),

        "lookback_minutes":
            LOOKBACK_MINUTES,

        "maximum_gap_minutes":
            MAX_GAP_MINUTES,

        "high_threshold_mg_dl":
            HIGH_THRESHOLD,

        "target": {

            "name":
                TARGET_COLUMN,

            "positive":
                positive,

            "negative":
                negative,

            "positive_rate":
                positive_rate,
        },

        "feature_columns":
            feature_columns,

        "feature_count":
            len(feature_columns),

        "missingness":
            missingness,

        "numeric_ranges":
            numeric_ranges,

        "invalid_value_checks":
            invalid_checks,

        "history_quality": {

            "rows_without_any_history":
                missing_history_rows,

            "rows_without_numeric_history":
                numeric_empty_rows,

            "rows_with_current_glucose_disagreement":
                current_disagreement_count,
        },

        "leakage_audit": {

            "future_timestamp_violations":
                leakage_violations,

            "forbidden_columns_in_feature_matrix":
                accidental_forbidden,

            "status":
                (
                    "PASS"
                    if (
                        leakage_violations == 0
                        and not accidental_forbidden
                    )
                    else
                    "FAIL"
                ),
        },

        "methodology_notes": [

            "Only raw CGM observations at or before "
            "each prediction timestamp are used.",

            "A history window cannot cross a gap "
            "greater than 15 minutes.",

            "Categorical 'Low' readings remain "
            "nonnumeric and are excluded from numeric "
            "trajectory statistics.",

            "No future/event-derived prediction "
            "columns are used as features.",

            "Time above 180 mg/dL is approximated "
            "from observed intervals and is not "
            "inferred across missing periods.",

            "Trajectory features are calculated "
            "directly from raw timestamped CGM "
            "observations.",
        ],
    }

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            summary,
            handle,
            indent=2,
        )

    # ------------------------------------------------------------------------
    # Final status
    # ------------------------------------------------------------------------

    print("\n" + "-" * 76)
    print("TRAJECTORY FEATURE DATASET")
    print("-" * 76)

    print(
        f"Rows:                         "
        f"{len(output):,}"
    )

    print(
        f"Participants:                 "
        f"{output['person_id'].nunique():,}"
    )

    print(
        f"Features:                     "
        f"{len(feature_columns)}"
    )

    print(
        f"Positive target rows:         "
        f"{positive:,}"
    )

    print(
        f"Negative target rows:         "
        f"{negative:,}"
    )

    print(
        f"Positive target rate:         "
        f"{positive_rate:.2%}"
    )

    print(
        f"Rows without history:         "
        f"{missing_history_rows:,}"
    )

    print(
        f"Rows without numeric history: "
        f"{numeric_empty_rows:,}"
    )

    print(
        "Current-glucose disagreements: "
        f"{current_disagreement_count:,}"
    )

    print("\nFeature columns:")

    for column in feature_columns:
        print(
            f"  - {column}"
        )

    print("\n" + "-" * 76)
    print("LEAKAGE RESULT")
    print("-" * 76)

    if (
        leakage_violations == 0
        and not accidental_forbidden
    ):

        print(
            "PASS — no future timestamps or "
            "forbidden event fields entered."
        )

    else:

        print(
            "FAIL — leakage or forbidden "
            "fields were detected."
        )

    print("\n" + "-" * 76)
    print("OUTPUT FILES")
    print("-" * 76)

    print(
        OUTPUT_PATH.resolve()
    )

    print(
        SUMMARY_PATH.resolve()
    )

    print("\n" + "=" * 76)
    print(
        "RAW-CGM TRAJECTORY FEATURE "
        "ENGINEERING COMPLETE"
    )
    print("=" * 76)


if __name__ == "__main__":
    main()
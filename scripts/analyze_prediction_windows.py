from pathlib import Path
import json
import sys

import pandas as pd


# ============================================================
# Configuration
# ============================================================

# Project root:
# scripts/analyze_prediction_windows.py
#        ↑
# parents[1] = project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "01_raw"
    / "Hall_2018"
)

GLUCOSE_DIR = (
    DATASET_DIR
    / "Hall_2018-extracted-glucose-files"
)

METADATA_FILE = (
    DATASET_DIR
    / "Hall_2018-metadata.csv"
)

REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# Prediction configuration
# ------------------------------------------------------------

# We are testing multiple candidate history lengths.
#
# These are not final modeling decisions.
# The purpose of this program is to compare them.
LOOKBACK_MINUTES = [30, 60, 120]

# Future prediction horizon.
PREDICTION_HORIZON_MINUTES = 120

# Working hyperglycemia threshold.
#
# A window is considered positive if the maximum numeric
# glucose value during the future horizon is > 180 mg/dL.
GLUCOSE_THRESHOLD = 180

# A gap larger than this means the CGM recording has entered
# a new continuous recording segment.
#
# IMPORTANT:
# This prevents us from treating a multi-hour/month-long
# recording gap as ordinary missing 5-minute measurements.
SEGMENT_GAP_MINUTES = 15


# ------------------------------------------------------------
# Output files
# ------------------------------------------------------------

WINDOW_CSV = (
    REPORT_DIR
    / "hall2018_window_analysis.csv"
)

WINDOW_JSON = (
    REPORT_DIR
    / "hall2018_window_analysis.json"
)


# ============================================================
# Utility functions
# ============================================================

def info(message):
    """Print an informational message."""
    print(f"[INFO] {message}")


def warning(message):
    """Print a warning without stopping the program."""
    print(f"[WARN] {message}")


def error(message):
    """Print an error without stopping the entire analysis."""
    print(f"[ERROR] {message}")


# ============================================================
# Validate project structure
# ============================================================

print()
print("=" * 70)
print("Hall 2018 Prediction Window Analysis")
print("=" * 70)
print()

info(f"Project root: {PROJECT_ROOT}")
info(f"Dataset directory: {DATASET_DIR}")
print()


# ------------------------------------------------------------
# Check required paths before doing any analysis.
# ------------------------------------------------------------

if not DATASET_DIR.exists():
    error(f"Dataset directory does not exist: {DATASET_DIR}")
    sys.exit(1)

if not GLUCOSE_DIR.exists():
    error(f"Glucose directory does not exist: {GLUCOSE_DIR}")
    sys.exit(1)

if not METADATA_FILE.exists():
    error(f"Metadata file does not exist: {METADATA_FILE}")
    sys.exit(1)


# ============================================================
# Load metadata
# ============================================================

info("Loading metadata...")

try:
    metadata = pd.read_csv(METADATA_FILE)

except Exception as exc:
    error(f"Could not read metadata: {exc}")
    sys.exit(1)


required_metadata_columns = {
    "person_id",
    "diabetes_type",
}

missing_metadata_columns = (
    required_metadata_columns
    - set(metadata.columns)
)

if missing_metadata_columns:
    error(
        "Metadata is missing required columns: "
        f"{sorted(missing_metadata_columns)}"
    )
    sys.exit(1)


# Normalize person IDs to strings.
metadata["person_id"] = (
    metadata["person_id"]
    .astype(str)
    .str.strip()
)


# ============================================================
# Discover glucose files
# ============================================================

glucose_files = sorted(
    GLUCOSE_DIR.glob("*.csv")
)

if not glucose_files:
    error(
        f"No glucose CSV files found in {GLUCOSE_DIR}"
    )
    sys.exit(1)

info(
    f"Metadata participants: {len(metadata)}"
)

info(
    f"Glucose files found: {len(glucose_files)}"
)

print()


# ============================================================
# Analysis function
# ============================================================

def analyze_participant(
    person_id,
    diabetes_type,
    file_path,
):
    """
    Analyze one participant.

    Returns a list of dictionaries, one for each combination
    of:

        lookback duration
        participant

    The function deliberately catches errors internally so
    that one problematic participant cannot terminate the
    entire dataset analysis.
    """

    results = []

    try:

        # ----------------------------------------------------
        # Read participant data
        # ----------------------------------------------------

        df = pd.read_csv(file_path)

        required_columns = {
            "timestamp",
            "glucose_value_mg_dl",
        }

        missing_columns = (
            required_columns
            - set(df.columns)
        )

        if missing_columns:
            raise ValueError(
                "Missing required columns: "
                f"{sorted(missing_columns)}"
            )

        # ----------------------------------------------------
        # Parse timestamps.
        #
        # Invalid timestamps become NaT rather than crashing.
        # ----------------------------------------------------

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce"
        )

        invalid_timestamp_count = int(
            df["timestamp"].isna().sum()
        )

        # Remove rows with unusable timestamps from temporal
        # analysis.
        #
        # We report their count rather than silently pretending
        # they existed.
        df = df.dropna(
            subset=["timestamp"]
        ).copy()

        # ----------------------------------------------------
        # Convert glucose to numeric.
        #
        # Important:
        #
        # "Low" becomes NaN here.
        #
        # We do NOT convert it to 0 or another invented value.
        # ----------------------------------------------------

        raw_glucose = df[
            "glucose_value_mg_dl"
        ]

        numeric_glucose = pd.to_numeric(
            raw_glucose,
            errors="coerce"
        )

        non_numeric_mask = (
            numeric_glucose.isna()
            & raw_glucose.notna()
        )

        non_numeric_count = int(
            non_numeric_mask.sum()
        )

        # ----------------------------------------------------
        # Store the numeric version separately.
        # ----------------------------------------------------

        df["numeric_glucose"] = numeric_glucose

        # ----------------------------------------------------
        # Sort chronologically.
        # ----------------------------------------------------

        df = (
            df
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        # ----------------------------------------------------
        # Remove duplicate timestamps for temporal-window
        # construction.
        #
        # We do not modify the raw file.
        #
        # For this analysis, duplicate timestamps cannot define
        # separate temporal positions.
        # ----------------------------------------------------

        duplicate_timestamp_count = int(
            df["timestamp"].duplicated().sum()
        )

        df = (
            df
            .drop_duplicates(
                subset=["timestamp"],
                keep="first"
            )
            .reset_index(drop=True)
        )

        # ----------------------------------------------------
        # Calculate intervals between observations.
        # ----------------------------------------------------

        intervals = (
            df["timestamp"]
            .diff()
            .dt.total_seconds()
            / 60
        )

        # ----------------------------------------------------
        # Identify continuous recording segments.
        #
        # A gap > 15 minutes starts a new segment.
        # ----------------------------------------------------

        segment_break = (
            intervals > SEGMENT_GAP_MINUTES
        )

        segment_break = (
            segment_break
            .fillna(False)
        )

        df["segment_id"] = (
            segment_break
            .cumsum()
        )

        segment_count = int(
            df["segment_id"].nunique()
        )

        # ----------------------------------------------------
        # Analyze each continuous segment independently.
        # ----------------------------------------------------

        for lookback_minutes in LOOKBACK_MINUTES:

            # ------------------------------------------------
            # Counters for this participant/lookback.
            # ------------------------------------------------

            valid_windows = 0
            positive_windows = 0
            negative_windows = 0

            # Number of candidate timestamps examined.
            candidate_windows = 0

            # Windows rejected because there wasn't enough
            # history.
            insufficient_history = 0

            # Windows rejected because there wasn't enough
            # future data.
            insufficient_future = 0

            # Windows rejected because the required future
            # observations contained no usable numeric glucose.
            no_numeric_future = 0

            # ------------------------------------------------
            # Iterate through each continuous recording segment.
            # ------------------------------------------------

            for _, segment in df.groupby("segment_id"):

                segment = (
                    segment
                    .sort_values("timestamp")
                    .reset_index(drop=True)
                )

                if len(segment) == 0:
                    continue

                timestamps = segment["timestamp"]

                segment_start = timestamps.min()
                segment_end = timestamps.max()

                # --------------------------------------------
                # A prediction time must have enough history
                # and enough future time inside this SAME
                # continuous segment.
                # --------------------------------------------

                earliest_prediction_time = (
                    segment_start
                    + pd.Timedelta(
                        minutes=lookback_minutes
                    )
                )

                latest_prediction_time = (
                    segment_end
                    - pd.Timedelta(
                        minutes=PREDICTION_HORIZON_MINUTES
                    )
                )

                if (
                    earliest_prediction_time
                    > latest_prediction_time
                ):
                    # This segment is too short to contain
                    # even one valid prediction window.
                    continue

                # --------------------------------------------
                # Candidate prediction times.
                #
                # We use the actual observed timestamps rather
                # than generating artificial 5-minute timestamps.
                # --------------------------------------------

                candidate_rows = segment[
                    (
                        segment["timestamp"]
                        >= earliest_prediction_time
                    )
                    &
                    (
                        segment["timestamp"]
                        <= latest_prediction_time
                    )
                ]

                for _, current_row in candidate_rows.iterrows():

                    candidate_windows += 1

                    prediction_time = (
                        current_row["timestamp"]
                    )

                    # ----------------------------------------
                    # Define the past/lookback interval.
                    #
                    # Current time itself is included.
                    # ----------------------------------------

                    history_start = (
                        prediction_time
                        - pd.Timedelta(
                            minutes=lookback_minutes
                        )
                    )

                    history = segment[
                        (
                            segment["timestamp"]
                            >= history_start
                        )
                        &
                        (
                            segment["timestamp"]
                            <= prediction_time
                        )
                    ]

                    # ----------------------------------------
                    # Verify that the observed history actually
                    # spans the requested lookback.
                    # ----------------------------------------

                    if history.empty:

                        insufficient_history += 1
                        continue

                    actual_history_duration = (
                        prediction_time
                        - history["timestamp"].min()
                    ).total_seconds() / 60

                    if (
                        actual_history_duration
                        < lookback_minutes
                    ):

                        insufficient_history += 1
                        continue

                    # ----------------------------------------
                    # Define the future prediction interval.
                    #
                    # Exclude the current time.
                    # Include observations up to 2 hours ahead.
                    # ----------------------------------------

                    future_end = (
                        prediction_time
                        + pd.Timedelta(
                            minutes=PREDICTION_HORIZON_MINUTES
                        )
                    )

                    future = segment[
                        (
                            segment["timestamp"]
                            > prediction_time
                        )
                        &
                        (
                            segment["timestamp"]
                            <= future_end
                        )
                    ]

                    if future.empty:

                        insufficient_future += 1
                        continue

                    # ----------------------------------------
                    # Verify that future observations actually
                    # span the entire 2-hour horizon.
                    #
                    # This prevents us from calling a 40-minute
                    # fragment a "2-hour prediction window."
                    # ----------------------------------------

                    actual_future_duration = (
                        future["timestamp"].max()
                        - prediction_time
                    ).total_seconds() / 60

                    if (
                        actual_future_duration
                        < PREDICTION_HORIZON_MINUTES
                    ):

                        insufficient_future += 1
                        continue

                    # ----------------------------------------
                    # Only numeric future glucose can be used
                    # for the threshold target.
                    # ----------------------------------------

                    numeric_future = (
                        future["numeric_glucose"]
                        .dropna()
                    )

                    if numeric_future.empty:

                        no_numeric_future += 1
                        continue

                    # ----------------------------------------
                    # We now have a valid prediction window.
                    # ----------------------------------------

                    valid_windows += 1

                    future_max = float(
                        numeric_future.max()
                    )

                    # ----------------------------------------
                    # Binary target:
                    #
                    # Did glucose exceed 180 mg/dL at any point
                    # during the following 2 hours?
                    # ----------------------------------------

                    if future_max > GLUCOSE_THRESHOLD:

                        positive_windows += 1

                    else:

                        negative_windows += 1

            # ------------------------------------------------
            # Calculate positive rate.
            # ------------------------------------------------

            if valid_windows > 0:

                positive_rate = (
                    positive_windows
                    / valid_windows
                )

            else:

                positive_rate = None

            # ------------------------------------------------
            # Store one summary row for this participant and
            # lookback configuration.
            # ------------------------------------------------

            results.append(
                {
                    "person_id": person_id,
                    "diabetes_type": diabetes_type,
                    "lookback_minutes": lookback_minutes,
                    "prediction_horizon_minutes":
                        PREDICTION_HORIZON_MINUTES,
                    "glucose_threshold_mg_dl":
                        GLUCOSE_THRESHOLD,

                    "raw_rows": int(
                        len(df)
                    ),

                    "invalid_timestamps":
                        invalid_timestamp_count,

                    "non_numeric_glucose":
                        non_numeric_count,

                    "duplicate_timestamps":
                        duplicate_timestamp_count,

                    "recording_segments":
                        segment_count,

                    "candidate_windows":
                        candidate_windows,

                    "valid_windows":
                        valid_windows,

                    "positive_windows":
                        positive_windows,

                    "negative_windows":
                        negative_windows,

                    "positive_rate":
                        positive_rate,

                    "insufficient_history":
                        insufficient_history,

                    "insufficient_future":
                        insufficient_future,

                    "no_numeric_future":
                        no_numeric_future,

                    "status":
                        "OK",
                }
            )

    except Exception as exc:

        # ----------------------------------------------------
        # Participant-level error handling.
        #
        # The program continues with the next participant.
        # ----------------------------------------------------

        error(
            f"{person_id}: {exc}"
        )

        # Produce an explicit error result so that the final
        # report shows that this participant was attempted.
        for lookback_minutes in LOOKBACK_MINUTES:

            results.append(
                {
                    "person_id": person_id,
                    "diabetes_type": diabetes_type,
                    "lookback_minutes": lookback_minutes,
                    "prediction_horizon_minutes":
                        PREDICTION_HORIZON_MINUTES,
                    "glucose_threshold_mg_dl":
                        GLUCOSE_THRESHOLD,

                    "raw_rows": None,
                    "invalid_timestamps": None,
                    "non_numeric_glucose": None,
                    "duplicate_timestamps": None,
                    "recording_segments": None,
                    "candidate_windows": None,
                    "valid_windows": None,
                    "positive_windows": None,
                    "negative_windows": None,
                    "positive_rate": None,
                    "insufficient_history": None,
                    "insufficient_future": None,
                    "no_numeric_future": None,

                    "status": "ERROR",
                    "error_message": str(exc),
                }
            )

    return results


# ============================================================
# Run analysis
# ============================================================

print()
print("=" * 70)
print("Scanning participants")
print("=" * 70)
print()

all_results = []

metadata_lookup = (
    metadata
    .set_index("person_id")
)


for index, file_path in enumerate(
    glucose_files,
    start=1
):

    person_id = file_path.stem

    # --------------------------------------------
    # Get diabetes type from metadata.
    # --------------------------------------------

    if person_id in metadata_lookup.index:

        diabetes_type = (
            metadata_lookup
            .loc[person_id, "diabetes_type"]
        )

    else:

        warning(
            f"{person_id}: no metadata record found"
        )

        diabetes_type = "Unknown"

    print(
        f"[{index:02d}/{len(glucose_files):02d}] "
        f"Analyzing {person_id}...",
        end=" "
    )

    participant_results = analyze_participant(
        person_id=person_id,
        diabetes_type=diabetes_type,
        file_path=file_path,
    )

    all_results.extend(
        participant_results
    )

    participant_statuses = {
        result["status"]
        for result in participant_results
    }

    if "ERROR" in participant_statuses:
        print("ERROR")

    else:
        print("OK")


# ============================================================
# Create result DataFrame
# ============================================================

results_df = pd.DataFrame(
    all_results
)

if results_df.empty:

    error(
        "No analysis results were generated."
    )

    sys.exit(1)


# ============================================================
# Save CSV
# ============================================================

try:

    results_df.to_csv(
        WINDOW_CSV,
        index=False
    )

    info(
        f"CSV report written to: {WINDOW_CSV}"
    )

except Exception as exc:

    error(
        f"Could not write CSV report: {exc}"
    )


# ============================================================
# Generate summary statistics
# ============================================================

print()
print("=" * 70)
print("Prediction Window Summary")
print("=" * 70)
print()


def summarize_population(
    df,
    population_name,
):
    """
    Print summary statistics for a population.

    Population may be:
        All participants
        T2D
        Prediabetes
        No diabetes
    """

    print()
    print(
        f"--- {population_name} ---"
    )

    for lookback in LOOKBACK_MINUTES:

        subset = df[
            df["lookback_minutes"]
            == lookback
        ]

        # Only successful analyses.
        valid_subset = subset[
            subset["status"] == "OK"
        ]

        valid_windows = int(
            valid_subset["valid_windows"]
            .fillna(0)
            .sum()
        )

        positive_windows = int(
            valid_subset["positive_windows"]
            .fillna(0)
            .sum()
        )

        negative_windows = int(
            valid_subset["negative_windows"]
            .fillna(0)
            .sum()
        )

        if valid_windows > 0:

            positive_rate = (
                positive_windows
                / valid_windows
            )

        else:

            positive_rate = None

        print()
        print(
            f"Lookback: {lookback} minutes"
        )

        print(
            f"  Valid windows: "
            f"{valid_windows:,}"
        )

        print(
            f"  Positive windows: "
            f"{positive_windows:,}"
        )

        print(
            f"  Negative windows: "
            f"{negative_windows:,}"
        )

        if positive_rate is not None:

            print(
                f"  Positive rate: "
                f"{positive_rate:.2%}"
            )

        else:

            print(
                "  Positive rate: N/A"
            )


# ============================================================
# Overall population
# ============================================================

summarize_population(
    results_df,
    "All participants"
)


# ============================================================
# T2D population
# ============================================================

t2d_df = results_df[
    results_df["diabetes_type"]
    == "T2D"
]

summarize_population(
    t2d_df,
    "T2D"
)


# ============================================================
# Prediabetes population
# ============================================================

prediabetes_df = results_df[
    results_df["diabetes_type"]
    == "Prediabetes"
]

summarize_population(
    prediabetes_df,
    "Prediabetes"
)


# ============================================================
# No-diabetes population
# ============================================================

no_diabetes_df = results_df[
    results_df["diabetes_type"]
    == "No diabetes"
]

summarize_population(
    no_diabetes_df,
    "No diabetes"
)


# ============================================================
# Per-participant T2D report
# ============================================================

print()
print("=" * 70)
print("T2D Participant Breakdown")
print("=" * 70)
print()

for person_id in sorted(
    t2d_df["person_id"].unique()
):

    person = t2d_df[
        t2d_df["person_id"]
        == person_id
    ]

    print(
        f"{person_id}:"
    )

    for _, row in person.iterrows():

        lookback = int(
            row["lookback_minutes"]
        )

        valid = row["valid_windows"]

        positive = row["positive_windows"]

        negative = row["negative_windows"]

        if pd.notna(
            row["positive_rate"]
        ):

            rate = (
                f"{row['positive_rate']:.2%}"
            )

        else:

            rate = "N/A"

        print(
            f"  {lookback:>3} min → "
            f"valid={valid}, "
            f"positive={positive}, "
            f"negative={negative}, "
            f"positive_rate={rate}"
        )


# ============================================================
# Identify analysis errors
# ============================================================

error_rows = results_df[
    results_df["status"]
    == "ERROR"
]

print()
print("=" * 70)
print("Analysis Integrity")
print("=" * 70)
print()

info(
    f"Participant files attempted: "
    f"{len(glucose_files)}"
)

info(
    f"Successful participant analyses: "
    f"{len(glucose_files) - len(
        error_rows['person_id'].unique()
    )}"
)

info(
    f"Participants with analysis errors: "
    f"{len(error_rows['person_id'].unique())}"
)

if not error_rows.empty:

    warning(
        "Participants with errors:"
    )

    for person_id in sorted(
        error_rows["person_id"]
        .unique()
    ):

        print(
            f"  - {person_id}"
        )


# ============================================================
# Save JSON summary
# ============================================================

def population_summary(df):
    """
    Convert a DataFrame population into a JSON-friendly summary.
    """

    output = {}

    for lookback in LOOKBACK_MINUTES:

        subset = df[
            (
                df["lookback_minutes"]
                == lookback
            )
            &
            (
                df["status"]
                == "OK"
            )
        ]

        valid = int(
            subset["valid_windows"]
            .fillna(0)
            .sum()
        )

        positive = int(
            subset["positive_windows"]
            .fillna(0)
            .sum()
        )

        negative = int(
            subset["negative_windows"]
            .fillna(0)
            .sum()
        )

        output[str(lookback)] = {
            "valid_windows": valid,
            "positive_windows": positive,
            "negative_windows": negative,
            "positive_rate": (
                positive / valid
                if valid > 0
                else None
            ),
        }

    return output


json_report = {
    "dataset": "Hall_2018",

    "configuration": {
        "lookback_minutes":
            LOOKBACK_MINUTES,

        "prediction_horizon_minutes":
            PREDICTION_HORIZON_MINUTES,

        "glucose_threshold_mg_dl":
            GLUCOSE_THRESHOLD,

        "segment_gap_minutes":
            SEGMENT_GAP_MINUTES,
    },

    "participants": {
        "metadata_count":
            int(len(metadata)),

        "glucose_file_count":
            int(len(glucose_files)),

        "analysis_error_count":
            int(
                error_rows["person_id"]
                .nunique()
            ),
    },

    "populations": {
        "all": population_summary(
            results_df
        ),

        "T2D": population_summary(
            t2d_df
        ),

        "Prediabetes": population_summary(
            prediabetes_df
        ),

        "No diabetes": population_summary(
            no_diabetes_df
        ),
    },

    "notes": [
        "Prediction windows are constructed using actual timestamps.",
        "Windows are never allowed to cross recording gaps larger than the configured segment threshold.",
        "Non-numeric glucose values are not converted into invented numeric values.",
        "The target is positive when numeric glucose exceeds 180 mg/dL during the following 2-hour horizon.",
        "This analysis does not create a train/test split.",
        "This analysis does not modify raw data.",
    ],
}


try:

    with open(
        WINDOW_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            json_report,
            f,
            indent=2,
            default=str
        )

    info(
        f"JSON report written to: {WINDOW_JSON}"
    )

except Exception as exc:

    error(
        f"Could not write JSON report: {exc}"
    )


# ============================================================
# Final status
# ============================================================

print()
print("=" * 70)

if error_rows.empty:

    print(
        "OVERALL STATUS: ANALYSIS COMPLETE"
    )

else:

    print(
        "OVERALL STATUS: COMPLETE WITH ERRORS"
    )

print("=" * 70)

print()
print(
    "Next step: inspect the number and distribution of valid "
    "T2D prediction windows before selecting a modeling strategy."
)
print()
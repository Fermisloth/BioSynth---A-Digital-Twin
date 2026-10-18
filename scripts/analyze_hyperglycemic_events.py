from pathlib import Path
import json
import sys

import pandas as pd


# ============================================================
# Configuration
# ============================================================

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
# Hyperglycemia definition
# ------------------------------------------------------------

# An observation is considered hyperglycemic when:
#
#     glucose > 180 mg/dL
#
# Note that this is strictly greater than 180.
GLUCOSE_THRESHOLD = 180


# ------------------------------------------------------------
# Recording continuity
# ------------------------------------------------------------

# A gap greater than 15 minutes starts a new recording segment.
#
# This prevents us from treating a multi-hour or multi-day gap
# as part of one continuous glucose trajectory.
SEGMENT_GAP_MINUTES = 15


# ------------------------------------------------------------
# Event continuity
# ------------------------------------------------------------

# CGM data is approximately every 5 minutes.
#
# If there is a gap larger than this between two >180 readings,
# we do not automatically call them one continuous excursion.
#
# This is deliberately conservative.
EVENT_GAP_MINUTES = 15


# ------------------------------------------------------------
# Output files
# ------------------------------------------------------------

EVENT_CSV = (
    REPORT_DIR
    / "hall2018_hyperglycemic_events.csv"
)

EVENT_JSON = (
    REPORT_DIR
    / "hall2018_hyperglycemic_events.json"
)


# ============================================================
# Utility functions
# ============================================================

def info(message):
    print(f"[INFO] {message}")


def warning(message):
    print(f"[WARN] {message}")


def error(message):
    print(f"[ERROR] {message}")


# ============================================================
# Initial validation
# ============================================================

print()
print("=" * 70)
print("Hall 2018 Hyperglycemic Event Analysis")
print("=" * 70)
print()

info(f"Project root: {PROJECT_ROOT}")
info(f"Dataset directory: {DATASET_DIR}")
print()


if not DATASET_DIR.exists():
    error(
        f"Dataset directory does not exist: {DATASET_DIR}"
    )
    sys.exit(1)


if not GLUCOSE_DIR.exists():
    error(
        f"Glucose directory does not exist: {GLUCOSE_DIR}"
    )
    sys.exit(1)


if not METADATA_FILE.exists():
    error(
        f"Metadata file does not exist: {METADATA_FILE}"
    )
    sys.exit(1)


# ============================================================
# Load metadata
# ============================================================

info("Loading metadata...")

try:
    metadata = pd.read_csv(
        METADATA_FILE
    )

except Exception as exc:
    error(
        f"Could not read metadata: {exc}"
    )
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


metadata["person_id"] = (
    metadata["person_id"]
    .astype(str)
    .str.strip()
)


metadata_lookup = (
    metadata
    .set_index("person_id")
)


# ============================================================
# Discover glucose files
# ============================================================

glucose_files = sorted(
    GLUCOSE_DIR.glob("*.csv")
)

if not glucose_files:
    error(
        f"No glucose files found in {GLUCOSE_DIR}"
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
# Analyze one participant
# ============================================================

def analyze_participant(
    person_id,
    diabetes_type,
    file_path,
):
    """
    Identify distinct hyperglycemic excursions for one person.

    Returns:
        events
        participant_summary

    The function handles errors locally so that one bad file
    does not stop the entire dataset analysis.
    """

    events = []

    try:

        # ----------------------------------------------------
        # Load CSV
        # ----------------------------------------------------

        df = pd.read_csv(
            file_path
        )

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

        raw_rows = len(df)

        # ----------------------------------------------------
        # Parse timestamps
        # ----------------------------------------------------

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce"
        )

        invalid_timestamps = int(
            df["timestamp"].isna().sum()
        )

        df = df.dropna(
            subset=["timestamp"]
        ).copy()

        # ----------------------------------------------------
        # Convert glucose to numeric
        #
        # Examples:
        #
        # 106 -> 106
        # "Low" -> NaN
        #
        # We never convert "Low" into zero.
        # ----------------------------------------------------

        raw_glucose = (
            df["glucose_value_mg_dl"]
        )

        df["numeric_glucose"] = pd.to_numeric(
            raw_glucose,
            errors="coerce"
        )

        non_numeric_mask = (
            df["numeric_glucose"].isna()
            &
            raw_glucose.notna()
        )

        non_numeric_glucose = int(
            non_numeric_mask.sum()
        )

        # ----------------------------------------------------
        # Sort chronologically
        # ----------------------------------------------------

        df = (
            df
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        # ----------------------------------------------------
        # Remove duplicate timestamps for event analysis.
        #
        # The raw files remain untouched.
        # ----------------------------------------------------

        duplicate_timestamps = int(
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

        if df.empty:

            return (
                events,
                {
                    "person_id": person_id,
                    "diabetes_type": diabetes_type,
                    "raw_rows": raw_rows,
                    "invalid_timestamps":
                        invalid_timestamps,
                    "non_numeric_glucose":
                        non_numeric_glucose,
                    "duplicate_timestamps":
                        duplicate_timestamps,
                    "recording_segments": 0,
                    "event_count": 0,
                    "status": "OK",
                }
            )

        # ----------------------------------------------------
        # Identify continuous recording segments.
        # ----------------------------------------------------

        intervals = (
            df["timestamp"]
            .diff()
            .dt.total_seconds()
            / 60
        )

        segment_break = (
            intervals
            > SEGMENT_GAP_MINUTES
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

        # ====================================================
        # Find hyperglycemic events
        # ====================================================

        for segment_id, segment in df.groupby(
            "segment_id"
        ):

            segment = (
                segment
                .sort_values("timestamp")
                .reset_index(drop=True)
            )

            # ------------------------------------------------
            # Keep only observations above threshold.
            # ------------------------------------------------

            high = segment[
                segment["numeric_glucose"]
                > GLUCOSE_THRESHOLD
            ].copy()

            if high.empty:
                continue

            # ------------------------------------------------
            # Find gaps between consecutive high observations.
            #
            # If the gap is <= 15 minutes, we consider the
            # readings part of the same excursion.
            #
            # If the gap is > 15 minutes, a new excursion begins.
            # ------------------------------------------------

            high["interval_minutes"] = (
                high["timestamp"]
                .diff()
                .dt.total_seconds()
                / 60
            )

            high["new_event"] = (
                high["interval_minutes"]
                > EVENT_GAP_MINUTES
            )

            high["new_event"] = (
                high["new_event"]
                .fillna(True)
            )

            high["event_number"] = (
                high["new_event"]
                .cumsum()
            )

            # ------------------------------------------------
            # Build each excursion.
            # ------------------------------------------------

            for event_number, event_data in high.groupby(
                "event_number"
            ):

                event_data = (
                    event_data
                    .sort_values("timestamp")
                )

                start_time = (
                    event_data["timestamp"].min()
                )

                end_time = (
                    event_data["timestamp"].max()
                )

                duration_minutes = (
                    end_time
                    - start_time
                ).total_seconds() / 60

                peak_index = (
                    event_data[
                        "numeric_glucose"
                    ]
                    .idxmax()
                )

                peak_row = event_data.loc[
                    peak_index
                ]

                peak_glucose = float(
                    peak_row[
                        "numeric_glucose"
                    ]
                )

                peak_time = (
                    peak_row["timestamp"]
                )

                time_to_peak_minutes = (
                    peak_time
                    - start_time
                ).total_seconds() / 60

                observation_count = int(
                    len(event_data)
                )

                # ------------------------------------------------
                # Estimate whether this is a meaningful temporal
                # excursion or just a single isolated measurement.
                #
                # We DO NOT discard single-point events here.
                # We report them separately.
                # ------------------------------------------------

                if observation_count == 1:
                    event_type = "single_observation"

                else:
                    event_type = "multi_observation"

                events.append(
                    {
                        "person_id":
                            person_id,

                        "diabetes_type":
                            diabetes_type,

                        "segment_id":
                            int(segment_id),

                        "event_id":
                            f"{person_id}-"
                            f"{int(segment_id)}-"
                            f"{int(event_number)}",

                        "start_time":
                            start_time,

                        "end_time":
                            end_time,

                        "duration_minutes":
                            duration_minutes,

                        "peak_time":
                            peak_time,

                        "peak_glucose_mg_dl":
                            peak_glucose,

                        "time_to_peak_minutes":
                            time_to_peak_minutes,

                        "observations_above_threshold":
                            observation_count,

                        "event_type":
                            event_type,
                    }
                )

        participant_summary = {
            "person_id":
                person_id,

            "diabetes_type":
                diabetes_type,

            "raw_rows":
                raw_rows,

            "invalid_timestamps":
                invalid_timestamps,

            "non_numeric_glucose":
                non_numeric_glucose,

            "duplicate_timestamps":
                duplicate_timestamps,

            "recording_segments":
                segment_count,

            "event_count":
                len(
                    [
                        e
                        for e in events
                        if e["person_id"]
                        == person_id
                    ]
                ),

            "status":
                "OK",
        }

        return (
            events,
            participant_summary
        )

    except Exception as exc:

        error(
            f"{person_id}: {exc}"
        )

        return (
            [],
            {
                "person_id":
                    person_id,

                "diabetes_type":
                    diabetes_type,

                "raw_rows":
                    None,

                "invalid_timestamps":
                    None,

                "non_numeric_glucose":
                    None,

                "duplicate_timestamps":
                    None,

                "recording_segments":
                    None,

                "event_count":
                    None,

                "status":
                    "ERROR",

                "error_message":
                    str(exc),
            }
        )


# ============================================================
# Scan all participants
# ============================================================

print()
print("=" * 70)
print("Scanning participants")
print("=" * 70)
print()

all_events = []
participant_summaries = []


for index, file_path in enumerate(
    glucose_files,
    start=1
):

    person_id = file_path.stem

    if person_id in metadata_lookup.index:

        diabetes_type = (
            metadata_lookup
            .loc[
                person_id,
                "diabetes_type"
            ]
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

    events, summary = (
        analyze_participant(
            person_id,
            diabetes_type,
            file_path,
        )
    )

    all_events.extend(events)

    participant_summaries.append(
        summary
    )

    if summary["status"] == "OK":

        print(
            f"OK "
            f"({summary['event_count']} events)"
        )

    else:

        print("ERROR")


# ============================================================
# Convert results to DataFrames
# ============================================================

events_df = pd.DataFrame(
    all_events
)

participants_df = pd.DataFrame(
    participant_summaries
)


# ============================================================
# Handle no-event case
# ============================================================

if events_df.empty:

    warning(
        "No hyperglycemic events were detected."
    )

    # Create columns anyway so that the CSV remains usable.
    events_df = pd.DataFrame(
        columns=[
            "person_id",
            "diabetes_type",
            "segment_id",
            "event_id",
            "start_time",
            "end_time",
            "duration_minutes",
            "peak_time",
            "peak_glucose_mg_dl",
            "time_to_peak_minutes",
            "observations_above_threshold",
            "event_type",
        ]
    )


# ============================================================
# Save event-level CSV
# ============================================================

try:

    events_df.to_csv(
        EVENT_CSV,
        index=False
    )

    info(
        f"Event CSV written to: {EVENT_CSV}"
    )

except Exception as exc:

    error(
        f"Could not write event CSV: {exc}"
    )


# ============================================================
# Summary helpers
# ============================================================

def population_event_summary(
    events,
    population_name,
):
    """
    Print and return statistics for a population.
    """

    subset = events[
        events["diabetes_type"]
        == population_name
    ]

    event_count = len(subset)

    participant_count = (
        subset["person_id"]
        .nunique()
    )

    if event_count == 0:

        return {
            "participants_with_events":
                participant_count,

            "events":
                0,

            "mean_duration_minutes":
                None,

            "median_duration_minutes":
                None,

            "max_duration_minutes":
                None,

            "mean_peak_glucose_mg_dl":
                None,

            "max_peak_glucose_mg_dl":
                None,
        }

    return {
        "participants_with_events":
            participant_count,

        "events":
            int(event_count),

        "mean_duration_minutes":
            float(
                subset["duration_minutes"]
                .mean()
            ),

        "median_duration_minutes":
            float(
                subset["duration_minutes"]
                .median()
            ),

        "max_duration_minutes":
            float(
                subset["duration_minutes"]
                .max()
            ),

        "mean_peak_glucose_mg_dl":
            float(
                subset["peak_glucose_mg_dl"]
                .mean()
            ),

        "max_peak_glucose_mg_dl":
            float(
                subset["peak_glucose_mg_dl"]
                .max()
            ),
    }


# ============================================================
# Overall summary
# ============================================================

print()
print("=" * 70)
print("Dataset Event Summary")
print("=" * 70)
print()

total_events = len(events_df)

info(
    f"Total hyperglycemic events: "
    f"{total_events:,}"
)

info(
    "Threshold: "
    f"> {GLUCOSE_THRESHOLD} mg/dL"
)

info(
    "Event gap threshold: "
    f"{EVENT_GAP_MINUTES} minutes"
)

info(
    "Recording segment gap: "
    f"{SEGMENT_GAP_MINUTES} minutes"
)


# ============================================================
# T2D summary
# ============================================================

print()
print("=" * 70)
print("T2D Hyperglycemic Events")
print("=" * 70)
print()

t2d_events = events_df[
    events_df["diabetes_type"]
    == "T2D"
]

t2d_participants = metadata[
    metadata["diabetes_type"]
    == "T2D"
]["person_id"].tolist()

t2d_summary = (
    population_event_summary(
        events_df,
        "T2D"
    )
)

info(
    f"T2D participants: "
    f"{len(t2d_participants)}"
)

info(
    f"T2D participants with "
    f"≥1 event: "
    f"{t2d_summary['participants_with_events']}"
)

info(
    f"T2D events: "
    f"{t2d_summary['events']:,}"
)

if t2d_summary[
    "mean_duration_minutes"
] is not None:

    info(
        f"Mean event duration: "
        f"{t2d_summary['mean_duration_minutes']:.2f} min"
    )

    info(
        f"Median event duration: "
        f"{t2d_summary['median_duration_minutes']:.2f} min"
    )

    info(
        f"Longest event: "
        f"{t2d_summary['max_duration_minutes']:.2f} min"
    )

    info(
        f"Mean peak glucose: "
        f"{t2d_summary['mean_peak_glucose_mg_dl']:.2f} mg/dL"
    )

    info(
        f"Highest observed peak: "
        f"{t2d_summary['max_peak_glucose_mg_dl']:.2f} mg/dL"
    )


# ============================================================
# T2D participant breakdown
# ============================================================

print()
print("=" * 70)
print("T2D Participant Breakdown")
print("=" * 70)
print()

for person_id in sorted(
    t2d_participants
):

    person_events = t2d_events[
        t2d_events["person_id"]
        == person_id
    ]

    event_count = len(
        person_events
    )

    if event_count == 0:

        print(
            f"{person_id}: "
            f"0 events"
        )

        continue

    total_high_minutes = (
        person_events[
            "duration_minutes"
        ]
        .sum()
    )

    max_peak = (
        person_events[
            "peak_glucose_mg_dl"
        ]
        .max()
    )

    median_duration = (
        person_events[
            "duration_minutes"
        ]
        .median()
    )

    print(
        f"{person_id}: "
        f"{event_count} events | "
        f"total event duration="
        f"{total_high_minutes:.1f} min | "
        f"median duration="
        f"{median_duration:.1f} min | "
        f"peak="
        f"{max_peak:.1f} mg/dL"
    )


# ============================================================
# Event type breakdown
# ============================================================

print()
print("=" * 70)
print("Event Structure")
print("=" * 70)
print()

if not events_df.empty:

    event_type_counts = (
        events_df["event_type"]
        .value_counts()
    )

    for event_type, count in (
        event_type_counts.items()
    ):

        info(
            f"{event_type}: "
            f"{count:,}"
        )


# ============================================================
# Duration distribution
# ============================================================

if not t2d_events.empty:

    print()
    print("=" * 70)
    print("T2D Event Duration Distribution")
    print("=" * 70)
    print()

    duration_stats = (
        t2d_events[
            "duration_minutes"
        ]
        .describe()
    )

    print(
        duration_stats.to_string()
    )


# ============================================================
# Participants with no events
# ============================================================

participants_with_events = set(
    events_df[
        events_df["diabetes_type"]
        == "T2D"
    ]["person_id"]
)

t2d_without_events = sorted(
    set(t2d_participants)
    - participants_with_events
)

print()
print("=" * 70)
print("T2D Participants Without Events")
print("=" * 70)
print()

if t2d_without_events:

    for person_id in t2d_without_events:

        print(
            f"  {person_id}"
        )

else:

    print(
        "All T2D participants have "
        "at least one hyperglycemic event."
    )


# ============================================================
# Build JSON report
# ============================================================

json_report = {
    "dataset": "Hall_2018",

    "configuration": {
        "glucose_threshold_mg_dl":
            GLUCOSE_THRESHOLD,

        "recording_segment_gap_minutes":
            SEGMENT_GAP_MINUTES,

        "event_gap_minutes":
            EVENT_GAP_MINUTES,
    },

    "participants": {
        "metadata_count":
            int(len(metadata)),

        "glucose_file_count":
            int(len(glucose_files)),

        "analysis_errors":
            int(
                participants_df[
                    participants_df["status"]
                    == "ERROR"
                ]["person_id"]
                .nunique()
            ),
    },

    "event_counts": {
        "all":
            int(len(events_df)),

        "T2D":
            int(len(t2d_events)),

        "Prediabetes":
            int(
                len(
                    events_df[
                        events_df["diabetes_type"]
                        == "Prediabetes"
                    ]
                )
            ),

        "No diabetes":
            int(
                len(
                    events_df[
                        events_df["diabetes_type"]
                        == "No diabetes"
                    ]
                )
            ),
    },

    "T2D_summary":
        t2d_summary,

    "T2D_participants_without_events":
        t2d_without_events,

    "event_type_counts":
        (
            events_df["event_type"]
            .value_counts()
            .to_dict()
            if not events_df.empty
            else {}
        ),

    "notes": [
        "An event consists of consecutive observations above the configured glucose threshold within the configured event-gap limit.",
        "Recording gaps larger than the configured segment threshold are treated as separate recording segments.",
        "Non-numeric glucose values are not converted into numeric values.",
        "Single-observation events are retained rather than silently discarded.",
        "Raw glucose files are never modified.",
        "This analysis identifies event structure; it does not train or evaluate a prediction model.",
    ],
}


# ============================================================
# Save JSON
# ============================================================

try:

    with open(
        EVENT_JSON,
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
        f"JSON report written to: {EVENT_JSON}"
    )

except Exception as exc:

    error(
        f"Could not write JSON report: {exc}"
    )


# ============================================================
# Final integrity report
# ============================================================

analysis_errors = participants_df[
    participants_df["status"]
    == "ERROR"
]

print()
print("=" * 70)

if analysis_errors.empty:

    print(
        "OVERALL STATUS: ANALYSIS COMPLETE"
    )

else:

    print(
        "OVERALL STATUS: COMPLETE WITH ERRORS"
    )

print("=" * 70)

print()

info(
    f"Participants attempted: "
    f"{len(glucose_files)}"
)

info(
    f"Participants with errors: "
    f"{analysis_errors['person_id'].nunique()}"
)

info(
    f"Total events detected: "
    f"{len(events_df):,}"
)

print()

print(
    "Next step: use the event distribution to determine "
    "whether the prediction target represents independent "
    "hyperglycemic excursions or clusters of correlated "
    "windows."
)

print()
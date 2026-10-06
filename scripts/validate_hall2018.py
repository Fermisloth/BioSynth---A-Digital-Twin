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

GLUCOSE_DIR = DATASET_DIR / "Hall_2018-extracted-glucose-files"
METADATA_FILE = DATASET_DIR / "Hall_2018-metadata.csv"

REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)

REPORT_JSON = REPORT_DIR / "hall2018_validation_report.json"
REPORT_CSV = REPORT_DIR / "hall2018_person_summary.csv"

# Exploratory threshold.
# We are NOT deleting observations.
# We only use this threshold to identify recording segments.
SEGMENT_GAP_MINUTES = 15

# Eventual prediction horizon.
PREDICTION_HORIZON_HOURS = 2

# Working target threshold.
GLUCOSE_THRESHOLD = 180


# ============================================================
# Utility functions
# ============================================================

def status(message):
    print(f"[INFO] {message}")


def success(message):
    print(f"[ OK ] {message}")


def warning(message):
    print(f"[WARN] {message}")


def failure(message):
    print(f"[FAIL] {message}")


def check(condition, success_message, failure_message):
    if condition:
        success(success_message)
        return True

    failure(failure_message)
    return False


# ============================================================
# 1. Check dataset structure
# ============================================================

print()
print("=" * 70)
print("Hall 2018 Dataset Validation")
print("=" * 70)
print()

status(f"Project root: {PROJECT_ROOT}")
status(f"Dataset directory: {DATASET_DIR}")
print()

structure_ok = True

structure_ok &= check(
    DATASET_DIR.exists(),
    "Dataset directory exists.",
    f"Dataset directory not found: {DATASET_DIR}"
)

structure_ok &= check(
    METADATA_FILE.exists(),
    "Metadata file exists.",
    f"Metadata file not found: {METADATA_FILE}"
)

structure_ok &= check(
    GLUCOSE_DIR.exists(),
    "Glucose directory exists.",
    f"Glucose directory not found: {GLUCOSE_DIR}"
)

if not structure_ok:
    failure("Dataset structure is incomplete. Stopping.")
    sys.exit(1)


# ============================================================
# 2. Load metadata
# ============================================================

status("Loading metadata...")

try:
    metadata = pd.read_csv(METADATA_FILE)
except Exception as exc:
    failure(f"Could not read metadata file: {exc}")
    sys.exit(1)


required_metadata_columns = {
    "person_id",
    "diabetes_type",
    "age",
    "gender",
    "race_ethnicity",
    "hba1c_%",
    "CGM_type",
    "glucose_level_record_count",
    "average_glucose_level_mg_dl",
    "count_days_with_CGM_data",
}

missing_metadata_columns = (
    required_metadata_columns - set(metadata.columns)
)

metadata_schema_ok = check(
    not missing_metadata_columns,
    "Metadata schema is correct.",
    f"Missing metadata columns: {sorted(missing_metadata_columns)}"
)

status(f"Metadata rows: {len(metadata)}")
status(f"Metadata columns: {len(metadata.columns)}")

duplicate_person_ids = metadata["person_id"].duplicated().sum()

metadata_unique_ok = check(
    duplicate_person_ids == 0,
    "No duplicate person_id values.",
    f"Found {duplicate_person_ids} duplicate person_id values."
)


# ============================================================
# 3. Count diabetes groups
# ============================================================

print()
status("Diabetes population:")

diabetes_counts = (
    metadata["diabetes_type"]
    .value_counts(dropna=False)
)

for diabetes_type, count in diabetes_counts.items():
    print(f"       {diabetes_type}: {count}")


# ============================================================
# 4. Match metadata to glucose files
# ============================================================

print()
status("Checking metadata ↔ glucose-file linkage...")

glucose_files = sorted(GLUCOSE_DIR.glob("*.csv"))

file_person_ids = {
    file.stem
    for file in glucose_files
}

metadata_person_ids = set(
    metadata["person_id"].astype(str)
)

missing_glucose_files = sorted(
    metadata_person_ids - file_person_ids
)

unexpected_glucose_files = sorted(
    file_person_ids - metadata_person_ids
)

linkage_ok = (
    not missing_glucose_files
    and not unexpected_glucose_files
)

check(
    linkage_ok,
    f"All {len(metadata_person_ids)} metadata people have matching glucose files.",
    "Metadata ↔ glucose-file mismatch detected."
)

if missing_glucose_files:
    warning(
        f"People missing glucose files: {missing_glucose_files}"
    )

if unexpected_glucose_files:
    warning(
        f"Glucose files without metadata: {unexpected_glucose_files}"
    )


# ============================================================
# 5. Inspect every glucose file
# ============================================================

print()
print("=" * 70)
print("Scanning glucose time-series files")
print("=" * 70)
print()

person_results = []

total_rows = 0
total_missing_glucose = 0
total_non_numeric_glucose = 0
total_invalid_timestamps = 0
total_duplicate_timestamps = 0
total_threshold_crossings = 0

files_failed_to_read = []
files_with_schema_errors = []
files_with_warnings = []


for index, file_path in enumerate(glucose_files, start=1):

    person_id = file_path.stem

    print(
        f"[{index:02d}/{len(glucose_files):02d}] "
        f"Checking {person_id}...",
        end=" "
    )

    # --------------------------------------------------------
    # Default result.
    #
    # Even if something fails, we create a result row.
    # This prevents the final report from silently losing people.
    # --------------------------------------------------------

    result = {
        "person_id": person_id,
        "file_status": "ERROR",
        "rows": 0,
        "invalid_timestamps": 0,
        "missing_glucose": 0,
        "non_numeric_glucose": 0,
        "duplicate_timestamps": 0,
        "segment_count": 0,
        "segments_long_enough_for_2h": 0,
        "min_glucose": None,
        "max_glucose": None,
        "mean_glucose": None,
        "values_above_180": 0,
        "median_interval_minutes": None,
        "max_interval_minutes": None,
        "non_numeric_values": [],
        "error_message": None,
    }

    try:

        # ----------------------------------------------------
        # Read CSV
        # ----------------------------------------------------

        try:
            df = pd.read_csv(file_path)
        except Exception as exc:
            result["error_message"] = f"Could not read CSV: {exc}"
            files_failed_to_read.append(person_id)

            person_results.append(result)

            print("FAILED")
            warning(f"{person_id}: {result['error_message']}")
            continue

        result["rows"] = len(df)

        # ----------------------------------------------------
        # Schema
        # ----------------------------------------------------

        required_columns = {
            "timestamp",
            "glucose_value_mg_dl",
        }

        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            result["error_message"] = (
                f"Missing columns: {sorted(missing_columns)}"
            )

            files_with_schema_errors.append(person_id)

            person_results.append(result)

            print("FAILED")
            warning(
                f"{person_id}: {result['error_message']}"
            )
            continue

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

        result["invalid_timestamps"] = invalid_timestamps
        total_invalid_timestamps += invalid_timestamps

        # ----------------------------------------------------
        # Glucose values
        #
        # IMPORTANT:
        #
        # Do not assume this column is numeric.
        #
        # Values such as:
        #
        #     Low
        #
        # are valid raw categorical readings that need
        # to be reported, not converted into invented numbers.
        # ----------------------------------------------------

        raw_glucose = df["glucose_value_mg_dl"]

        missing_glucose = int(
            raw_glucose.isna().sum()
        )

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

        non_numeric_values = (
            raw_glucose[non_numeric_mask]
            .astype(str)
            .value_counts()
            .to_dict()
        )

        result["missing_glucose"] = missing_glucose
        result["non_numeric_glucose"] = non_numeric_count
        result["non_numeric_values"] = non_numeric_values

        total_missing_glucose += missing_glucose
        total_non_numeric_glucose += non_numeric_count

        # ----------------------------------------------------
        # Duplicate timestamps
        # ----------------------------------------------------

        duplicate_timestamps = int(
            df["timestamp"]
            .duplicated()
            .sum()
        )

        result["duplicate_timestamps"] = duplicate_timestamps
        total_duplicate_timestamps += duplicate_timestamps

        # ----------------------------------------------------
        # Sort by time
        # ----------------------------------------------------

        df = df.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        # ----------------------------------------------------
        # Time intervals
        # ----------------------------------------------------

        intervals = (
            df["timestamp"]
            .diff()
            .dt.total_seconds()
            / 60
        )

        valid_intervals = intervals.dropna()

        # ----------------------------------------------------
        # Recording segments
        # ----------------------------------------------------

        segment_breaks = (
            intervals > SEGMENT_GAP_MINUTES
        )

        segment_id = (
            segment_breaks
            .fillna(False)
            .cumsum()
        )

        df["segment_id"] = segment_id

        segment_count = int(
            df["segment_id"].nunique()
        )

        result["segment_count"] = segment_count

        # ----------------------------------------------------
        # Segment-level statistics
        # ----------------------------------------------------

        segment_durations = []

        for _, segment in df.groupby("segment_id"):

            valid_segment_timestamps = (
                segment["timestamp"]
                .dropna()
            )

            if len(valid_segment_timestamps) < 2:
                continue

            start = valid_segment_timestamps.min()
            end = valid_segment_timestamps.max()

            duration_hours = (
                end - start
            ).total_seconds() / 3600

            segment_durations.append(
                duration_hours
            )

        # ----------------------------------------------------
        # 2-hour eligibility
        # ----------------------------------------------------

        valid_2h_segments = sum(
            duration >= PREDICTION_HORIZON_HOURS
            for duration in segment_durations
        )

        result["segments_long_enough_for_2h"] = (
            valid_2h_segments
        )

        # ----------------------------------------------------
        # Numeric glucose statistics
        #
        # Only genuinely numeric glucose observations are
        # used here.
        # ----------------------------------------------------

        valid_numeric_glucose = (
            numeric_glucose.dropna()
        )

        if len(valid_numeric_glucose):

            result["min_glucose"] = float(
                valid_numeric_glucose.min()
            )

            result["max_glucose"] = float(
                valid_numeric_glucose.max()
            )

            result["mean_glucose"] = float(
                valid_numeric_glucose.mean()
            )

            threshold_count = int(
                (
                    valid_numeric_glucose
                    > GLUCOSE_THRESHOLD
                ).sum()
            )

        else:

            threshold_count = 0

        result["values_above_180"] = (
            threshold_count
        )

        total_threshold_crossings += (
            threshold_count
        )

        # ----------------------------------------------------
        # Interval statistics
        # ----------------------------------------------------

        if len(valid_intervals):

            result["median_interval_minutes"] = (
                float(valid_intervals.median())
            )

            result["max_interval_minutes"] = (
                float(valid_intervals.max())
            )

        # ----------------------------------------------------
        # Determine file status
        # ----------------------------------------------------

        problems = []

        if invalid_timestamps:
            problems.append(
                f"{invalid_timestamps} invalid timestamps"
            )

        if missing_glucose:
            problems.append(
                f"{missing_glucose} missing glucose values"
            )

        if non_numeric_count:
            problems.append(
                f"{non_numeric_count} non-numeric glucose values"
            )

        if duplicate_timestamps:
            problems.append(
                f"{duplicate_timestamps} duplicate timestamps"
            )

        if problems:

            result["file_status"] = "WARNING"
            files_with_warnings.append(person_id)

            print("WARN")

            for problem in problems:
                print(f"    [WARN] {problem}")

            if non_numeric_values:
                print(
                    f"    [INFO] Non-numeric values: "
                    f"{non_numeric_values}"
                )

        else:

            result["file_status"] = "PASS"
            print("OK")

        person_results.append(result)

        total_rows += len(df)

    except Exception as exc:

        # ----------------------------------------------------
        # Catch unexpected errors.
        #
        # The validator must continue with the next person.
        # ----------------------------------------------------

        result["file_status"] = "ERROR"
        result["error_message"] = str(exc)

        person_results.append(result)

        print("FAILED")
        warning(
            f"{person_id}: unexpected validation error: {exc}"
        )


# ============================================================
# 6. Build person-level report
# ============================================================

results_df = pd.DataFrame(person_results)

results_df = results_df.merge(
    metadata,
    on="person_id",
    how="left"
)

results_df.to_csv(
    REPORT_CSV,
    index=False
)


# ============================================================
# 7. Dataset-wide validation
# ============================================================

print()
print("=" * 70)
print("Dataset-wide summary")
print("=" * 70)
print()

successful_files = int(
    (results_df["file_status"] == "PASS").sum()
)

warning_files = int(
    (results_df["file_status"] == "WARNING").sum()
)

error_files = int(
    (results_df["file_status"] == "ERROR").sum()
)

status(f"People in metadata: {len(metadata)}")
status(f"Glucose files: {len(glucose_files)}")
status(f"Total glucose observations: {total_rows}")
status(f"People scanned: {len(results_df)}")
status(f"Successful files: {successful_files}")
status(f"Files with warnings: {warning_files}")
status(f"Files with errors: {error_files}")
status(f"Missing glucose values: {total_missing_glucose}")
status(f"Non-numeric glucose values: {total_non_numeric_glucose}")
status(f"Invalid timestamps: {total_invalid_timestamps}")
status(f"Duplicate timestamps: {total_duplicate_timestamps}")

status(
    f"Glucose observations above {GLUCOSE_THRESHOLD} mg/dL: "
    f"{total_threshold_crossings}"
)

if files_with_warnings:
    warning(
        f"Files requiring review: {files_with_warnings}"
    )

if files_failed_to_read:
    warning(
        f"Files that could not be read: {files_failed_to_read}"
    )

if files_with_schema_errors:
    warning(
        f"Files with schema errors: {files_with_schema_errors}"
    )

print()


# ============================================================
# 8. T2D-specific summary
# ============================================================

t2d = results_df[
    results_df["diabetes_type"] == "T2D"
]

print("=" * 70)
print("T2D population")
print("=" * 70)
print()

status(f"T2D participants: {len(t2d)}")

if len(t2d):

    status(
        f"T2D glucose observations: "
        f"{t2d['rows'].sum():,.0f}"
    )

    status(
        f"T2D observations above 180 mg/dL: "
        f"{t2d['values_above_180'].sum():,.0f}"
    )

    status(
        f"T2D participants with ≥1 value above 180: "
        f"{(t2d['values_above_180'] > 0).sum()}"
    )

    status(
        f"T2D participants requiring review: "
        f"{(
            t2d['file_status'].isin(
                ['WARNING', 'ERROR']
            )
        ).sum()}"
    )


# ============================================================
# 9. Final validation status
# ============================================================

# These describe structural integrity.
#
# Data-quality anomalies such as "Low" are reported as
# warnings rather than causing the validator to crash.

critical_conditions = {
    "metadata_schema": metadata_schema_ok,
    "metadata_unique_person_ids": metadata_unique_ok,
    "metadata_to_file_linkage": linkage_ok,
    "all_files_scanned": (
        len(results_df) == len(glucose_files)
    ),
    "no_file_read_errors": (
        error_files == 0
    ),
    "no_schema_errors": (
        len(files_with_schema_errors) == 0
    ),
}

overall_valid = all(
    critical_conditions.values()
)

print()
print("=" * 70)

if overall_valid and warning_files == 0:
    overall_status = "PASS"
    print("OVERALL STATUS: PASS")

elif overall_valid:
    overall_status = "PASS_WITH_WARNINGS"
    print("OVERALL STATUS: PASS WITH WARNINGS")

else:
    overall_status = "REVIEW_REQUIRED"
    print("OVERALL STATUS: REVIEW REQUIRED")

print("=" * 70)

for condition, passed in critical_conditions.items():
    state = "PASS" if passed else "REVIEW"
    print(f"{state:>6}  {condition}")

print()

if warning_files:
    print(
        f"NOTE: {warning_files} file(s) contain data-quality "
        f"warnings. These were not silently discarded."
    )

print()


# ============================================================
# 10. Save machine-readable report
# ============================================================

report = {
    "dataset": "Hall_2018",

    "overall_status": overall_status,

    "configuration": {
        "segment_gap_minutes": SEGMENT_GAP_MINUTES,
        "prediction_horizon_hours": PREDICTION_HORIZON_HOURS,
        "glucose_threshold_mg_dl": GLUCOSE_THRESHOLD,
    },

    "dataset_summary": {
        "metadata_people": len(metadata),
        "glucose_files": len(glucose_files),
        "people_scanned": len(results_df),
        "successful_files": successful_files,
        "files_with_warnings": warning_files,
        "files_with_errors": error_files,
        "total_glucose_observations": total_rows,
        "missing_glucose_values": total_missing_glucose,
        "non_numeric_glucose_values": total_non_numeric_glucose,
        "invalid_timestamps": total_invalid_timestamps,
        "duplicate_timestamps": total_duplicate_timestamps,
        "observations_above_180": total_threshold_crossings,
    },

    "diabetes_population": (
        diabetes_counts
        .to_dict()
    ),

    "critical_conditions": critical_conditions,

    "files_with_warnings": files_with_warnings,

    "files_with_errors": [
        person_id
        for person_id in results_df.loc[
            results_df["file_status"] == "ERROR",
            "person_id"
        ].tolist()
    ],
}

with open(
    REPORT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=2,
        default=str
    )


success(
    f"Detailed CSV report: {REPORT_CSV}"
)

success(
    f"Machine-readable report: {REPORT_JSON}"
)

print()
print("Validation complete.")
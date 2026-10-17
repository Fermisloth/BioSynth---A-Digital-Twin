from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CGM_DIR = PROJECT_ROOT / "data" / "02_synthetic" / "cgm"
PARTICIPANT_METADATA_PATH = (
    PROJECT_ROOT / "data" / "02_synthetic" / "metadata" / "synthetic_participants.csv"
)
REPORT_DIR = PROJECT_ROOT / "reports"

OUTPUT_DATASET = REPORT_DIR / "synthetic_prediction_dataset.csv"
OUTPUT_EVENTS = REPORT_DIR / "synthetic_detected_hyperglycemic_events.csv"
OUTPUT_SUMMARY = REPORT_DIR / "synthetic_prediction_dataset_summary.json"

LOOKBACK_MINUTES = 60
HORIZON_MINUTES = 120
HYPERGLYCEMIA_THRESHOLD = 180.0
MAX_SEGMENT_GAP_MINUTES = 15.0
EVENT_GAP_MINUTES = 15.0


def fail(message: str) -> None:
    print(f"\nERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_participants() -> pd.DataFrame:
    if not PARTICIPANT_METADATA_PATH.exists():
        fail(
            "Synthetic participant metadata not found. "
            "Run scripts/generate_synthetic_cgm.py first."
        )

    participants = pd.read_csv(PARTICIPANT_METADATA_PATH)
    required = {"person_id", "simulation_phenotype"}
    missing = required - set(participants.columns)
    if missing:
        fail(f"Synthetic participant metadata is missing columns: {sorted(missing)}")

    participants["person_id"] = participants["person_id"].astype(str)
    return participants


def load_cgm_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"timestamp", "glucose_value_mg_dl"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError(f"{path.name} contains invalid timestamps")

    df["glucose_numeric"] = pd.to_numeric(df["glucose_value_mg_dl"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["gap_minutes"] = df["timestamp"].diff().dt.total_seconds().div(60.0)
    df["segment_id"] = df["gap_minutes"].gt(MAX_SEGMENT_GAP_MINUTES).fillna(False).cumsum()
    return df


def detect_hyperglycemic_events(
    person_id: str,
    simulation_phenotype: str,
    cgm: pd.DataFrame,
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []

    for segment_id, segment in cgm.groupby("segment_id"):
        high = segment.loc[
            segment["glucose_numeric"] > HYPERGLYCEMIA_THRESHOLD
        ].copy()
        if high.empty:
            continue

        high["high_gap_minutes"] = high["timestamp"].diff().dt.total_seconds().div(60.0)
        high["new_event"] = high["high_gap_minutes"].gt(EVENT_GAP_MINUTES).fillna(True)
        high["event_number"] = high["new_event"].cumsum()

        for event_number, event_data in high.groupby("event_number"):
            event_data = event_data.sort_values("timestamp")
            peak_index = event_data["glucose_numeric"].idxmax()
            peak_row = event_data.loc[peak_index]
            start_time = event_data["timestamp"].min()
            end_time = event_data["timestamp"].max()

            events.append(
                {
                    "person_id": person_id,
                    "simulation_phenotype": simulation_phenotype,
                    "segment_id": int(segment_id),
                    "event_id": f"{person_id}-{int(segment_id)}-{int(event_number)}",
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration_minutes": float(
                        (end_time - start_time).total_seconds() / 60.0
                    ),
                    "peak_time": peak_row["timestamp"],
                    "peak_glucose_mg_dl": float(peak_row["glucose_numeric"]),
                    "observations_above_threshold": int(len(event_data)),
                    "event_type": (
                        "single_observation"
                        if len(event_data) == 1
                        else "multi_observation"
                    ),
                }
            )

    return events


def find_event_relationship(
    prediction_time: pd.Timestamp,
    person_events: pd.DataFrame,
) -> dict[str, object]:
    result = {
        "event_status": "no_event",
        "target_event_id": None,
        "time_to_start_time_minutes": None,
        "time_from_start_time_minutes": None,
    }

    if person_events.empty:
        return result

    underway = person_events.loc[
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

    future_events = person_events.loc[
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


def build_prediction_rows(
    person_id: str,
    simulation_phenotype: str,
    cgm: pd.DataFrame,
    events: pd.DataFrame,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    event_records = (
        events.sort_values("start_time")[
            ["event_id", "start_time", "end_time"]
        ].to_dict("records")
        if not events.empty
        else []
    )
    event_index = 0
    history_steps = int(LOOKBACK_MINUTES / 5)
    future_steps = int(HORIZON_MINUTES / 5)
    cgm = cgm.reset_index(drop=True)
    glucose = cgm["glucose_numeric"]
    future_max_values = (
        glucose.iloc[::-1]
        .shift(1)
        .rolling(window=future_steps, min_periods=future_steps)
        .max()
        .iloc[::-1]
    )
    timestamps = cgm["timestamp"].to_numpy()
    glucose_values = cgm["glucose_numeric"].to_numpy()
    segment_values = cgm["segment_id"].to_numpy()
    future_max_array = future_max_values.to_numpy()

    for row_index in range(history_steps, len(cgm) - future_steps):
        prediction_time = pd.Timestamp(timestamps[row_index])
        current_segment = int(segment_values[row_index])
        current_glucose = glucose_values[row_index]

        history_segments = segment_values[row_index - history_steps : row_index + 1]
        future_segments = segment_values[row_index + 1 : row_index + future_steps + 1]
        if (
            history_segments.min() != history_segments.max()
            or future_segments.min() != future_segments.max()
        ):
            continue

        future_max = future_max_array[row_index]
        if pd.isna(future_max):
            continue

        future_max = float(future_max)
        future_hyperglycemia = future_max > HYPERGLYCEMIA_THRESHOLD

        while (
            event_index < len(event_records)
            and prediction_time > event_records[event_index]["end_time"]
        ):
            event_index += 1

        relationship = {
            "event_status": "no_event",
            "target_event_id": None,
            "time_to_start_time_minutes": None,
            "time_from_start_time_minutes": None,
        }
        if event_index < len(event_records):
            event = event_records[event_index]
            if event["start_time"] <= prediction_time <= event["end_time"]:
                relationship["event_status"] = "event_underway"
                relationship["target_event_id"] = str(event["event_id"])
                relationship["time_from_start_time_minutes"] = (
                    prediction_time - event["start_time"]
                ).total_seconds() / 60.0
            elif event["start_time"] > prediction_time:
                relationship["event_status"] = "pre_event"
                relationship["target_event_id"] = str(event["event_id"])
                relationship["time_to_start_time_minutes"] = (
                    event["start_time"] - prediction_time
                ).total_seconds() / 60.0

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
            prediction_class = "future_positive_unmatched_event"
        else:
            prediction_class = "negative"

        rows.append(
            {
                "person_id": person_id,
                "simulation_phenotype": simulation_phenotype,
                "prediction_timestamp": prediction_time,
                "lookback_minutes": LOOKBACK_MINUTES,
                "horizon_minutes": HORIZON_MINUTES,
                "current_glucose_mg_dl": (
                    float(current_glucose) if pd.notna(current_glucose) else None
                ),
                "current_is_high": bool(
                    pd.notna(current_glucose)
                    and current_glucose > HYPERGLYCEMIA_THRESHOLD
                ),
                "future_max_glucose_mg_dl": future_max,
                "future_hyperglycemia": bool(future_hyperglycemia),
                "event_status": relationship["event_status"],
                "target_event_id": relationship["target_event_id"],
                "time_to_start_time_minutes": relationship["time_to_start_time_minutes"],
                "time_from_start_time_minutes": relationship[
                    "time_from_start_time_minutes"
                ],
                "pre_event_target": pre_event_target,
                "prediction_class": prediction_class,
                "history_observation_count": int(history_steps + 1),
                "future_observation_count": int(future_steps),
                "actual_history_minutes": float(LOOKBACK_MINUTES),
                "actual_future_minutes": float(HORIZON_MINUTES),
                "segment_id": int(current_segment),
            }
        )

    return rows


def build_summary(
    participants: pd.DataFrame,
    events: pd.DataFrame,
    dataset: pd.DataFrame,
    failed_files: list[dict[str, str]],
) -> dict[str, object]:
    return {
        "configuration": {
            "lookback_minutes": LOOKBACK_MINUTES,
            "prediction_horizon_minutes": HORIZON_MINUTES,
            "hyperglycemia_threshold_mg_dl": HYPERGLYCEMIA_THRESHOLD,
            "max_segment_gap_minutes": MAX_SEGMENT_GAP_MINUTES,
            "event_gap_minutes": EVENT_GAP_MINUTES,
        },
        "processing": {
            "metadata_participants": int(len(participants)),
            "cgm_files_found": int(len(list(CGM_DIR.glob("*.csv")))),
            "participants_in_dataset": int(dataset["person_id"].nunique()),
            "failed_files": failed_files,
        },
        "detected_events": {
            "total": int(len(events)),
            "by_phenotype": events["simulation_phenotype"].value_counts().to_dict(),
            "by_event_type": events["event_type"].value_counts().to_dict(),
        },
        "dataset": {
            "rows": int(len(dataset)),
            "participants": int(dataset["person_id"].nunique()),
        },
        "target_distribution": {
            "future_hyperglycemia": {
                "positive": int(dataset["future_hyperglycemia"].sum()),
                "negative": int((~dataset["future_hyperglycemia"]).sum()),
                "positive_rate": float(dataset["future_hyperglycemia"].mean()),
            },
            "pre_event_target": {
                "positive": int(dataset["pre_event_target"].sum()),
                "negative": int((~dataset["pre_event_target"]).sum()),
                "positive_rate": float(dataset["pre_event_target"].mean()),
            },
        },
        "prediction_class_counts": dataset["prediction_class"].value_counts().to_dict(),
        "event_relationship_counts": dataset["event_status"].value_counts().to_dict(),
        "rows_by_simulation_phenotype": dataset[
            "simulation_phenotype"
        ].value_counts().to_dict(),
        "outputs": {
            "prediction_dataset": str(OUTPUT_DATASET.relative_to(PROJECT_ROOT)),
            "detected_events": str(OUTPUT_EVENTS.relative_to(PROJECT_ROOT)),
            "summary": str(OUTPUT_SUMMARY.relative_to(PROJECT_ROOT)),
        },
        "methodology_notes": [
            "Synthetic simulator event metadata is not used as a model feature.",
            "Detected hyperglycemic events are derived from raw synthetic CGM using glucose >180 mg/dL.",
            "pre_event_target is positive only before a new detected event starts within the 120-minute horizon.",
            "Rows where an event is already underway are kept but are not positive early-warning targets.",
            "This remains synthetic POC data and is not clinically validated.",
        ],
    }


def main() -> int:
    if not CGM_DIR.exists():
        fail("Synthetic CGM directory not found. Run generate_synthetic_cgm.py first.")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    participants = load_participants()
    participant_lookup = participants.set_index("person_id")

    all_events: list[dict[str, object]] = []
    cgm_cache: dict[str, pd.DataFrame] = {}
    failed_files: list[dict[str, str]] = []

    for cgm_path in sorted(CGM_DIR.glob("*.csv")):
        person_id = cgm_path.stem
        if person_id not in participant_lookup.index:
            failed_files.append({"person_id": person_id, "error": "No metadata row"})
            continue

        phenotype = str(participant_lookup.loc[person_id, "simulation_phenotype"])
        try:
            cgm = load_cgm_file(cgm_path)
        except Exception as exc:
            failed_files.append({"person_id": person_id, "error": str(exc)})
            continue

        cgm_cache[person_id] = cgm
        all_events.extend(detect_hyperglycemic_events(person_id, phenotype, cgm))

    events = pd.DataFrame(all_events)
    if events.empty:
        fail("No synthetic hyperglycemic events were detected.")

    events["start_time"] = pd.to_datetime(events["start_time"], errors="coerce")
    events["end_time"] = pd.to_datetime(events["end_time"], errors="coerce")
    events.to_csv(OUTPUT_EVENTS, index=False)

    prediction_rows: list[dict[str, object]] = []
    for person_id, cgm in cgm_cache.items():
        phenotype = str(participant_lookup.loc[person_id, "simulation_phenotype"])
        person_events = events.loc[events["person_id"] == person_id].copy()
        prediction_rows.extend(
            build_prediction_rows(person_id, phenotype, cgm, person_events)
        )

    dataset = pd.DataFrame(prediction_rows)
    if dataset.empty:
        fail("No synthetic prediction windows were generated.")

    dataset = dataset.sort_values(["person_id", "prediction_timestamp"]).reset_index(
        drop=True
    )
    dataset.to_csv(OUTPUT_DATASET, index=False)

    summary = build_summary(participants, events, dataset, failed_files)
    with OUTPUT_SUMMARY.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)

    print()
    print("Synthetic prediction dataset complete")
    print("=" * 45)
    print(f"Participants represented: {dataset['person_id'].nunique()}")
    print(f"Prediction rows: {len(dataset):,}")
    print(f"Detected hyperglycemic events: {len(events):,}")
    print(
        "Pre-event positives: "
        f"{int(dataset['pre_event_target'].sum()):,} "
        f"({dataset['pre_event_target'].mean():.2%})"
    )
    print(f"Prediction classes: {dataset['prediction_class'].value_counts().to_dict()}")
    print(f"Rows by phenotype: {dataset['simulation_phenotype'].value_counts().to_dict()}")
    print(f"Output dataset: {OUTPUT_DATASET}")
    print(f"Output events: {OUTPUT_EVENTS}")
    print(f"Summary JSON: {OUTPUT_SUMMARY}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

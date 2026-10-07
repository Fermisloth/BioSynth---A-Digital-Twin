from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTION_PATH = PROJECT_ROOT / "reports" / "synthetic_prediction_dataset.csv"
RAW_CGM_DIR = PROJECT_ROOT / "data" / "02_synthetic" / "cgm"
PARTICIPANT_METADATA_PATH = (
    PROJECT_ROOT / "data" / "02_synthetic" / "metadata" / "synthetic_participants.csv"
)
OUTPUT_PATH = PROJECT_ROOT / "reports" / "synthetic_cgm_trajectory_features.csv"
SUMMARY_PATH = PROJECT_ROOT / "reports" / "synthetic_cgm_trajectory_features_summary.json"

LOOKBACK_MINUTES = 60
SAMPLING_INTERVAL_MINUTES = 5
LOOKBACK_STEPS = LOOKBACK_MINUTES // SAMPLING_INTERVAL_MINUTES
HIGH_THRESHOLD = 180.0

TARGET_COLUMN = "pre_event_target"

FEATURE_COLUMNS = [
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

FUTURE_EVENT_BLOCKLIST = {
    "future_max_glucose_mg_dl",
    "future_hyperglycemia",
    "future_observation_count",
    "event_status",
    "target_event_id",
    "time_to_start_time_minutes",
    "time_from_start_time_minutes",
    "prediction_class",
    "event_start",
    "event_end",
    "event_id",
}

SIMULATOR_PARAMETER_BLOCKLIST = {
    "baseline_glucose_mg_dl",
    "circadian_amplitude",
    "noise_scale",
    "meal_response_scale",
    "recovery_parameter",
    "random_seed",
}

SYNTHETIC_EVENT_GROUND_TRUTH_BLOCKLIST = {
    "event_timestamp",
    "event_type",
    "event_magnitude",
    "event_duration_parameter",
    "meal_number",
}


def fail(message: str) -> None:
    print(f"\nERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_prediction_dataset() -> pd.DataFrame:
    if not PREDICTION_PATH.exists():
        fail(f"Synthetic prediction dataset not found: {PREDICTION_PATH}")

    prediction = pd.read_csv(PREDICTION_PATH)
    required = {"person_id", "prediction_timestamp", TARGET_COLUMN}
    missing = required - set(prediction.columns)
    if missing:
        fail(f"Synthetic prediction dataset is missing columns: {sorted(missing)}")

    prediction["person_id"] = prediction["person_id"].astype(str)
    prediction["prediction_timestamp"] = pd.to_datetime(
        prediction["prediction_timestamp"], errors="coerce"
    )
    if prediction["prediction_timestamp"].isna().any():
        fail("Synthetic prediction dataset contains invalid prediction timestamps.")

    prediction[TARGET_COLUMN] = prediction[TARGET_COLUMN].astype(bool)

    if "simulation_phenotype" not in prediction.columns:
        if not PARTICIPANT_METADATA_PATH.exists():
            fail("simulation_phenotype missing and participant metadata was not found.")
        participants = pd.read_csv(PARTICIPANT_METADATA_PATH)
        prediction = prediction.merge(
            participants[["person_id", "simulation_phenotype"]],
            on="person_id",
            how="left",
        )

    if prediction["simulation_phenotype"].isna().any():
        fail("Some prediction rows do not have simulation_phenotype context.")

    return prediction.sort_values(["person_id", "prediction_timestamp"]).reset_index(drop=True)


def load_raw_cgm(person_id: str) -> pd.DataFrame:
    path = RAW_CGM_DIR / f"{person_id}.csv"
    if not path.exists():
        fail(f"Raw synthetic CGM file not found for {person_id}: {path}")

    raw = pd.read_csv(path)
    required = {"timestamp", "glucose_value_mg_dl"}
    missing = required - set(raw.columns)
    if missing:
        fail(f"{path.name} is missing columns: {sorted(missing)}")

    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
    if raw["timestamp"].isna().any():
        fail(f"{path.name} contains invalid timestamps.")

    raw["glucose_numeric"] = pd.to_numeric(raw["glucose_value_mg_dl"], errors="coerce")
    raw = raw.sort_values("timestamp").reset_index(drop=True)
    if raw["timestamp"].duplicated().any():
        fail(f"{path.name} contains duplicate timestamps.")

    return raw


def slope_per_hour(history: pd.DataFrame) -> float:
    numeric = history.loc[history["glucose_numeric"].notna()]
    if len(numeric) < 2:
        return np.nan

    elapsed_hours = (
        numeric["timestamp"] - numeric["timestamp"].iloc[0]
    ).dt.total_seconds().to_numpy() / 3600.0
    if elapsed_hours.max() <= 0:
        return np.nan

    return float(np.polyfit(elapsed_hours, numeric["glucose_numeric"].to_numpy(), 1)[0])


def latest_change(history: pd.DataFrame, prediction_time: pd.Timestamp, minutes: int) -> float:
    target_time = prediction_time - pd.Timedelta(minutes=minutes)
    eligible = history.loc[
        (history["timestamp"] <= target_time) & history["glucose_numeric"].notna()
    ]
    current = history.loc[
        (history["timestamp"] <= prediction_time) & history["glucose_numeric"].notna()
    ]
    if eligible.empty or current.empty:
        return np.nan
    return float(current.iloc[-1]["glucose_numeric"] - eligible.iloc[-1]["glucose_numeric"])


def calculate_features(history: pd.DataFrame, prediction_time: pd.Timestamp) -> dict[str, float | int]:
    numeric = history.loc[history["glucose_numeric"].notna()].copy()
    if numeric.empty:
        return {column: np.nan for column in FEATURE_COLUMNS}

    values = numeric["glucose_numeric"]
    current_glucose = float(values.iloc[-1])
    history_min = float(values.min())
    history_max = float(values.max())
    high_mask = values > HIGH_THRESHOLD

    intervals = numeric["timestamp"].diff().dt.total_seconds().div(60.0)
    high_minutes = float(intervals.loc[(intervals > 0) & high_mask].sum())
    actual_history_minutes = float(
        (numeric["timestamp"].iloc[-1] - numeric["timestamp"].iloc[0]).total_seconds()
        / 60.0
    )
    history_mean = float(values.mean())
    history_std = float(values.std(ddof=1)) if len(values) >= 2 else np.nan

    return {
        "current_glucose_mg_dl": current_glucose,
        "current_is_high": int(current_glucose > HIGH_THRESHOLD),
        "history_mean_glucose": history_mean,
        "history_median_glucose": float(values.median()),
        "history_min_glucose": history_min,
        "history_max_glucose": history_max,
        "history_range_glucose": history_max - history_min,
        "glucose_change_15m": latest_change(numeric, prediction_time, 15),
        "glucose_change_30m": latest_change(numeric, prediction_time, 30),
        "glucose_change_60m": latest_change(numeric, prediction_time, 60),
        "glucose_slope_per_hour": slope_per_hour(numeric),
        "history_std_glucose": history_std,
        "history_cv_glucose": (
            float(history_std / history_mean)
            if pd.notna(history_std) and history_mean != 0
            else np.nan
        ),
        "history_high_count": int(high_mask.sum()),
        "history_high_fraction": float(high_mask.mean()),
        "history_minutes_above_180": high_minutes,
        "trajectory_numeric_count": int(len(numeric)),
        "trajectory_actual_history_minutes": actual_history_minutes,
    }


def rolling_slope(values: np.ndarray) -> float:
    if np.isnan(values).sum() or len(values) < 2:
        return np.nan
    elapsed_hours = np.arange(len(values), dtype=float) * SAMPLING_INTERVAL_MINUTES / 60.0
    return float(np.polyfit(elapsed_hours, values, 1)[0])


def calculate_rolling_features(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.sort_values("timestamp").reset_index(drop=True).copy()
    values = raw["glucose_numeric"]
    rolling = values.rolling(window=LOOKBACK_STEPS + 1, min_periods=1)

    output = pd.DataFrame(
        {
            "timestamp": raw["timestamp"],
            "current_glucose_mg_dl": values,
            "current_is_high": (values > HIGH_THRESHOLD).astype(int),
            "history_mean_glucose": rolling.mean(),
            "history_median_glucose": rolling.median(),
            "history_min_glucose": rolling.min(),
            "history_max_glucose": rolling.max(),
            "history_std_glucose": rolling.std(ddof=1),
            "trajectory_numeric_count": rolling.count().astype(int),
        }
    )
    output["history_range_glucose"] = (
        output["history_max_glucose"] - output["history_min_glucose"]
    )
    output["glucose_change_15m"] = values - values.shift(3)
    output["glucose_change_30m"] = values - values.shift(6)
    output["glucose_change_60m"] = values - values.shift(12)
    output["glucose_slope_per_hour"] = values.rolling(
        window=LOOKBACK_STEPS + 1,
        min_periods=2,
    ).apply(rolling_slope, raw=True)
    output["history_cv_glucose"] = (
        output["history_std_glucose"] / output["history_mean_glucose"]
    )

    high_numeric = (values > HIGH_THRESHOLD).astype(int)
    output["history_high_count"] = high_numeric.rolling(
        window=LOOKBACK_STEPS + 1,
        min_periods=1,
    ).sum().astype(int)
    output["history_high_fraction"] = (
        output["history_high_count"] / output["trajectory_numeric_count"]
    )

    intervals = raw["timestamp"].diff().dt.total_seconds().div(60.0).fillna(0)
    high_minutes = intervals.where(values > HIGH_THRESHOLD, 0.0)
    output["history_minutes_above_180"] = high_minutes.rolling(
        window=LOOKBACK_STEPS + 1,
        min_periods=1,
    ).sum()

    observed_history_rows = np.minimum(np.arange(len(raw)) + 1, LOOKBACK_STEPS + 1)
    output["trajectory_actual_history_minutes"] = (
        observed_history_rows - 1
    ) * SAMPLING_INTERVAL_MINUTES

    return output[["timestamp", *FEATURE_COLUMNS]]


def build_features(prediction: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    feature_frames = []
    complete_history = 0
    short_history = 0
    missing_current_glucose = 0
    feature_failures = 0
    current_disagreements = 0

    for person_id, group in prediction.groupby("person_id", sort=False):
        raw = load_raw_cgm(person_id)
        raw_features = calculate_rolling_features(raw)
        merge_group = group.drop(
            columns=[column for column in FEATURE_COLUMNS if column in group.columns],
            errors="ignore",
        )
        group_features = merge_group.merge(
            raw_features,
            left_on="prediction_timestamp",
            right_on="timestamp",
            how="left",
            validate="many_to_one",
        )
        group_features = group_features[
            [
                "person_id",
                "prediction_timestamp",
                "simulation_phenotype",
                TARGET_COLUMN,
                *FEATURE_COLUMNS,
            ]
        ]

        complete_history += int(
            (group_features["trajectory_actual_history_minutes"] >= LOOKBACK_MINUTES).sum()
        )
        short_history += int(
            (group_features["trajectory_actual_history_minutes"] < LOOKBACK_MINUTES).sum()
        )
        missing_current_glucose += int(group_features["current_glucose_mg_dl"].isna().sum())
        feature_failures += int(group_features[FEATURE_COLUMNS].isna().all(axis=1).sum())

        expected_current = group["current_glucose_mg_dl"].reset_index(drop=True)
        observed_current = group_features["current_glucose_mg_dl"].reset_index(drop=True)
        current_disagreements += int(
            (~np.isclose(expected_current, observed_current, equal_nan=True)).sum()
        )

        feature_frames.append(group_features)

    completeness = {
        "rows_with_complete_60_min_history": complete_history,
        "rows_with_shorter_than_60_min_history": short_history,
        "rows_missing_current_glucose": missing_current_glucose,
        "rows_where_features_could_not_be_calculated": feature_failures,
        "rows_with_current_glucose_disagreement": current_disagreements,
    }
    return pd.concat(feature_frames, ignore_index=True), completeness


def build_summary(
    prediction: pd.DataFrame,
    features: pd.DataFrame,
    completeness: dict[str, int],
) -> dict[str, object]:
    forbidden_future = sorted(set(FEATURE_COLUMNS) & FUTURE_EVENT_BLOCKLIST)
    forbidden_simulator = sorted(set(FEATURE_COLUMNS) & SIMULATOR_PARAMETER_BLOCKLIST)
    forbidden_event_truth = sorted(set(FEATURE_COLUMNS) & SYNTHETIC_EVENT_GROUND_TRUTH_BLOCKLIST)

    alignment_ok = bool(
        len(prediction) == len(features)
        and prediction["person_id"].reset_index(drop=True).equals(features["person_id"])
        and prediction["prediction_timestamp"].reset_index(drop=True).equals(
            features["prediction_timestamp"]
        )
    )
    target_unchanged = bool(
        prediction[TARGET_COLUMN].reset_index(drop=True).equals(features[TARGET_COLUMN])
    )

    positive_rows = int(features[TARGET_COLUMN].sum())
    row_count = int(len(features))

    by_phenotype = {}
    for phenotype, group in features.groupby("simulation_phenotype"):
        by_phenotype[str(phenotype)] = {
            "rows": int(len(group)),
            "positive_rows": int(group[TARGET_COLUMN].sum()),
            "positive_rate": float(group[TARGET_COLUMN].mean()),
        }

    return {
        "source_files": {
            "prediction_dataset": str(PREDICTION_PATH.relative_to(PROJECT_ROOT)),
            "raw_cgm_directory": str(RAW_CGM_DIR.relative_to(PROJECT_ROOT)),
            "participant_metadata": str(PARTICIPANT_METADATA_PATH.relative_to(PROJECT_ROOT)),
        },
        "feature_dataset_path": str(OUTPUT_PATH.relative_to(PROJECT_ROOT)),
        "row_count": row_count,
        "participant_count": int(features["person_id"].nunique()),
        "feature_count": len(FEATURE_COLUMNS),
        "feature_names": FEATURE_COLUMNS,
        "target_distribution": {
            "target": TARGET_COLUMN,
            "positive_rows": positive_rows,
            "negative_rows": int(row_count - positive_rows),
            "positive_rate": float(features[TARGET_COLUMN].mean()),
        },
        "phenotype_distribution": by_phenotype,
        "trajectory_completeness": completeness,
        "leakage_checks": {
            "output_row_count_matches_prediction_dataset": row_count == len(prediction),
            "person_id_and_prediction_timestamp_aligned": alignment_ok,
            "pre_event_target_unchanged": target_unchanged,
            "future_event_derived_features": {
                "status": "PASS" if not forbidden_future else "FAIL",
                "columns": forbidden_future,
            },
            "simulator_parameters_excluded": {
                "status": "PASS" if not forbidden_simulator else "FAIL",
                "columns": forbidden_simulator,
            },
            "simulator_event_ground_truth_excluded": {
                "status": "PASS" if not forbidden_event_truth else "FAIL",
                "columns": forbidden_event_truth,
            },
            "current_glucose_agrees_with_raw_cgm": completeness[
                "rows_with_current_glucose_disagreement"
            ]
            == 0,
        },
        "methodology_notes": [
            "Only raw synthetic CGM observations at or before each prediction timestamp are used.",
            "pre_event_target is retained as the modeling label.",
            "simulation_phenotype is retained as context metadata, not as a predictive feature.",
            "Future/event-derived prediction columns are excluded from the feature list.",
            "Simulator generation parameters and synthetic event ground truth are excluded from predictive features.",
        ],
    }


def print_summary(summary: dict[str, object]) -> None:
    target = summary["target_distribution"]
    leakage = summary["leakage_checks"]
    completeness = summary["trajectory_completeness"]

    print()
    print("Synthetic CGM trajectory feature engineering complete")
    print("=====================================================")
    print(f"Rows: {summary['row_count']:,}")
    print(f"Participants: {summary['participant_count']:,}")
    print(f"Features: {summary['feature_count']}")
    print(f"Positive target: {target['positive_rows']:,}")
    print(f"Negative target: {target['negative_rows']:,}")
    print(f"Positive rate: {target['positive_rate']:.2%}")
    print()
    print("By phenotype:")
    for phenotype in ["No diabetes-like", "Prediabetes-like", "T2D-like"]:
        item = summary["phenotype_distribution"].get(phenotype, {})
        print(
            f"  {phenotype}: rows={item.get('rows', 0):,}, "
            f"positive={item.get('positive_rows', 0):,}, "
            f"positive_rate={item.get('positive_rate', 0):.2%}"
        )
    print()
    print("Trajectory completeness:")
    print(
        "  Complete 60-min history: "
        f"{completeness['rows_with_complete_60_min_history']:,}"
    )
    print(
        "  Short history: "
        f"{completeness['rows_with_shorter_than_60_min_history']:,}"
    )
    print(
        "  Missing current glucose: "
        f"{completeness['rows_missing_current_glucose']:,}"
    )
    print(
        "  Feature calculation failures: "
        f"{completeness['rows_where_features_could_not_be_calculated']:,}"
    )
    print()
    print("Leakage checks:")
    print(
        "  Future/event-derived features: "
        f"{leakage['future_event_derived_features']['status']}"
    )
    print(
        "  Simulator parameters excluded: "
        f"{leakage['simulator_parameters_excluded']['status']}"
    )
    print(
        "  Simulator event ground truth excluded: "
        f"{leakage['simulator_event_ground_truth_excluded']['status']}"
    )
    print()
    print("Output:")
    print(f"  {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")


def main() -> int:
    if not RAW_CGM_DIR.exists():
        fail(f"Raw synthetic CGM directory not found: {RAW_CGM_DIR}")

    prediction = load_prediction_dataset()
    features, completeness = build_features(prediction)

    if len(features) != len(prediction):
        fail("Feature row count does not match prediction dataset row count.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT_PATH, index=False)

    summary = build_summary(prediction, features, completeness)
    with SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

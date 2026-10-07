from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CGM_DIR = PROJECT_ROOT / "data" / "02_synthetic" / "cgm"
METADATA_DIR = PROJECT_ROOT / "data" / "02_synthetic" / "metadata"
PARTICIPANTS_PATH = METADATA_DIR / "synthetic_participants.csv"
SIMULATOR_EVENTS_PATH = METADATA_DIR / "synthetic_events.csv"
PREDICTION_SUMMARY_PATH = PROJECT_ROOT / "reports" / "synthetic_prediction_dataset_summary.json"

REPORT_DIR = PROJECT_ROOT / "reports"
PARTICIPANT_ANALYSIS_PATH = REPORT_DIR / "synthetic_cgm_analysis.csv"
SUMMARY_PATH = REPORT_DIR / "synthetic_cgm_analysis_summary.json"
TRAJECTORY_PLOT_PATH = REPORT_DIR / "synthetic_cgm_sample_trajectories.png"

HIGH_THRESHOLD = 180.0
EVENT_GAP_MINUTES = 15.0
PHENOTYPE_ORDER = ["No diabetes-like", "Prediabetes-like", "T2D-like"]


def fail(message: str) -> None:
    print(f"\nERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def pct(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator * 100) if denominator else 0.0


def numeric_summary(values: pd.Series) -> dict[str, float | int]:
    quantiles = values.quantile([0.01, 0.05, 0.25, 0.75, 0.95, 0.99])
    return {
        "observations": int(values.shape[0]),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(values.std(ddof=1)),
        "min": float(values.min()),
        "max": float(values.max()),
        "p01": float(quantiles.loc[0.01]),
        "p05": float(quantiles.loc[0.05]),
        "p25": float(quantiles.loc[0.25]),
        "p75": float(quantiles.loc[0.75]),
        "p95": float(quantiles.loc[0.95]),
        "p99": float(quantiles.loc[0.99]),
    }


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not CGM_DIR.exists():
        fail(f"Synthetic CGM directory not found: {CGM_DIR}")
    if not PARTICIPANTS_PATH.exists():
        fail(f"Synthetic participant metadata not found: {PARTICIPANTS_PATH}")
    if not SIMULATOR_EVENTS_PATH.exists():
        fail(f"Synthetic simulator event metadata not found: {SIMULATOR_EVENTS_PATH}")

    participants = pd.read_csv(PARTICIPANTS_PATH)
    required_participant_columns = {"person_id", "simulation_phenotype"}
    missing = required_participant_columns - set(participants.columns)
    if missing:
        fail(f"Participant metadata is missing columns: {sorted(missing)}")

    cgm_frames = []
    for path in sorted(CGM_DIR.glob("*.csv")):
        person_id = path.stem
        df = pd.read_csv(path)
        required_cgm_columns = {"timestamp", "glucose_value_mg_dl"}
        missing = required_cgm_columns - set(df.columns)
        if missing:
            fail(f"{path.name} is missing columns: {sorted(missing)}")
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        if df["timestamp"].isna().any():
            fail(f"{path.name} contains invalid timestamps")
        df["glucose_value_mg_dl"] = pd.to_numeric(
            df["glucose_value_mg_dl"], errors="coerce"
        )
        if df["glucose_value_mg_dl"].isna().any():
            fail(f"{path.name} contains nonnumeric synthetic glucose values")
        df["person_id"] = person_id
        cgm_frames.append(df)

    if not cgm_frames:
        fail(f"No synthetic CGM CSV files found in {CGM_DIR}")

    cgm = pd.concat(cgm_frames, ignore_index=True)
    participants["person_id"] = participants["person_id"].astype(str)
    cgm = cgm.merge(
        participants[["person_id", "simulation_phenotype"]],
        on="person_id",
        how="left",
    )
    if cgm["simulation_phenotype"].isna().any():
        fail("At least one synthetic CGM file has no participant metadata row")

    simulator_events = pd.read_csv(SIMULATOR_EVENTS_PATH)
    return participants, cgm, simulator_events


def analyze_boundaries(cgm: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    values = cgm["glucose_value_mg_dl"]
    total = len(cgm)
    checks = {
        "equal_40": values.eq(40),
        "equal_400": values.eq(400),
        "lte_54": values.le(54),
        "gte_250": values.ge(250),
        "gte_300": values.ge(300),
    }
    return {
        name: {"count": int(mask.sum()), "percent": pct(int(mask.sum()), total)}
        for name, mask in checks.items()
    }


def phenotype_distribution(cgm: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    result = {}
    for phenotype in PHENOTYPE_ORDER:
        group = cgm[cgm["simulation_phenotype"] == phenotype]
        values = group["glucose_value_mg_dl"]
        quantiles = values.quantile([0.05, 0.25, 0.75, 0.95])
        high_by_person = group.groupby("person_id")["glucose_value_mg_dl"].apply(
            lambda s: bool((s > HIGH_THRESHOLD).any())
        )
        result[phenotype] = {
            "participants": int(group["person_id"].nunique()),
            "observations": int(len(group)),
            "mean": float(values.mean()),
            "median": float(values.median()),
            "std": float(values.std(ddof=1)),
            "p05": float(quantiles.loc[0.05]),
            "p25": float(quantiles.loc[0.25]),
            "p75": float(quantiles.loc[0.75]),
            "p95": float(quantiles.loc[0.95]),
            "percent_gt_180": pct(int((values > HIGH_THRESHOLD).sum()), len(values)),
            "percent_gte_250": pct(int((values >= 250).sum()), len(values)),
            "participants_with_gt_180": int(high_by_person.sum()),
        }
    return result


def participant_analysis(cgm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (person_id, phenotype), group in cgm.groupby(
        ["person_id", "simulation_phenotype"], sort=True
    ):
        values = group["glucose_value_mg_dl"]
        rows.append(
            {
                "person_id": person_id,
                "simulation_phenotype": phenotype,
                "observations": int(len(group)),
                "mean_glucose": float(values.mean()),
                "median_glucose": float(values.median()),
                "std_glucose": float(values.std(ddof=1)),
                "min_glucose": float(values.min()),
                "max_glucose": float(values.max()),
                "percent_gt_180": pct(int((values > HIGH_THRESHOLD).sum()), len(values)),
                "percent_gte_250": pct(int((values >= 250).sum()), len(values)),
                "observations_gt_180": int((values > HIGH_THRESHOLD).sum()),
                "observations_gte_250": int((values >= 250).sum()),
            }
        )
    return pd.DataFrame(rows)


def summarize_participant_metrics(participants: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
    metrics = [
        "mean_glucose",
        "median_glucose",
        "std_glucose",
        "min_glucose",
        "max_glucose",
        "percent_gt_180",
        "percent_gte_250",
    ]
    result = {}
    for phenotype in PHENOTYPE_ORDER:
        group = participants[participants["simulation_phenotype"] == phenotype]
        result[phenotype] = {}
        for metric in metrics:
            series = group[metric]
            result[phenotype][metric] = {
                "min": float(series.min()),
                "median": float(series.median()),
                "mean": float(series.mean()),
                "max": float(series.max()),
            }
    return result


def smoothness_analysis(cgm: pd.DataFrame) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    ordered = cgm.sort_values(["person_id", "timestamp"]).copy()
    ordered["abs_5min_change"] = (
        ordered.groupby("person_id")["glucose_value_mg_dl"].diff().abs()
    )
    changes = ordered["abs_5min_change"].dropna()

    def summarize(changes_series: pd.Series) -> dict[str, float]:
        return {
            "median_abs_5min_change": float(changes_series.median()),
            "p95_abs_5min_change": float(changes_series.quantile(0.95)),
            "p99_abs_5min_change": float(changes_series.quantile(0.99)),
            "max_abs_5min_change": float(changes_series.max()),
            "percent_gt_20": pct(int((changes_series > 20).sum()), len(changes_series)),
            "percent_gt_30": pct(int((changes_series > 30).sum()), len(changes_series)),
            "percent_gt_50": pct(int((changes_series > 50).sum()), len(changes_series)),
        }

    by_phenotype = {}
    for phenotype in PHENOTYPE_ORDER:
        phenotype_changes = ordered.loc[
            ordered["simulation_phenotype"] == phenotype, "abs_5min_change"
        ].dropna()
        by_phenotype[phenotype] = summarize(phenotype_changes)

    return summarize(changes), by_phenotype


def hyperglycemia_distribution(participants: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    result = {}
    affected = participants[participants["observations_gt_180"] > 0]
    for phenotype in PHENOTYPE_ORDER:
        group = participants[participants["simulation_phenotype"] == phenotype]
        affected_group = affected[affected["simulation_phenotype"] == phenotype]
        total_high = int(group["observations_gt_180"].sum())
        result[phenotype] = {
            "total_gt_180_observations": total_high,
            "percent_gt_180": float(group["percent_gt_180"].mean()),
            "participants_with_gt_180": int(len(affected_group)),
            "median_gt_180_observations_per_affected_participant": (
                float(affected_group["observations_gt_180"].median())
                if len(affected_group)
                else 0.0
            ),
            "max_gt_180_observations_for_one_participant": int(
                group["observations_gt_180"].max()
            ),
        }
    return result


def detect_events(cgm: pd.DataFrame) -> pd.DataFrame:
    events = []
    for (person_id, phenotype), group in cgm.groupby(
        ["person_id", "simulation_phenotype"], sort=True
    ):
        group = group.sort_values("timestamp").reset_index(drop=True)
        high = group[group["glucose_value_mg_dl"] > HIGH_THRESHOLD].copy()
        if high.empty:
            continue
        high["gap_minutes"] = high["timestamp"].diff().dt.total_seconds().div(60.0)
        high["new_event"] = high["gap_minutes"].gt(EVENT_GAP_MINUTES).fillna(True)
        high["event_number"] = high["new_event"].cumsum()
        for event_number, event_data in high.groupby("event_number"):
            event_data = event_data.sort_values("timestamp")
            peak = event_data.loc[event_data["glucose_value_mg_dl"].idxmax()]
            start = event_data["timestamp"].min()
            end = event_data["timestamp"].max()
            events.append(
                {
                    "person_id": person_id,
                    "simulation_phenotype": phenotype,
                    "event_id": f"{person_id}-0-{int(event_number)}",
                    "start_time": start,
                    "end_time": end,
                    "duration_minutes": float((end - start).total_seconds() / 60.0),
                    "peak_time": peak["timestamp"],
                    "peak_glucose_mg_dl": float(peak["glucose_value_mg_dl"]),
                    "observations_above_threshold": int(len(event_data)),
                }
            )
    return pd.DataFrame(events)


def event_structure(events: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    result = {}
    for phenotype in PHENOTYPE_ORDER:
        group = events[events["simulation_phenotype"] == phenotype]
        counts = group.groupby("person_id").size()
        result[phenotype] = {
            "events": int(len(group)),
            "participants_with_events": int(group["person_id"].nunique()),
            "median_events_per_affected_participant": (
                float(counts.median()) if len(counts) else 0.0
            ),
            "median_event_duration": (
                float(group["duration_minutes"].median()) if len(group) else 0.0
            ),
            "mean_event_duration": (
                float(group["duration_minutes"].mean()) if len(group) else 0.0
            ),
            "maximum_event_duration": (
                float(group["duration_minutes"].max()) if len(group) else 0.0
            ),
            "median_event_peak_glucose": (
                float(group["peak_glucose_mg_dl"].median()) if len(group) else 0.0
            ),
            "maximum_event_peak_glucose": (
                float(group["peak_glucose_mg_dl"].max()) if len(group) else 0.0
            ),
        }
    return result


def no_diabetes_investigation(
    participants: pd.DataFrame,
    events: pd.DataFrame,
) -> dict[str, float | int | str | dict[str, int]]:
    no_diabetes = participants[
        participants["simulation_phenotype"] == "No diabetes-like"
    ].copy()
    affected = no_diabetes[no_diabetes["observations_gt_180"] > 0]
    affected_events = events[
        (events["simulation_phenotype"] == "No diabetes-like")
        & (events["person_id"].isin(affected["person_id"]))
    ]

    event_counts = affected_events.groupby("person_id").size().to_dict()
    if len(affected_events) <= 2:
        concentration = "concentrated in very few excursions"
    elif len(affected) <= 3:
        concentration = "concentrated in a small number of participants/excursions"
    else:
        concentration = "spread across multiple participants/excursions"

    return {
        "participants_with_gt_180": int(len(affected)),
        "total_gt_180_observations": int(affected["observations_gt_180"].sum()),
        "maximum_glucose_among_affected_participants": (
            float(affected["max_glucose"].max()) if len(affected) else 0.0
        ),
        "events_among_affected_participants": int(len(affected_events)),
        "event_counts_by_affected_participant": {
            str(k): int(v) for k, v in event_counts.items()
        },
        "concentration_assessment": concentration,
    }


def overlap_analysis(participants: pd.DataFrame) -> dict[str, object]:
    ranges = {}
    for phenotype in PHENOTYPE_ORDER:
        group = participants[participants["simulation_phenotype"] == phenotype]
        ranges[phenotype] = {
            "mean_glucose_range": [
                float(group["mean_glucose"].min()),
                float(group["mean_glucose"].max()),
            ],
            "max_glucose_range": [
                float(group["max_glucose"].min()),
                float(group["max_glucose"].max()),
            ],
        }

    mean_overlap_pairs = []
    max_overlap_pairs = []
    for left, right in zip(PHENOTYPE_ORDER, PHENOTYPE_ORDER[1:]):
        left_mean = ranges[left]["mean_glucose_range"]
        right_mean = ranges[right]["mean_glucose_range"]
        left_max = ranges[left]["max_glucose_range"]
        right_max = ranges[right]["max_glucose_range"]
        if max(left_mean[0], right_mean[0]) <= min(left_mean[1], right_mean[1]):
            mean_overlap_pairs.append(f"{left} / {right}")
        if max(left_max[0], right_max[0]) <= min(left_max[1], right_max[1]):
            max_overlap_pairs.append(f"{left} / {right}")

    if len(mean_overlap_pairs) == 2:
        conclusion = "substantial overlap"
    elif len(mean_overlap_pairs) == 1 and len(max_overlap_pairs) >= 1:
        conclusion = "moderate overlap"
    elif len(mean_overlap_pairs) == 1:
        conclusion = "little overlap"
    else:
        conclusion = "effectively separated"

    explanation = (
        f"Participant-level mean glucose overlaps for {mean_overlap_pairs or 'no adjacent phenotype pairs'}; "
        f"participant-level maximum glucose overlaps for {max_overlap_pairs or 'no adjacent phenotype pairs'}."
    )

    return {
        "ranges": ranges,
        "mean_overlap_pairs": mean_overlap_pairs,
        "max_overlap_pairs": max_overlap_pairs,
        "qualitative_conclusion": conclusion,
        "explanation": explanation,
    }


def create_sample_trajectory_plot(cgm: pd.DataFrame, participants: pd.DataFrame) -> None:
    selected_ids = []
    for phenotype in PHENOTYPE_ORDER:
        group = participants[participants["simulation_phenotype"] == phenotype].copy()
        median_mean = group["mean_glucose"].median()
        group["distance_to_phenotype_median_mean"] = (
            group["mean_glucose"] - median_mean
        ).abs()
        selected_ids.append(
            group.sort_values("distance_to_phenotype_median_mean").iloc[0]["person_id"]
        )

    fig, ax = plt.subplots(figsize=(12, 6))
    for person_id in selected_ids:
        person = cgm[cgm["person_id"] == person_id].sort_values("timestamp").head(288)
        phenotype = person["simulation_phenotype"].iloc[0]
        hours = (
            person["timestamp"] - person["timestamp"].iloc[0]
        ).dt.total_seconds() / 3600.0
        ax.plot(hours, person["glucose_value_mg_dl"], linewidth=1.8, label=f"{person_id} ({phenotype})")

    ax.axhline(HIGH_THRESHOLD, color="black", linestyle="--", linewidth=1, label="180 mg/dL")
    ax.set_title("Representative 24-hour Synthetic CGM Trajectories")
    ax.set_xlabel("Hours from start")
    ax.set_ylabel("Glucose (mg/dL)")
    ax.set_xlim(0, 24)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(TRAJECTORY_PLOT_PATH, dpi=160)
    plt.close(fig)


def recommendation(summary: dict[str, object]) -> tuple[str, list[str]]:
    boundary = summary["boundary_behavior"]
    smoothness = summary["temporal_smoothness"]["overall"]
    overlap = summary["phenotype_overlap"]["qualitative_conclusion"]
    reasons = []

    if boundary["equal_40"]["percent"] > 1.0:
        reasons.append("more than 1% of readings are clipped at 40 mg/dL")
    if boundary["equal_400"]["percent"] > 1.0:
        reasons.append("more than 1% of readings are clipped at 400 mg/dL")
    if smoothness["percent_gt_50"] > 1.0:
        reasons.append("more than 1% of 5-minute changes exceed 50 mg/dL")
    if overlap == "effectively separated":
        reasons.append("phenotype groups are effectively separated by participant-level mean glucose")

    if reasons:
        return "SYNTHETIC DATA STATUS: ADJUST GENERATOR", reasons
    return "SYNTHETIC DATA STATUS: ACCEPT FOR POC", [
        "boundary clipping is limited, smoothness is acceptable, and phenotype groups are not fully separated"
    ]


def main() -> int:
    participants_meta, cgm, _ = load_inputs()
    participant_stats = participant_analysis(cgm)
    participant_stats.to_csv(PARTICIPANT_ANALYSIS_PATH, index=False)

    events = detect_events(cgm)
    calculated_event_total = int(len(events))
    previous_event_total = None
    event_total_agreement = None
    if PREDICTION_SUMMARY_PATH.exists():
        with PREDICTION_SUMMARY_PATH.open("r", encoding="utf-8") as handle:
            previous_event_total = int(json.load(handle)["detected_events"]["total"])
        event_total_agreement = calculated_event_total == previous_event_total

    overall_smoothness, smoothness_by_phenotype = smoothness_analysis(cgm)
    summary = {
        "overall_distribution": {
            "participants": int(cgm["person_id"].nunique()),
            **numeric_summary(cgm["glucose_value_mg_dl"]),
        },
        "boundary_behavior": analyze_boundaries(cgm),
        "phenotype_distributions": phenotype_distribution(cgm),
        "participant_level_metric_summary_by_phenotype": summarize_participant_metrics(
            participant_stats
        ),
        "temporal_smoothness": {
            "overall": overall_smoothness,
            "by_phenotype": smoothness_by_phenotype,
        },
        "hyperglycemia_distribution": hyperglycemia_distribution(participant_stats),
        "no_diabetes_like_hyperglycemia_investigation": no_diabetes_investigation(
            participant_stats, events
        ),
        "hyperglycemic_event_structure": event_structure(events),
        "event_total_comparison": {
            "calculated_event_total": calculated_event_total,
            "previous_prediction_stage_event_total": previous_event_total,
            "agrees_with_prediction_stage": event_total_agreement,
            "explanation": (
                "Totals agree because both analyses detect events directly from synthetic raw CGM using glucose >180 mg/dL and a >15-minute gap to start a new event."
                if event_total_agreement
                else "Totals differ; compare event threshold, event-gap, and input-file rules."
            ),
        },
        "phenotype_overlap": overlap_analysis(participant_stats),
        "outputs": {
            "participant_csv": str(PARTICIPANT_ANALYSIS_PATH.relative_to(PROJECT_ROOT)),
            "summary_json": str(SUMMARY_PATH.relative_to(PROJECT_ROOT)),
            "sample_trajectory_png": str(TRAJECTORY_PLOT_PATH.relative_to(PROJECT_ROOT)),
        },
        "clinical_validation_note": "Synthetic phenotypes are simulation groups, not clinical diagnoses.",
    }
    status, reasons = recommendation(summary)
    summary["recommendation"] = {
        "status": status,
        "reasons": reasons,
    }

    create_sample_trajectory_plot(cgm, participant_stats)
    with SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    overall = summary["overall_distribution"]
    boundary = summary["boundary_behavior"]
    smooth = summary["temporal_smoothness"]["overall"]
    no_diabetes = summary["no_diabetes_like_hyperglycemia_investigation"]

    print()
    print("Synthetic CGM sanity analysis complete")
    print("=" * 45)
    print(
        "Overall glucose: "
        f"mean={overall['mean']:.2f}, median={overall['median']:.2f}, "
        f"std={overall['std']:.2f}, p05={overall['p05']:.1f}, "
        f"p95={overall['p95']:.1f}, min={overall['min']:.1f}, max={overall['max']:.1f}"
    )
    print(
        "Clipping: "
        f"equal 40={boundary['equal_40']['count']} ({boundary['equal_40']['percent']:.3f}%), "
        f"equal 400={boundary['equal_400']['count']} ({boundary['equal_400']['percent']:.3f}%)"
    )
    print("Phenotype glucose and >180 rates:")
    for phenotype in PHENOTYPE_ORDER:
        phenotype_summary = summary["phenotype_distributions"][phenotype]
        print(
            f"  {phenotype}: mean={phenotype_summary['mean']:.2f}, "
            f"median={phenotype_summary['median']:.2f}, "
            f">180={phenotype_summary['percent_gt_180']:.2f}%"
        )
    print(
        "Temporal smoothness: "
        f"median abs 5-min change={smooth['median_abs_5min_change']:.2f}, "
        f"p95={smooth['p95_abs_5min_change']:.2f}, "
        f"p99={smooth['p99_abs_5min_change']:.2f}, "
        f"max={smooth['max_abs_5min_change']:.2f}, "
        f">20={smooth['percent_gt_20']:.2f}%, "
        f">30={smooth['percent_gt_30']:.2f}%, "
        f">50={smooth['percent_gt_50']:.2f}%"
    )
    print("Hyperglycemic event counts by phenotype:")
    for phenotype in PHENOTYPE_ORDER:
        event_summary = summary["hyperglycemic_event_structure"][phenotype]
        print(f"  {phenotype}: {event_summary['events']} events")
    print(
        "No diabetes-like >180 investigation: "
        f"{no_diabetes['total_gt_180_observations']} readings across "
        f"{no_diabetes['participants_with_gt_180']} participants and "
        f"{no_diabetes['events_among_affected_participants']} events; "
        f"max glucose={no_diabetes['maximum_glucose_among_affected_participants']:.1f}; "
        f"{no_diabetes['concentration_assessment']}"
    )
    print(
        "Phenotype overlap: "
        f"{summary['phenotype_overlap']['qualitative_conclusion']} - "
        f"{summary['phenotype_overlap']['explanation']}"
    )
    print(
        "Event total comparison: "
        f"calculated={calculated_event_total}, previous={previous_event_total}, "
        f"agreement={event_total_agreement}"
    )
    print(f"Outputs: {PARTICIPANT_ANALYSIS_PATH}, {SUMMARY_PATH}, {TRAJECTORY_PLOT_PATH}")
    if status.endswith("ADJUST GENERATOR"):
        print("Adjustment reasons: " + "; ".join(reasons))
    print(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

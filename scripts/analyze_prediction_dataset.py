"""
Analyze the Hall 2018 supervised prediction dataset.

Purpose
-------
Audit the effective structure of the prediction target before feature
engineering or model training.

The analysis focuses on:

1. Overall target balance.
2. Target balance by diabetes type.
3. T2D participant-level distribution.
4. Hyperglycemic-event-level dependence.
5. Prediction rows per event.
6. Lead time before event onset.
7. Whether a small number of participants/events dominate positives.

This script does NOT modify the supervised dataset.
It only reads the generated CSV and writes analysis reports.

Input
-----
reports/hall2018_prediction_dataset.csv

Outputs
-------
reports/
    hall2018_prediction_dataset_analysis.csv
    hall2018_prediction_dataset_analysis.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DATASET = (
    PROJECT_ROOT
    / "reports"
    / "hall2018_prediction_dataset.csv"
)

REPORT_DIR = PROJECT_ROOT / "reports"

OUTPUT_CSV = (
    REPORT_DIR
    / "hall2018_prediction_dataset_analysis.csv"
)

OUTPUT_JSON = (
    REPORT_DIR
    / "hall2018_prediction_dataset_analysis.json"
)

T2D_LABEL = "T2D"


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def fail(message: str) -> None:
    """Print a clean error and terminate."""
    print()
    print(f"ERROR: {message}")
    sys.exit(1)


def load_dataset() -> pd.DataFrame:
    """Load and validate the supervised prediction dataset."""

    if not INPUT_DATASET.exists():
        fail(
            "Prediction dataset was not found:\n"
            f"  {INPUT_DATASET}\n\n"
            "Run build_prediction_dataset.py first."
        )

    try:
        df = pd.read_csv(INPUT_DATASET)
    except Exception as exc:
        fail(f"Could not read prediction dataset: {exc}")

    required_columns = {
        "person_id",
        "diabetes_type",
        "prediction_timestamp",
        "future_hyperglycemia",
        "pre_event_target",
        "prediction_class",
        "event_status",
        "target_event_id",
        "time_to_start_time_minutes",
    }

    missing = required_columns - set(df.columns)

    if missing:
        fail(
            "Prediction dataset is missing required columns:\n  "
            + "\n  ".join(sorted(missing))
        )

    df["person_id"] = df["person_id"].astype(str)
    df["diabetes_type"] = df["diabetes_type"].astype(str)

    # Explicit Boolean conversion.
    #
    # CSV files can represent booleans as True/False strings.
    for column in [
        "future_hyperglycemia",
        "pre_event_target",
    ]:
        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(
                {
                    "true": True,
                    "false": False,
                    "1": True,
                    "0": False,
                }
            )
        )

        if df[column].isna().any():
            fail(
                f"Column '{column}' contains values that could not "
                "be interpreted as boolean."
            )

    df["prediction_timestamp"] = pd.to_datetime(
        df["prediction_timestamp"],
        errors="coerce",
    )

    if df["prediction_timestamp"].isna().any():
        fail(
            "Prediction dataset contains invalid "
            "prediction_timestamp values."
        )

    df["time_to_start_time_minutes"] = pd.to_numeric(
        df["time_to_start_time_minutes"],
        errors="coerce",
    )

    return df


# ---------------------------------------------------------------------
# General target analysis
# ---------------------------------------------------------------------

def analyze_overall_targets(df: pd.DataFrame) -> dict:
    """Calculate overall target counts and rates."""

    total = len(df)

    future_positive = int(
        df["future_hyperglycemia"].sum()
    )

    pre_event_positive = int(
        df["pre_event_target"].sum()
    )

    event_underway = int(
        (df["event_status"] == "event_underway").sum()
    )

    return {
        "total_rows": total,
        "participants": int(df["person_id"].nunique()),

        "future_hyperglycemia": {
            "positive": future_positive,
            "negative": total - future_positive,
            "positive_rate": (
                future_positive / total
                if total
                else 0.0
            ),
        },

        "pre_event_target": {
            "positive": pre_event_positive,
            "negative": total - pre_event_positive,
            "positive_rate": (
                pre_event_positive / total
                if total
                else 0.0
            ),
        },

        "event_underway_rows": event_underway,

        "prediction_class_counts": {
            str(k): int(v)
            for k, v in (
                df["prediction_class"]
                .value_counts()
                .items()
            )
        },

        "event_status_counts": {
            str(k): int(v)
            for k, v in (
                df["event_status"]
                .value_counts()
                .items()
            )
        },
    }


# ---------------------------------------------------------------------
# Diabetes-group analysis
# ---------------------------------------------------------------------

def analyze_by_diabetes_type(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize target distribution by diabetes type."""

    grouped = (
        df.groupby("diabetes_type")
        .agg(
            rows=("person_id", "size"),
            participants=("person_id", "nunique"),
            future_positive=("future_hyperglycemia", "sum"),
            pre_event_positive=("pre_event_target", "sum"),
            event_underway=(
                "event_status",
                lambda x: (x == "event_underway").sum(),
            ),
        )
        .reset_index()
    )

    grouped["future_positive_rate"] = (
        grouped["future_positive"]
        / grouped["rows"]
    )

    grouped["pre_event_positive_rate"] = (
        grouped["pre_event_positive"]
        / grouped["rows"]
    )

    return grouped


# ---------------------------------------------------------------------
# T2D participant analysis
# ---------------------------------------------------------------------

def analyze_t2d_participants(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Analyze target distribution for each T2D participant."""

    t2d = df[
        df["diabetes_type"] == T2D_LABEL
    ].copy()

    if t2d.empty:
        return pd.DataFrame(
            columns=[
                "person_id",
                "rows",
                "future_positive",
                "pre_event_positive",
                "event_underway",
                "unique_events",
                "future_positive_rate",
                "pre_event_positive_rate",
            ]
        )

    result = (
        t2d.groupby("person_id")
        .agg(
            rows=("person_id", "size"),
            future_positive=(
                "future_hyperglycemia",
                "sum",
            ),
            pre_event_positive=(
                "pre_event_target",
                "sum",
            ),
            event_underway=(
                "event_status",
                lambda x: (x == "event_underway").sum(),
            ),
            unique_events=(
                "target_event_id",
                lambda x: x.dropna().nunique(),
            ),
        )
        .reset_index()
    )

    result["future_positive_rate"] = (
        result["future_positive"]
        / result["rows"]
    )

    result["pre_event_positive_rate"] = (
        result["pre_event_positive"]
        / result["rows"]
    )

    return result.sort_values(
        "pre_event_positive",
        ascending=False,
    ).reset_index(drop=True)


# ---------------------------------------------------------------------
# Event-level analysis
# ---------------------------------------------------------------------

def analyze_events(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Analyze how many prediction rows belong to each target event.

    Only pre-event positive rows are used for the primary event
    dependence analysis.
    """

    pre_event = df[
        df["pre_event_target"]
        & df["target_event_id"].notna()
    ].copy()

    if pre_event.empty:
        return pd.DataFrame(), {
            "unique_events_with_pre_event_predictions": 0,
            "total_pre_event_rows": 0,
            "median_rows_per_event": 0,
            "mean_rows_per_event": 0,
            "max_rows_per_event": 0,
        }

    event_counts = (
        pre_event.groupby(
            [
                "person_id",
                "target_event_id",
            ]
        )
        .agg(
            pre_event_prediction_rows=(
                "person_id",
                "size",
            ),
            earliest_prediction=(
                "prediction_timestamp",
                "min",
            ),
            latest_prediction=(
                "prediction_timestamp",
                "max",
            ),
            earliest_time_to_event=(
                "time_to_start_time_minutes",
                "max",
            ),
            latest_time_to_event=(
                "time_to_start_time_minutes",
                "min",
            ),
        )
        .reset_index()
    )

    # Number of unique events represented.
    unique_events = len(event_counts)

    rows_per_event = event_counts[
        "pre_event_prediction_rows"
    ]

    summary = {
        "unique_events_with_pre_event_predictions": int(
            unique_events
        ),

        "total_pre_event_rows": int(
            len(pre_event)
        ),

        "median_rows_per_event": float(
            rows_per_event.median()
        ),

        "mean_rows_per_event": float(
            rows_per_event.mean()
        ),

        "max_rows_per_event": int(
            rows_per_event.max()
        ),

        "events_with_1_row": int(
            (rows_per_event == 1).sum()
        ),

        "events_with_2_to_5_rows": int(
            rows_per_event.between(2, 5).sum()
        ),

        "events_with_6_to_20_rows": int(
            rows_per_event.between(6, 20).sum()
        ),

        "events_with_more_than_20_rows": int(
            (rows_per_event > 20).sum()
        ),
    }

    return (
        event_counts.sort_values(
            "pre_event_prediction_rows",
            ascending=False,
        ).reset_index(drop=True),
        summary,
    )


# ---------------------------------------------------------------------
# Lead-time analysis
# ---------------------------------------------------------------------

def analyze_lead_time(df: pd.DataFrame) -> dict:
    """
    Analyze how far in advance the pre-event predictions occur.

    The bins are deliberately aligned with the 2-hour prediction
    horizon.
    """

    pre_event = df[
        df["pre_event_target"]
        & df["time_to_start_time_minutes"].notna()
    ].copy()

    if pre_event.empty:
        return {
            "rows": 0,
            "bins": {},
        }

    lead = pre_event[
        "time_to_start_time_minutes"
    ]

    bins = [
        -0.001,
        15,
        30,
        60,
        120,
    ]

    labels = [
        "0-15_minutes",
        "15-30_minutes",
        "30-60_minutes",
        "60-120_minutes",
    ]

    categorized = pd.cut(
        lead,
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    counts = (
        categorized
        .value_counts()
        .reindex(labels, fill_value=0)
    )

    percentages = (
        counts / len(pre_event)
    )

    return {
        "rows": int(len(pre_event)),

        "min_minutes": float(lead.min()),
        "median_minutes": float(lead.median()),
        "mean_minutes": float(lead.mean()),
        "max_minutes": float(lead.max()),

        "bins": {
            label: {
                "count": int(counts[label]),
                "percentage": float(
                    percentages[label]
                ),
            }
            for label in labels
        },
    }


# ---------------------------------------------------------------------
# Concentration analysis
# ---------------------------------------------------------------------

def analyze_concentration(
    t2d_participants: pd.DataFrame,
    event_counts: pd.DataFrame,
) -> dict:
    """
    Estimate whether positive examples are concentrated among a small
    number of participants or events.

    This is descriptive only. It is not a statistical significance test.
    """

    result = {}

    if not t2d_participants.empty:

        participant_values = (
            t2d_participants[
                "pre_event_positive"
            ]
            .sort_values(ascending=False)
            .reset_index(drop=True)
        )

        total = participant_values.sum()

        if total > 0:
            result["participant_concentration"] = {
                "top_1_share": float(
                    participant_values.iloc[0] / total
                ),
                "top_2_share": float(
                    participant_values.iloc[:2].sum()
                    / total
                ),
                "participants_with_pre_event_rows": int(
                    (participant_values > 0).sum()
                ),
            }

    if not event_counts.empty:

        event_values = (
            event_counts[
                "pre_event_prediction_rows"
            ]
            .sort_values(ascending=False)
            .reset_index(drop=True)
        )

        total = event_values.sum()

        if total > 0:
            result["event_concentration"] = {
                "top_1_share": float(
                    event_values.iloc[0] / total
                ),
                "top_5_share": float(
                    event_values.iloc[:5].sum()
                    / total
                ),
                "top_10_share": float(
                    event_values.iloc[:10].sum()
                    / total
                ),
                "events_with_pre_event_rows": int(
                    len(event_values)
                ),
            }

    return result


# ---------------------------------------------------------------------
# Build a compact report table
# ---------------------------------------------------------------------

def build_report_table(
    diabetes_summary: pd.DataFrame,
    t2d_summary: pd.DataFrame,
    event_counts: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create one flat CSV containing the most important audit rows.

    This makes the analysis easy to inspect in Excel/VS Code.
    """

    rows = []

    # Diabetes-group rows.
    for _, row in diabetes_summary.iterrows():

        rows.append(
            {
                "section": "diabetes_type",
                "entity": str(row["diabetes_type"]),
                "metric": "rows",
                "value": float(row["rows"]),
            }
        )

        rows.append(
            {
                "section": "diabetes_type",
                "entity": str(row["diabetes_type"]),
                "metric": "future_positive",
                "value": float(row["future_positive"]),
            }
        )

        rows.append(
            {
                "section": "diabetes_type",
                "entity": str(row["diabetes_type"]),
                "metric": "pre_event_positive",
                "value": float(row["pre_event_positive"]),
            }
        )

        rows.append(
            {
                "section": "diabetes_type",
                "entity": str(row["diabetes_type"]),
                "metric": "pre_event_positive_rate",
                "value": float(
                    row["pre_event_positive_rate"]
                ),
            }
        )

    # T2D participant rows.
    for _, row in t2d_summary.iterrows():

        rows.append(
            {
                "section": "t2d_participant",
                "entity": str(row["person_id"]),
                "metric": "pre_event_positive",
                "value": float(
                    row["pre_event_positive"]
                ),
            }
        )

        rows.append(
            {
                "section": "t2d_participant",
                "entity": str(row["person_id"]),
                "metric": "unique_events",
                "value": float(
                    row["unique_events"]
                ),
            }
        )

    # Event rows.
    for _, row in event_counts.iterrows():

        rows.append(
            {
                "section": "event",
                "entity": str(
                    row["target_event_id"]
                ),
                "metric": "pre_event_prediction_rows",
                "value": float(
                    row["pre_event_prediction_rows"]
                ),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    print("=" * 72)
    print("Hall 2018 — Supervised Prediction Dataset Audit")
    print("=" * 72)
    print()

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_dataset()

    print(
        f"Prediction rows loaded:      {len(df):,}"
    )

    print(
        f"Participants represented:    "
        f"{df['person_id'].nunique():,}"
    )

    print()

    # -------------------------------------------------------------
    # Overall
    # -------------------------------------------------------------

    overall = analyze_overall_targets(df)

    # -------------------------------------------------------------
    # Diabetes groups
    # -------------------------------------------------------------

    diabetes_summary = analyze_by_diabetes_type(df)

    # -------------------------------------------------------------
    # T2D participants
    # -------------------------------------------------------------

    t2d_summary = analyze_t2d_participants(df)

    # -------------------------------------------------------------
    # Events
    # -------------------------------------------------------------

    event_counts, event_summary = analyze_events(df)

    # -------------------------------------------------------------
    # Lead time
    # -------------------------------------------------------------

    lead_time = analyze_lead_time(df)

    # -------------------------------------------------------------
    # Concentration
    # -------------------------------------------------------------

    concentration = analyze_concentration(
        t2d_summary,
        event_counts,
    )

    # -------------------------------------------------------------
    # Human-readable output
    # -------------------------------------------------------------

    print("-" * 72)
    print("OVERALL TARGETS")
    print("-" * 72)

    print(
        f"Future-positive windows:     "
        f"{overall['future_hyperglycemia']['positive']:,} "
        f"({overall['future_hyperglycemia']['positive_rate']:.2%})"
    )

    print(
        f"Pre-event positive windows:  "
        f"{overall['pre_event_target']['positive']:,} "
        f"({overall['pre_event_target']['positive_rate']:.2%})"
    )

    print(
        f"Event-underway windows:      "
        f"{overall['event_underway_rows']:,}"
    )

    print()

    print("-" * 72)
    print("TARGETS BY DIABETES TYPE")
    print("-" * 72)

    print(
        diabetes_summary.to_string(
            index=False,
            formatters={
                "future_positive_rate": "{:.2%}".format,
                "pre_event_positive_rate": "{:.2%}".format,
            },
        )
    )

    print()

    print("-" * 72)
    print("T2D PARTICIPANT BREAKDOWN")
    print("-" * 72)

    if t2d_summary.empty:
        print("No T2D rows found.")

    else:
        print(
            t2d_summary.to_string(
                index=False,
                formatters={
                    "future_positive_rate": "{:.2%}".format,
                    "pre_event_positive_rate": "{:.2%}".format,
                },
            )
        )

    print()

    print("-" * 72)
    print("EVENT-LEVEL DEPENDENCE")
    print("-" * 72)

    print(
        f"Unique events represented by pre-event rows: "
        f"{event_summary['unique_events_with_pre_event_predictions']:,}"
    )

    print(
        f"Pre-event rows: "
        f"{event_summary['total_pre_event_rows']:,}"
    )

    print(
        f"Median rows per event: "
        f"{event_summary['median_rows_per_event']:.1f}"
    )

    print(
        f"Mean rows per event: "
        f"{event_summary['mean_rows_per_event']:.1f}"
    )

    print(
        f"Maximum rows for one event: "
        f"{event_summary['max_rows_per_event']:,}"
    )

    print(
        f"Events with 1 row: "
        f"{event_summary['events_with_1_row']:,}"
    )

    print(
        f"Events with 2-5 rows: "
        f"{event_summary['events_with_2_to_5_rows']:,}"
    )

    print(
        f"Events with 6-20 rows: "
        f"{event_summary['events_with_6_to_20_rows']:,}"
    )

    print(
        f"Events with >20 rows: "
        f"{event_summary['events_with_more_than_20_rows']:,}"
    )

    print()

    print("-" * 72)
    print("PRE-EVENT LEAD TIME")
    print("-" * 72)

    if lead_time["rows"]:

        print(
            f"Minimum:  {lead_time['min_minutes']:.1f} min"
        )

        print(
            f"Median:   {lead_time['median_minutes']:.1f} min"
        )

        print(
            f"Mean:     {lead_time['mean_minutes']:.1f} min"
        )

        print(
            f"Maximum:  {lead_time['max_minutes']:.1f} min"
        )

        print()

        for label, values in (
            lead_time["bins"].items()
        ):
            print(
                f"  {label:<18} "
                f"{values['count']:>6,} "
                f"({values['percentage']:.2%})"
            )

    else:
        print("No pre-event rows available.")

    print()

    print("-" * 72)
    print("CONCENTRATION")
    print("-" * 72)

    participant_concentration = concentration.get(
        "participant_concentration"
    )

    if participant_concentration:

        print(
            "Top T2D participant share of pre-event rows: "
            f"{participant_concentration['top_1_share']:.2%}"
        )

        print(
            "Top 2 T2D participants share:                "
            f"{participant_concentration['top_2_share']:.2%}"
        )

        print(
            "T2D participants with pre-event rows:         "
            f"{participant_concentration['participants_with_pre_event_rows']}"
        )

    event_concentration = concentration.get(
        "event_concentration"
    )

    if event_concentration:

        print(
            "Top event share of pre-event rows:            "
            f"{event_concentration['top_1_share']:.2%}"
        )

        print(
            "Top 5 events share:                            "
            f"{event_concentration['top_5_share']:.2%}"
        )

        print(
            "Top 10 events share:                           "
            f"{event_concentration['top_10_share']:.2%}"
        )

    # -------------------------------------------------------------
    # Determine a preliminary status
    # -------------------------------------------------------------

    unique_events = event_summary[
        "unique_events_with_pre_event_predictions"
    ]

    t2d_participants_with_positive = 0

    if not t2d_summary.empty:
        t2d_participants_with_positive = int(
            (
                t2d_summary["pre_event_positive"] > 0
            ).sum()
        )

    if unique_events == 0:
        preliminary_status = (
            "NOT VIABLE YET — no pre-event event links found."
        )

    elif unique_events < 10:
        preliminary_status = (
            "CAUTION — very few independent events represented."
        )

    elif t2d_participants_with_positive < 2:
        preliminary_status = (
            "CAUTION — pre-event positives are concentrated "
            "in too few T2D participants."
        )

    else:
        preliminary_status = (
            "PROCEED TO TARGET REVIEW — event and participant "
            "structure should now be used to finalize the modeling target."
        )

    print()
    print("-" * 72)
    print("PRELIMINARY STATUS")
    print("-" * 72)
    print(preliminary_status)

    # -------------------------------------------------------------
    # Save JSON
    # -------------------------------------------------------------

    json_report = {
        "input": str(INPUT_DATASET),

        "overall": overall,

        "diabetes_type_summary": (
            diabetes_summary
            .to_dict(orient="records")
        ),

        "t2d_participant_summary": (
            t2d_summary
            .to_dict(orient="records")
        ),

        "event_summary": event_summary,

        "lead_time": lead_time,

        "concentration": concentration,

        "preliminary_status": preliminary_status,
    }

    try:
        with OUTPUT_JSON.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                json_report,
                f,
                indent=2,
                default=str,
            )

    except Exception as exc:
        fail(
            f"Could not write JSON report: {exc}"
        )

    # -------------------------------------------------------------
    # Save compact CSV
    # -------------------------------------------------------------

    report_table = build_report_table(
        diabetes_summary,
        t2d_summary,
        event_counts,
    )

    try:
        report_table.to_csv(
            OUTPUT_CSV,
            index=False,
        )

    except Exception as exc:
        fail(
            f"Could not write CSV report: {exc}"
        )

    print()
    print("-" * 72)
    print("OUTPUT FILES")
    print("-" * 72)

    print(OUTPUT_CSV)
    print(OUTPUT_JSON)

    print()
    print("=" * 72)
    print("AUDIT COMPLETE — NO MODEL TRAINING PERFORMED")
    print("=" * 72)


if __name__ == "__main__":
    main()
from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

OOF_PATH = Path(
    "reports/hall2018_baseline_oof_predictions.csv"
)

PREDICTION_DATASET_PATH = Path(
    "reports/hall2018_prediction_dataset.csv"
)

EVENTS_PATH = Path(
    "reports/hall2018_hyperglycemic_events.csv"
)

OUTPUT_EVENTS = Path(
    "reports/hall2018_event_level_predictions.csv"
)

OUTPUT_LEAD_TIME = Path(
    "reports/hall2018_event_lead_time_summary.csv"
)

OUTPUT_SUMMARY = Path(
    "reports/hall2018_event_level_prediction_summary.json"
)

PERSON_ID = "person_id"
PREDICTION_TIMESTAMP = "prediction_timestamp"
TARGET_EVENT_ID = "target_event_id"
PRE_EVENT_TARGET = "pre_event_target"
TRAJECTORY_PROBABILITY = "trajectory_probability"

EVENT_START = "start_time"
EVENT_END = "end_time"

# Probability thresholds to examine.
#
# These are NOT being selected as the final alert threshold.
# They allow us to see how event detection changes as the threshold changes.
THRESHOLDS = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
]

# Lead-time requirements.
LEAD_TIME_WINDOWS = [
    15,
    30,
    60,
    120,
]


# ============================================================================
# HELPERS
# ============================================================================

def fail(message):
    raise RuntimeError(
        f"\nERROR: {message}"
    )


def print_section(title):
    print("\n" + "-" * 76)
    print(title)
    print("-" * 76)


def require_file(path):
    if not path.exists():
        fail(
            "Required file not found:\n"
            f"{path.resolve()}"
        )


def require_columns(df, columns, name):
    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        fail(
            f"{name} is missing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )


def safe_numeric(series):
    return pd.to_numeric(
        series,
        errors="coerce",
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 76)
    print(
        "Hall 2018 — Event-Level Prediction Evaluation"
    )
    print("=" * 76)

    # ------------------------------------------------------------------------
    # CHECK INPUT FILES
    # ------------------------------------------------------------------------

    print_section(
        "CHECKING INPUT FILES"
    )

    require_file(OOF_PATH)
    require_file(PREDICTION_DATASET_PATH)
    require_file(EVENTS_PATH)

    print(
        f"OOF predictions:             "
        f"{OOF_PATH.resolve()}"
    )

    print(
        f"Prediction dataset:          "
        f"{PREDICTION_DATASET_PATH.resolve()}"
    )

    print(
        f"Events:                       "
        f"{EVENTS_PATH.resolve()}"
    )

    # ------------------------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------------------------

    print_section(
        "LOADING DATA"
    )

    oof = pd.read_csv(
        OOF_PATH
    )

    prediction = pd.read_csv(
        PREDICTION_DATASET_PATH
    )

    events = pd.read_csv(
        EVENTS_PATH
    )

    print(
        f"OOF rows:                    "
        f"{len(oof):,}"
    )

    print(
        f"Prediction rows:             "
        f"{len(prediction):,}"
    )

    print(
        f"Event rows:                  "
        f"{len(events):,}"
    )

    # ------------------------------------------------------------------------
    # REQUIRED COLUMNS
    # ------------------------------------------------------------------------

    require_columns(
        oof,
        [
            PERSON_ID,
            PRE_EVENT_TARGET,
            TRAJECTORY_PROBABILITY,
        ],
        "OOF prediction dataset",
    )

    require_columns(
        prediction,
        [
            PERSON_ID,
            PREDICTION_TIMESTAMP,
            PRE_EVENT_TARGET,
            TARGET_EVENT_ID,
        ],
        "prediction dataset",
    )

    require_columns(
        events,
        [
            PERSON_ID,
            "event_id",
            EVENT_START,
            EVENT_END,
        ],
        "event dataset",
    )

    # ------------------------------------------------------------------------
    # PREPARE OOF DATA
    # ------------------------------------------------------------------------

    print_section(
        "PREPARING PREDICTION DATA"
    )

    oof[PREDICTION_TIMESTAMP] = pd.to_datetime(
        oof.get(
            PREDICTION_TIMESTAMP,
            pd.Series(
                pd.NaT,
                index=oof.index,
            ),
        ),
        errors="coerce",
    )

    prediction[PREDICTION_TIMESTAMP] = (
        pd.to_datetime(
            prediction[PREDICTION_TIMESTAMP],
            errors="coerce",
        )
    )

    events[EVENT_START] = pd.to_datetime(
        events[EVENT_START],
        errors="coerce",
    )

    events[EVENT_END] = pd.to_datetime(
        events[EVENT_END],
        errors="coerce",
    )

    oof[TRAJECTORY_PROBABILITY] = safe_numeric(
        oof[TRAJECTORY_PROBABILITY]
    )

    prediction[PRE_EVENT_TARGET] = (
        safe_numeric(
            prediction[PRE_EVENT_TARGET]
        )
    )

    # ------------------------------------------------------------------------
    # ALIGN OOF PREDICTIONS WITH ORIGINAL PREDICTION DATASET
    # ------------------------------------------------------------------------

    print_section(
        "ALIGNING OOF PREDICTIONS"
    )

    # The OOF file contains participant + target + probability and may also
    # contain prediction_timestamp depending on the source dataset.
    #
    # If timestamp exists, use it for an explicit one-to-one alignment.
    # Otherwise, preserve row order because the OOF script constructed the
    # predictions directly from the feature dataframe.

    if (
        PREDICTION_TIMESTAMP in oof.columns
        and oof[PREDICTION_TIMESTAMP].notna().all()
    ):

        merge_keys = [
            PERSON_ID,
            PREDICTION_TIMESTAMP,
        ]

        merged = prediction.merge(
            oof[
                merge_keys
                + [
                    TRAJECTORY_PROBABILITY
                ]
            ],
            on=merge_keys,
            how="left",
            validate="one_to_one",
        )

        alignment_mode = (
            "person_id + prediction_timestamp"
        )

    else:

        if len(oof) != len(prediction):
            fail(
                "OOF and prediction datasets have different "
                "row counts and OOF predictions do not contain "
                "prediction timestamps."
            )

        merged = prediction.copy()

        merged[
            TRAJECTORY_PROBABILITY
        ] = oof[
            TRAJECTORY_PROBABILITY
        ].values

        alignment_mode = (
            "row-order alignment"
        )

    missing_probabilities = int(
        merged[
            TRAJECTORY_PROBABILITY
        ].isna().sum()
    )

    if missing_probabilities > 0:
        fail(
            f"{missing_probabilities:,} prediction rows "
            "have no trajectory probability after alignment."
        )

    print(
        f"Alignment method:           "
        f"{alignment_mode}"
    )

    print(
        f"Aligned prediction rows:     "
        f"{len(merged):,}"
    )

    # ------------------------------------------------------------------------
    # IDENTIFY EVENT-WINDOW PREDICTIONS
    # ------------------------------------------------------------------------

    print_section(
        "IDENTIFYING EVENT-RELATED PREDICTIONS"
    )

    pre_event = merged[
        merged[PRE_EVENT_TARGET] == 1
    ].copy()

    pre_event[
        TARGET_EVENT_ID
    ] = pre_event[
        TARGET_EVENT_ID
    ].astype(str)

    # Exclude missing/empty event IDs.
    pre_event = pre_event[
        ~pre_event[TARGET_EVENT_ID].isin(
            [
                "",
                "nan",
                "None",
            ]
        )
    ].copy()

    print(
        f"Pre-event prediction rows:   "
        f"{len(pre_event):,}"
    )

    print(
        f"Distinct target events:      "
        f"{pre_event[TARGET_EVENT_ID].nunique():,}"
    )

    # ------------------------------------------------------------------------
    # BUILD EVENT LOOKUP
    # ------------------------------------------------------------------------

    print_section(
        "BUILDING EVENT LOOKUP"
    )

    event_lookup = events[
        [
            PERSON_ID,
            "event_id",
            EVENT_START,
            EVENT_END,
        ]
    ].copy()

    event_lookup["event_id"] = (
        event_lookup["event_id"]
        .astype(str)
    )

    event_lookup = event_lookup.rename(
        columns={
            "event_id": TARGET_EVENT_ID,
        }
    )

    # Check event IDs are unique.
    duplicate_event_ids = (
        event_lookup[
            TARGET_EVENT_ID
        ]
        .duplicated()
        .sum()
    )

    if duplicate_event_ids > 0:
        fail(
            f"Found {duplicate_event_ids} duplicate event IDs "
            "in the event dataset."
        )

    pre_event = pre_event.merge(
        event_lookup,
        on=[
            PERSON_ID,
            TARGET_EVENT_ID,
        ],
        how="left",
        validate="many_to_one",
    )

    missing_event_start = int(
        pre_event[
            EVENT_START
        ].isna().sum()
    )

    if missing_event_start > 0:
        fail(
            f"{missing_event_start:,} pre-event prediction rows "
            "could not be matched to an event start."
        )

    # ------------------------------------------------------------------------
    # CALCULATE LEAD TIME
    # ------------------------------------------------------------------------

    print_section(
        "CALCULATING EVENT LEAD TIMES"
    )

    pre_event["lead_time_minutes"] = (
        (
            pre_event[EVENT_START]
            - pre_event[PREDICTION_TIMESTAMP]
        )
        .dt.total_seconds()
        / 60.0
    )

    # A pre-event row should be before the event start.
    invalid_lead_times = pre_event[
        pre_event["lead_time_minutes"] <= 0
    ]

    if len(invalid_lead_times) > 0:
        fail(
            f"Found {len(invalid_lead_times):,} pre-event "
            "rows with non-positive lead time."
        )

    print(
        f"Minimum lead time:           "
        f"{pre_event['lead_time_minutes'].min():.1f} min"
    )

    print(
        f"Median lead time:             "
        f"{pre_event['lead_time_minutes'].median():.1f} min"
    )

    print(
        f"Maximum lead time:             "
        f"{pre_event['lead_time_minutes'].max():.1f} min"
    )

    # ------------------------------------------------------------------------
    # EVENT-LEVEL ANALYSIS
    # ------------------------------------------------------------------------

    print_section(
        "EVENT-LEVEL DETECTION"
    )

    event_records = []

    grouped_events = pre_event.groupby(
        [
            PERSON_ID,
            TARGET_EVENT_ID,
        ],
        sort=False,
    )

    for (
        person_id,
        event_id,
    ), group in grouped_events:

        event_start = group[
            EVENT_START
        ].iloc[0]

        event_end = group[
            EVENT_END
        ].iloc[0]

        record = {
            PERSON_ID: person_id,
            TARGET_EVENT_ID: event_id,
            "event_start": event_start,
            "event_end": event_end,
            "event_prediction_rows": int(
                len(group)
            ),
            "earliest_prediction_timestamp": (
                group[
                    PREDICTION_TIMESTAMP
                ].min()
            ),
            "latest_prediction_timestamp": (
                group[
                    PREDICTION_TIMESTAMP
                ].max()
            ),
            "maximum_probability": float(
                group[
                    TRAJECTORY_PROBABILITY
                ].max()
            ),
            "mean_probability": float(
                group[
                    TRAJECTORY_PROBABILITY
                ].mean()
            ),
            "earliest_lead_time_minutes": float(
                group[
                    "lead_time_minutes"
                ].max()
            ),
            "latest_lead_time_minutes": float(
                group[
                    "lead_time_minutes"
                ].min()
            ),
        }

        # --------------------------------------------------------------------
        # THRESHOLD-SPECIFIC EVENT DETECTION
        # --------------------------------------------------------------------

        for threshold in THRESHOLDS:

            threshold_mask = (
                group[
                    TRAJECTORY_PROBABILITY
                ] >= threshold
            )

            detected = bool(
                threshold_mask.any()
            )

            prefix = (
                f"threshold_{threshold:.2f}"
            )

            record[
                f"{prefix}_detected"
            ] = detected

            if detected:

                detected_rows = group[
                    threshold_mask
                ]

                # Earliest warning = largest lead time.
                record[
                    f"{prefix}_earliest_lead_minutes"
                ] = float(
                    detected_rows[
                        "lead_time_minutes"
                    ].max()
                )

                # First chronological alert before event.
                earliest_alert = (
                    detected_rows
                    .sort_values(
                        PREDICTION_TIMESTAMP
                    )
                    .iloc[0]
                )

                record[
                    f"{prefix}_first_alert_timestamp"
                ] = earliest_alert[
                    PREDICTION_TIMESTAMP
                ]

                record[
                    f"{prefix}_alert_count"
                ] = int(
                    len(detected_rows)
                )

            else:

                record[
                    f"{prefix}_earliest_lead_minutes"
                ] = np.nan

                record[
                    f"{prefix}_first_alert_timestamp"
                ] = pd.NaT

                record[
                    f"{prefix}_alert_count"
                ] = 0

        # --------------------------------------------------------------------
        # LEAD-TIME DETECTION
        # --------------------------------------------------------------------

        for lead_window in LEAD_TIME_WINDOWS:

            # A warning qualifies if there is at least one prediction
            # between `lead_window` minutes before the event and the event.
            qualifying = group[
                group[
                    "lead_time_minutes"
                ] >= lead_window
            ]

            record[
                f"detected_at_least_{lead_window}m"
            ] = bool(
                len(qualifying) > 0
            )

            if len(qualifying) > 0:

                record[
                    f"best_lead_at_least_{lead_window}m"
                ] = float(
                    qualifying[
                        "lead_time_minutes"
                    ].max()
                )

            else:

                record[
                    f"best_lead_at_least_{lead_window}m"
                ] = np.nan

        event_records.append(
            record
        )

    event_results = pd.DataFrame(
        event_records
    )

    if len(event_results) == 0:
        fail(
            "No event-level records were produced."
        )

    # ------------------------------------------------------------------------
    # EVENT SUMMARY
    # ------------------------------------------------------------------------

    print_section(
        "EVENT-LEVEL SUMMARY"
    )

    total_events = len(
        event_results
    )

    print(
        f"Distinct events evaluated:   "
        f"{total_events:,}"
    )

    print()

    for threshold in THRESHOLDS:

        column = (
            f"threshold_{threshold:.2f}_detected"
        )

        detected_events = int(
            event_results[column].sum()
        )

        detection_rate = (
            detected_events
            / total_events
        )

        print(
            f"Threshold {threshold:.2f}: "
            f"{detected_events:,}/{total_events:,} "
            f"events detected "
            f"({detection_rate * 100:.1f}%)"
        )

    print()

    for lead_window in LEAD_TIME_WINDOWS:

        column = (
            f"detected_at_least_{lead_window}m"
        )

        detected_events = int(
            event_results[column].sum()
        )

        detection_rate = (
            detected_events
            / total_events
        )

        print(
            f">= {lead_window:3d} min warning: "
            f"{detected_events:,}/{total_events:,} "
            f"events "
            f"({detection_rate * 100:.1f}%)"
        )

    # ------------------------------------------------------------------------
    # EVENT DETECTION DETAILS
    # ------------------------------------------------------------------------

    print_section(
        "EVENT WARNING DEPTH"
    )

    print(
        f"Median prediction rows/event: "
        f"{event_results['event_prediction_rows'].median():.1f}"
    )

    print(
        f"Mean prediction rows/event:   "
        f"{event_results['event_prediction_rows'].mean():.1f}"
    )

    print(
        f"Maximum rows/event:            "
        f"{event_results['event_prediction_rows'].max():,}"
    )

    print(
        f"Median maximum probability:    "
        f"{event_results['maximum_probability'].median():.4f}"
    )

    print(
        f"Maximum event probability:     "
        f"{event_results['maximum_probability'].max():.4f}"
    )

    print(
        f"Median earliest warning:       "
        f"{event_results['earliest_lead_time_minutes'].median():.1f} min"
    )

    print(
        f"Mean earliest warning:         "
        f"{event_results['earliest_lead_time_minutes'].mean():.1f} min"
    )

    # ------------------------------------------------------------------------
    # LEAD-TIME SUMMARY TABLE
    # ------------------------------------------------------------------------

    lead_records = []

    for lead_window in LEAD_TIME_WINDOWS:

        detected_column = (
            f"detected_at_least_{lead_window}m"
        )

        lead_values = event_results[
            event_results[
                detected_column
            ]
        ][
            f"best_lead_at_least_{lead_window}m"
        ].dropna()

        detected_events = int(
            len(lead_values)
        )

        record = {
            "required_lead_time_minutes": (
                lead_window
            ),
            "total_events": (
                total_events
            ),
            "detected_events": (
                detected_events
            ),
            "detection_rate": (
                detected_events
                / total_events
            ),
        }

        if detected_events > 0:

            record[
                "median_actual_lead_time_minutes"
            ] = float(
                lead_values.median()
            )

            record[
                "mean_actual_lead_time_minutes"
            ] = float(
                lead_values.mean()
            )

            record[
                "minimum_actual_lead_time_minutes"
            ] = float(
                lead_values.min()
            )

            record[
                "maximum_actual_lead_time_minutes"
            ] = float(
                lead_values.max()
            )

        else:

            record[
                "median_actual_lead_time_minutes"
            ] = np.nan

            record[
                "mean_actual_lead_time_minutes"
            ] = np.nan

            record[
                "minimum_actual_lead_time_minutes"
            ] = np.nan

            record[
                "maximum_actual_lead_time_minutes"
            ] = np.nan

        lead_records.append(
            record
        )

    lead_summary = pd.DataFrame(
        lead_records
    )

    print_section(
        "LEAD-TIME SUMMARY"
    )

    print(
        lead_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    # ------------------------------------------------------------------------
    # SAVE EVENT RESULTS
    # ------------------------------------------------------------------------

    print_section(
        "SAVING RESULTS"
    )

    OUTPUT_EVENTS.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    event_results.to_csv(
        OUTPUT_EVENTS,
        index=False,
    )

    lead_summary.to_csv(
        OUTPUT_LEAD_TIME,
        index=False,
    )

    print(
        f"Event-level results:         "
        f"{OUTPUT_EVENTS.resolve()}"
    )

    print(
        f"Lead-time summary:           "
        f"{OUTPUT_LEAD_TIME.resolve()}"
    )

    # ------------------------------------------------------------------------
    # SUMMARY JSON
    # ------------------------------------------------------------------------

    threshold_summary = {}

    for threshold in THRESHOLDS:

        detected_column = (
            f"threshold_{threshold:.2f}_detected"
        )

        lead_column = (
            f"threshold_{threshold:.2f}_earliest_lead_minutes"
        )

        detected = event_results[
            detected_column
        ]

        detected_lead = event_results.loc[
            detected,
            lead_column,
        ].dropna()

        threshold_summary[
            f"{threshold:.2f}"
        ] = {
            "detected_events": int(
                detected.sum()
            ),
            "total_events": int(
                total_events
            ),
            "detection_rate": float(
                detected.mean()
            ),
            "median_earliest_lead_minutes": (
                float(
                    detected_lead.median()
                )
                if len(detected_lead) > 0
                else None
            ),
            "mean_earliest_lead_minutes": (
                float(
                    detected_lead.mean()
                )
                if len(detected_lead) > 0
                else None
            ),
        }

    summary = {
        "inputs": {
            "oof_predictions": str(
                OOF_PATH
            ),
            "prediction_dataset": str(
                PREDICTION_DATASET_PATH
            ),
            "events": str(
                EVENTS_PATH
            ),
        },
        "alignment": {
            "method": alignment_mode,
            "aligned_rows": int(
                len(merged)
            ),
        },
        "event_population": {
            "distinct_events_evaluated": int(
                total_events
            ),
            "pre_event_prediction_rows": int(
                len(pre_event)
            ),
        },
        "threshold_results": (
            threshold_summary
        ),
        "lead_time_results": (
            lead_summary.to_dict(
                orient="records"
            )
        ),
        "overall_warning_depth": {
            "median_rows_per_event": float(
                event_results[
                    "event_prediction_rows"
                ].median()
            ),
            "mean_rows_per_event": float(
                event_results[
                    "event_prediction_rows"
                ].mean()
            ),
            "median_earliest_warning_minutes": float(
                event_results[
                    "earliest_lead_time_minutes"
                ].median()
            ),
            "mean_earliest_warning_minutes": float(
                event_results[
                    "earliest_lead_time_minutes"
                ].mean()
            ),
        },
        "methodology_notes": [
            "Evaluation is performed at the distinct hyperglycemic-event level.",
            "An event is considered detected when at least one pre-event prediction reaches the specified probability threshold.",
            "Earliest warning means the prediction furthest before event start that satisfies the threshold.",
            "This avoids treating overlapping prediction windows for one event as independent successful events.",
            "Thresholds are evaluated for characterization and are not declared clinical alert thresholds.",
            "Event-underway prediction rows are excluded from the early-warning event detection calculation.",
        ],
    }

    with open(
        OUTPUT_SUMMARY,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            default=str,
        )

    print(
        f"Summary JSON:                "
        f"{OUTPUT_SUMMARY.resolve()}"
    )

    print("\n" + "=" * 76)
    print(
        "EVENT-LEVEL PREDICTION EVALUATION COMPLETE"
    )
    print("=" * 76)


if __name__ == "__main__":
    main()
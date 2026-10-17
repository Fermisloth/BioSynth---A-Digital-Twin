from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

PREDICTIONS_PATH = Path(
    "reports/hall2018_baseline_oof_predictions.csv"
)

OUTPUT_THRESHOLDS = Path(
    "reports/hall2018_threshold_analysis.csv"
)

OUTPUT_SUMMARY = Path(
    "reports/hall2018_threshold_analysis_summary.json"
)

TARGET = "pre_event_target"
PROBABILITY_COLUMN = "trajectory_probability"

# Evaluate a dense set of thresholds.
THRESHOLD_START = 0.01
THRESHOLD_END = 0.50
THRESHOLD_STEP = 0.01


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


def evaluate_threshold(
    y_true,
    probabilities,
    threshold,
):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    total = len(y_true)
    actual_positive = int(np.sum(y_true))
    actual_negative = int(total - actual_positive)

    predicted_positive = int(
        np.sum(predictions)
    )

    predicted_negative = int(
        total - predicted_positive
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    false_positive_rate = (
        fp / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    alert_rate = (
        predicted_positive / total
    )

    capture_rate = (
        tp / actual_positive
        if actual_positive > 0
        else np.nan
    )

    # How much more likely a flagged row is to be positive
    # compared with the underlying positive rate.
    base_rate = (
        actual_positive / total
    )

    lift = (
        precision / base_rate
        if base_rate > 0
        else np.nan
    )

    return {
        "threshold": float(threshold),
        "total_rows": int(total),
        "actual_positive_rows": actual_positive,
        "actual_negative_rows": actual_negative,
        "predicted_alert_rows": predicted_positive,
        "predicted_non_alert_rows": predicted_negative,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "false_positive_rate": float(
            false_positive_rate
        ),
        "alert_rate": float(alert_rate),
        "capture_rate": float(capture_rate),
        "lift_vs_base_rate": float(lift),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 76)
    print(
        "Hall 2018 — Prediction Threshold Analysis"
    )
    print("=" * 76)

    # ------------------------------------------------------------------------
    # LOAD OOF PREDICTIONS
    # ------------------------------------------------------------------------

    print_section(
        "LOADING OUT-OF-FOLD PREDICTIONS"
    )

    if not PREDICTIONS_PATH.exists():
        fail(
            "OOF prediction file not found:\n"
            f"{PREDICTIONS_PATH.resolve()}"
        )

    df = pd.read_csv(
        PREDICTIONS_PATH
    )

    print(
        f"Rows loaded:                 "
        f"{len(df):,}"
    )

    required_columns = [
        TARGET,
        PROBABILITY_COLUMN,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        fail(
            "Required columns are missing:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing_columns
            )
        )

    # ------------------------------------------------------------------------
    # VALIDATE PREDICTIONS
    # ------------------------------------------------------------------------

    print_section(
        "PREDICTION VALIDATION"
    )

    if df[TARGET].isna().any():
        fail(
            "Target contains missing values."
        )

    if df[PROBABILITY_COLUMN].isna().any():
        fail(
            "Trajectory probabilities contain missing values."
        )

    probabilities = (
        df[PROBABILITY_COLUMN]
        .astype(float)
        .values
    )

    y_true = (
        df[TARGET]
        .astype(int)
        .values
    )

    if np.any(probabilities < 0) or np.any(
        probabilities > 1
    ):
        fail(
            "Trajectory probabilities contain "
            "values outside [0, 1]."
        )

    if not set(
        np.unique(y_true)
    ).issubset({0, 1}):
        fail(
            "Target contains values other than 0 and 1."
        )

    positive_count = int(
        np.sum(y_true)
    )

    negative_count = int(
        len(y_true) - positive_count
    )

    base_rate = (
        positive_count / len(y_true)
    )

    print(
        f"Positive rows:               "
        f"{positive_count:,}"
    )

    print(
        f"Negative rows:               "
        f"{negative_count:,}"
    )

    print(
        f"Base positive rate:          "
        f"{base_rate * 100:.2f}%"
    )

    print(
        f"Minimum probability:         "
        f"{probabilities.min():.6f}"
    )

    print(
        f"Maximum probability:         "
        f"{probabilities.max():.6f}"
    )

    print(
        f"Median probability:          "
        f"{np.median(probabilities):.6f}"
    )

    # ------------------------------------------------------------------------
    # THRESHOLD GRID
    # ------------------------------------------------------------------------

    print_section(
        "EVALUATING THRESHOLDS"
    )

    thresholds = np.arange(
        THRESHOLD_START,
        THRESHOLD_END + (
            THRESHOLD_STEP / 2
        ),
        THRESHOLD_STEP,
    )

    threshold_records = []

    for threshold in thresholds:

        record = evaluate_threshold(
            y_true,
            probabilities,
            float(threshold),
        )

        threshold_records.append(
            record
        )

    results = pd.DataFrame(
        threshold_records
    )

    print(
        f"Thresholds evaluated:        "
        f"{len(results)}"
    )

    # ------------------------------------------------------------------------
    # BEST THRESHOLDS
    # ------------------------------------------------------------------------

    print_section(
        "BEST THRESHOLDS"
    )

    best_f1 = results.loc[
        results["f1"].idxmax()
    ]

    best_precision = results.loc[
        results["precision"].idxmax()
    ]

    best_recall = results.loc[
        results["recall"].idxmax()
    ]

    # Highest recall while maintaining at least
    # 5% precision.
    precision_5 = results[
        results["precision"] >= 0.05
    ]

    if len(precision_5) > 0:
        best_recall_at_5_precision = (
            precision_5.loc[
                precision_5["recall"].idxmax()
            ]
        )
    else:
        best_recall_at_5_precision = None

    # Highest recall while maintaining at least
    # 10% precision.
    precision_10 = results[
        results["precision"] >= 0.10
    ]

    if len(precision_10) > 0:
        best_recall_at_10_precision = (
            precision_10.loc[
                precision_10["recall"].idxmax()
            ]
        )
    else:
        best_recall_at_10_precision = None

    def print_threshold_result(
        label,
        row,
    ):

        if row is None:
            print(
                f"{label}:                  "
                "No threshold satisfies constraint."
            )
            return

        print(
            f"{label}:                  "
            f"{row['threshold']:.2f}"
        )

        print(
            f"  Precision:                 "
            f"{row['precision'] * 100:.2f}%"
        )

        print(
            f"  Recall:                    "
            f"{row['recall'] * 100:.2f}%"
        )

        print(
            f"  F1:                        "
            f"{row['f1']:.4f}"
        )

        print(
            f"  Alert rate:                "
            f"{row['alert_rate'] * 100:.2f}%"
        )

        print(
            f"  False-positive rate:       "
            f"{row['false_positive_rate'] * 100:.2f}%"
        )

        print(
            f"  Lift vs base rate:         "
            f"{row['lift_vs_base_rate']:.2f}x"
        )

        print(
            f"  TP / FP / FN / TN:         "
            f"{int(row['true_positives'])} / "
            f"{int(row['false_positives'])} / "
            f"{int(row['false_negatives'])} / "
            f"{int(row['true_negatives'])}"
        )

    print_threshold_result(
        "Best F1 threshold",
        best_f1,
    )

    print()

    print_threshold_result(
        "Best precision threshold",
        best_precision,
    )

    print()

    print_threshold_result(
        "Best recall threshold",
        best_recall,
    )

    print()

    print_threshold_result(
        "Best recall with >=5% precision",
        best_recall_at_5_precision,
    )

    print()

    print_threshold_result(
        "Best recall with >=10% precision",
        best_recall_at_10_precision,
    )

    # ------------------------------------------------------------------------
    # PRACTICAL THRESHOLD TABLE
    # ------------------------------------------------------------------------

    print_section(
        "PRACTICAL THRESHOLD COMPARISON"
    )

    practical_thresholds = [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.40,
        0.50,
    ]

    practical = results[
        results["threshold"].isin(
            practical_thresholds
        )
    ].copy()

    display_columns = [
        "threshold",
        "precision",
        "recall",
        "f1",
        "alert_rate",
        "false_positive_rate",
        "lift_vs_base_rate",
        "true_positives",
        "false_positives",
        "false_negatives",
    ]

    print(
        practical[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ------------------------------------------------------------------------
    # SAVE COMPLETE THRESHOLD RESULTS
    # ------------------------------------------------------------------------

    print_section(
        "SAVING RESULTS"
    )

    OUTPUT_THRESHOLDS.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_THRESHOLDS,
        index=False,
    )

    print(
        f"Threshold results:           "
        f"{OUTPUT_THRESHOLDS.resolve()}"
    )

    # ------------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------------

    def row_to_dict(row):
        if row is None:
            return None

        return {
            key: (
                None
                if pd.isna(value)
                else (
                    float(value)
                    if isinstance(
                        value,
                        (np.floating, float)
                    )
                    else int(value)
                    if isinstance(
                        value,
                        (np.integer, int)
                    )
                    else value
                )
            )
            for key, value in row.items()
        }

    summary = {
        "dataset": {
            "prediction_file": str(
                PREDICTIONS_PATH
            ),
            "rows": int(len(df)),
            "positive_rows": positive_count,
            "negative_rows": negative_count,
            "base_positive_rate": float(
                base_rate
            ),
        },
        "prediction_column": PROBABILITY_COLUMN,
        "threshold_grid": {
            "start": THRESHOLD_START,
            "end": THRESHOLD_END,
            "step": THRESHOLD_STEP,
        },
        "best_thresholds": {
            "best_f1": row_to_dict(
                best_f1
            ),
            "best_precision": row_to_dict(
                best_precision
            ),
            "best_recall": row_to_dict(
                best_recall
            ),
            "best_recall_at_5_percent_precision":
                row_to_dict(
                    best_recall_at_5_precision
                ),
            "best_recall_at_10_percent_precision":
                row_to_dict(
                    best_recall_at_10_precision
                ),
        },
        "methodology_notes": [
            "Threshold analysis uses participant-level out-of-fold predictions.",
            "No model retraining occurs in this script.",
            "Threshold-dependent metrics are descriptive operating-point analysis.",
            "The underlying positive rate is approximately 1.86%.",
            "Lift compares precision at a threshold against the overall positive rate.",
            "This analysis does not establish a clinical alert threshold.",
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
        )

    print(
        f"Threshold summary:           "
        f"{OUTPUT_SUMMARY.resolve()}"
    )

    print("\n" + "=" * 76)
    print(
        "PREDICTION THRESHOLD ANALYSIS COMPLETE"
    )
    print("=" * 76)


if __name__ == "__main__":
    main()
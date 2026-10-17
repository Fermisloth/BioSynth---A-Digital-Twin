from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================================
# CONFIGURATION
# ============================================================================

FEATURE_PATH = Path(
    "reports/hall2018_cgm_trajectory_features.csv"
)

OUTPUT_PREDICTIONS = Path(
    "reports/hall2018_baseline_oof_predictions.csv"
)

OUTPUT_RESULTS = Path(
    "reports/hall2018_cross_validation_results.csv"
)

OUTPUT_SUMMARY = Path(
    "reports/hall2018_cross_validation_summary.json"
)

TARGET = "pre_event_target"
PERSON_ID = "person_id"

N_SPLITS = 5
RANDOM_STATE = 42


# ============================================================================
# FEATURE DEFINITIONS
# ============================================================================

BASELINE_FEATURES = [
    "current_glucose_mg_dl",
]

TRAJECTORY_FEATURES = [
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


def build_model(feature_columns):
    """
    Leakage-safe logistic regression pipeline.

    Imputation and scaling are fitted separately inside every CV training
    fold. Test-fold information therefore does not enter preprocessing.
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def calculate_metrics(
    y_true,
    probabilities,
    threshold=0.5,
):
    """
    Calculate threshold-independent and threshold-dependent metrics.
    """

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "pr_auc": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    if (tn + fp) > 0:
        metrics["specificity"] = float(
            tn / (tn + fp)
        )

        metrics["false_positive_rate"] = float(
            fp / (tn + fp)
        )
    else:
        metrics["specificity"] = np.nan
        metrics["false_positive_rate"] = np.nan

    return metrics


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 76)
    print("Hall 2018 — 5-Fold Participant-Level Cross-Validation")
    print("=" * 76)

    # ------------------------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------------------------

    print_section(
        "LOADING TRAJECTORY FEATURE DATASET"
    )

    if not FEATURE_PATH.exists():
        fail(
            "Feature dataset not found:\n"
            f"{FEATURE_PATH.resolve()}"
        )

    df = pd.read_csv(
        FEATURE_PATH
    )

    print(
        f"Rows loaded:                 "
        f"{len(df):,}"
    )

    print(
        f"Participants:                "
        f"{df[PERSON_ID].nunique():,}"
    )

    # ------------------------------------------------------------------------
    # REQUIRED COLUMNS
    # ------------------------------------------------------------------------

    required_columns = (
        [PERSON_ID, TARGET]
        + BASELINE_FEATURES
        + TRAJECTORY_FEATURES
    )

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
    # TARGET
    # ------------------------------------------------------------------------

    print_section(
        "TARGET"
    )

    if df[TARGET].isna().any():
        fail(
            "Target contains missing values."
        )

    target_values = set(
        df[TARGET].unique()
    )

    if not target_values.issubset({0, 1}):
        fail(
            "Target contains unexpected values: "
            f"{sorted(target_values)}"
        )

    positive_count = int(
        df[TARGET].sum()
    )

    negative_count = int(
        (df[TARGET] == 0).sum()
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
        f"Positive rate:               "
        f"{df[TARGET].mean() * 100:.2f}%"
    )

    # ------------------------------------------------------------------------
    # PARTICIPANT INFORMATION
    # ------------------------------------------------------------------------

    participants = np.array(
        sorted(
            df[PERSON_ID].unique()
        )
    )

    if len(participants) < N_SPLITS:
        fail(
            f"Need at least {N_SPLITS} participants "
            f"for {N_SPLITS}-fold CV. "
            f"Found {len(participants)}."
        )

    print(
        f"\nParticipants available:      "
        f"{len(participants)}"
    )

    print(
        f"CV folds:                    "
        f"{N_SPLITS}"
    )

    print(
        "\nSplitting by participant — "
        "no participant will appear in both "
        "training and validation data within a fold."
    )

    # ------------------------------------------------------------------------
    # GROUP K-FOLD
    # ------------------------------------------------------------------------

    groups = df[PERSON_ID]

    cv = GroupKFold(
        n_splits=N_SPLITS
    )

    # Store OOF probabilities for each model.
    baseline_oof = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    trajectory_oof = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    fold_records = []

    # ------------------------------------------------------------------------
    # CROSS-VALIDATION
    # ------------------------------------------------------------------------

    print_section(
        "CROSS-VALIDATION"
    )

    for fold_number, (
        train_indices,
        validation_indices,
    ) in enumerate(
        cv.split(
            df,
            df[TARGET],
            groups=groups,
        ),
        start=1,
    ):

        train_df = df.iloc[
            train_indices
        ]

        validation_df = df.iloc[
            validation_indices
        ]

        train_participants = set(
            train_df[PERSON_ID]
        )

        validation_participants = set(
            validation_df[PERSON_ID]
        )

        overlap = (
            train_participants
            & validation_participants
        )

        if overlap:
            fail(
                f"Participant leakage in fold {fold_number}: "
                f"{sorted(overlap)}"
            )

        print(
            f"\nFold {fold_number}/{N_SPLITS}"
        )

        print(
            f"  Training participants:     "
            f"{len(train_participants)}"
        )

        print(
            f"  Validation participants:   "
            f"{len(validation_participants)}"
        )

        print(
            f"  Training rows:              "
            f"{len(train_df):,}"
        )

        print(
            f"  Validation rows:            "
            f"{len(validation_df):,}"
        )

        print(
            f"  Validation positives:       "
            f"{int(validation_df[TARGET].sum()):,}"
        )

        # --------------------------------------------------------------------
        # CURRENT GLUCOSE BASELINE
        # --------------------------------------------------------------------

        baseline_model = build_model(
            BASELINE_FEATURES
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            baseline_model.fit(
                train_df[
                    BASELINE_FEATURES
                ],
                train_df[TARGET],
            )

        baseline_probabilities = (
            baseline_model.predict_proba(
                validation_df[
                    BASELINE_FEATURES
                ]
            )[:, 1]
        )

        baseline_oof[
            validation_indices
        ] = baseline_probabilities

        # --------------------------------------------------------------------
        # TRAJECTORY MODEL
        # --------------------------------------------------------------------

        trajectory_model = build_model(
            TRAJECTORY_FEATURES
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            trajectory_model.fit(
                train_df[
                    TRAJECTORY_FEATURES
                ],
                train_df[TARGET],
            )

        trajectory_probabilities = (
            trajectory_model.predict_proba(
                validation_df[
                    TRAJECTORY_FEATURES
                ]
            )[:, 1]
        )

        trajectory_oof[
            validation_indices
        ] = trajectory_probabilities

        # --------------------------------------------------------------------
        # FOLD METRICS
        # --------------------------------------------------------------------

        baseline_metrics = calculate_metrics(
            validation_df[TARGET].values,
            baseline_probabilities,
        )

        trajectory_metrics = calculate_metrics(
            validation_df[TARGET].values,
            trajectory_probabilities,
        )

        fold_records.append(
            {
                "fold": fold_number,
                "model": "current_glucose_baseline",
                "participants": len(
                    validation_participants
                ),
                "rows": len(
                    validation_df
                ),
                "positive_rows": int(
                    validation_df[TARGET].sum()
                ),
                **baseline_metrics,
            }
        )

        fold_records.append(
            {
                "fold": fold_number,
                "model": "cgm_trajectory_logistic",
                "participants": len(
                    validation_participants
                ),
                "rows": len(
                    validation_df
                ),
                "positive_rows": int(
                    validation_df[TARGET].sum()
                ),
                **trajectory_metrics,
            }
        )

        print(
            f"  Baseline PR-AUC:            "
            f"{baseline_metrics['pr_auc']:.4f}"
        )

        print(
            f"  Trajectory PR-AUC:          "
            f"{trajectory_metrics['pr_auc']:.4f}"
        )

    # ------------------------------------------------------------------------
    # OOF VALIDATION
    # ------------------------------------------------------------------------

    print_section(
        "OUT-OF-FOLD PREDICTION CHECK"
    )

    if np.isnan(
        baseline_oof
    ).any():
        fail(
            "Some rows do not have baseline "
            "out-of-fold predictions."
        )

    if np.isnan(
        trajectory_oof
    ).any():
        fail(
            "Some rows do not have trajectory "
            "out-of-fold predictions."
        )

    print(
        "Baseline OOF predictions:    "
        f"{len(baseline_oof):,}/{len(df):,}"
    )

    print(
        "Trajectory OOF predictions:  "
        f"{len(trajectory_oof):,}/{len(df):,}"
    )

    # ------------------------------------------------------------------------
    # OVERALL OOF METRICS
    # ------------------------------------------------------------------------

    print_section(
        "OVERALL OUT-OF-FOLD PERFORMANCE"
    )

    y = df[TARGET].astype(int).values

    baseline_overall = calculate_metrics(
        y,
        baseline_oof,
    )

    trajectory_overall = calculate_metrics(
        y,
        trajectory_oof,
    )

    overall_records = [
        {
            "fold": "overall_oof",
            "model": "current_glucose_baseline",
            "participants": int(
                df[PERSON_ID].nunique()
            ),
            "rows": len(df),
            "positive_rows": positive_count,
            **baseline_overall,
        },
        {
            "fold": "overall_oof",
            "model": "cgm_trajectory_logistic",
            "participants": int(
                df[PERSON_ID].nunique()
            ),
            "rows": len(df),
            "positive_rows": positive_count,
            **trajectory_overall,
        },
    ]

    results = pd.DataFrame(
        fold_records + overall_records
    )

    display_columns = [
        "fold",
        "model",
        "pr_auc",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "specificity",
        "false_positive_rate",
    ]

    print(
        results[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ------------------------------------------------------------------------
    # MODEL COMPARISON
    # ------------------------------------------------------------------------

    print_section(
        "OVERALL TRAJECTORY VS BASELINE"
    )

    baseline_pr = (
        baseline_overall["pr_auc"]
    )

    trajectory_pr = (
        trajectory_overall["pr_auc"]
    )

    baseline_roc = (
        baseline_overall["roc_auc"]
    )

    trajectory_roc = (
        trajectory_overall["roc_auc"]
    )

    pr_delta = (
        trajectory_pr
        - baseline_pr
    )

    roc_delta = (
        trajectory_roc
        - baseline_roc
    )

    print(
        f"Baseline PR-AUC:             "
        f"{baseline_pr:.4f}"
    )

    print(
        f"Trajectory PR-AUC:           "
        f"{trajectory_pr:.4f}"
    )

    print(
        f"PR-AUC change:               "
        f"{pr_delta:+.4f}"
    )

    print()

    print(
        f"Baseline ROC-AUC:            "
        f"{baseline_roc:.4f}"
    )

    print(
        f"Trajectory ROC-AUC:          "
        f"{trajectory_roc:.4f}"
    )

    print(
        f"ROC-AUC change:              "
        f"{roc_delta:+.4f}"
    )

    if baseline_pr > 0:
        relative_pr_change = (
            pr_delta
            / baseline_pr
        )
    else:
        relative_pr_change = np.nan

    print(
        f"Relative PR-AUC change:      "
        f"{relative_pr_change * 100:+.1f}%"
    )

    # ------------------------------------------------------------------------
    # SAVE OOF PREDICTIONS
    # ------------------------------------------------------------------------

    print_section(
        "SAVING OUT-OF-FOLD PREDICTIONS"
    )

    oof_predictions = df[
        [
            PERSON_ID,
            TARGET,
        ]
    ].copy()

    # Keep useful identifiers if present.
    optional_columns = [
        "prediction_timestamp",
        "diabetes_type",
        "segment_id",
    ]

    for column in optional_columns:
        if column in df.columns:
            oof_predictions[
                column
            ] = df[column]

    oof_predictions[
        "baseline_probability"
    ] = baseline_oof

    oof_predictions[
        "trajectory_probability"
    ] = trajectory_oof

    oof_predictions.to_csv(
        OUTPUT_PREDICTIONS,
        index=False,
    )

    print(
        f"OOF predictions:             "
        f"{OUTPUT_PREDICTIONS.resolve()}"
    )

    # ------------------------------------------------------------------------
    # SAVE RESULTS
    # ------------------------------------------------------------------------

    results.to_csv(
        OUTPUT_RESULTS,
        index=False,
    )

    print(
        f"CV results:                  "
        f"{OUTPUT_RESULTS.resolve()}"
    )

    # ------------------------------------------------------------------------
    # SAVE SUMMARY
    # ------------------------------------------------------------------------

    fold_summary = []

    for fold_number in range(
        1,
        N_SPLITS + 1,
    ):

        fold_df = results[
            results["fold"] == fold_number
        ]

        baseline_fold = fold_df[
            fold_df["model"]
            == "current_glucose_baseline"
        ].iloc[0]

        trajectory_fold = fold_df[
            fold_df["model"]
            == "cgm_trajectory_logistic"
        ].iloc[0]

        fold_summary.append(
            {
                "fold": fold_number,
                "baseline_pr_auc": float(
                    baseline_fold["pr_auc"]
                ),
                "trajectory_pr_auc": float(
                    trajectory_fold["pr_auc"]
                ),
                "pr_auc_delta": float(
                    trajectory_fold["pr_auc"]
                    - baseline_fold["pr_auc"]
                ),
                "baseline_roc_auc": float(
                    baseline_fold["roc_auc"]
                ),
                "trajectory_roc_auc": float(
                    trajectory_fold["roc_auc"]
                ),
                "roc_auc_delta": float(
                    trajectory_fold["roc_auc"]
                    - baseline_fold["roc_auc"]
                ),
            }
        )

    summary = {
        "dataset": {
            "feature_path": str(
                FEATURE_PATH
            ),
            "rows": int(len(df)),
            "participants": int(
                df[PERSON_ID].nunique()
            ),
            "positive_rows": positive_count,
            "negative_rows": negative_count,
            "positive_rate": float(
                df[TARGET].mean()
            ),
        },
        "cross_validation": {
            "method": "GroupKFold",
            "group_column": PERSON_ID,
            "n_splits": N_SPLITS,
            "random_state": RANDOM_STATE,
            "participant_leakage": False,
            "all_rows_have_oof_predictions": True,
        },
        "models": {
            "current_glucose_baseline": {
                "features": BASELINE_FEATURES,
                "overall_oof_metrics": baseline_overall,
            },
            "cgm_trajectory_logistic": {
                "features": TRAJECTORY_FEATURES,
                "overall_oof_metrics": trajectory_overall,
            },
        },
        "comparison": {
            "pr_auc_delta": float(
                pr_delta
            ),
            "roc_auc_delta": float(
                roc_delta
            ),
            "relative_pr_auc_change": float(
                relative_pr_change
            ),
        },
        "folds": fold_summary,
        "methodology_notes": [
            "Primary target is pre_event_target.",
            "Participants are the grouping unit for cross-validation.",
            "Every row receives an out-of-fold prediction.",
            "No participant is present in both training and validation within a fold.",
            "Preprocessing is fitted independently inside each training fold.",
            "Class weighting is used because the positive target is rare.",
            "PR-AUC is emphasized because the target is highly imbalanced.",
            "The 0.5 threshold is retained only for descriptive threshold-dependent metrics.",
            "Threshold optimization is deferred until cross-validation establishes generalization.",
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
        f"CV summary:                 "
        f"{OUTPUT_SUMMARY.resolve()}"
    )

    # ------------------------------------------------------------------------
    # FINAL STATUS
    # ------------------------------------------------------------------------

    print("\n" + "=" * 76)
    print(
        "5-FOLD PARTICIPANT-LEVEL "
        "CROSS-VALIDATION COMPLETE"
    )
    print("=" * 76)


if __name__ == "__main__":
    main()
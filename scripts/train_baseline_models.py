from pathlib import Path

import json
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================================
# CONFIGURATION
# ============================================================================

FEATURE_PATH = Path("reports/hall2018_cgm_trajectory_features.csv")

OUTPUT_RESULTS = Path("reports/hall2018_baseline_model_results.csv")
OUTPUT_SUMMARY = Path("reports/hall2018_baseline_model_summary.json")

TARGET = "pre_event_target"
PERSON_ID = "person_id"

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Minimum number of participants required in both groups.
# We have only 57 participants total, so keep the split simple and reproducible.
MIN_TEST_PARTICIPANTS = 1


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
    raise RuntimeError(f"\nERROR: {message}")


def print_section(title):
    print("\n" + "-" * 76)
    print(title)
    print("-" * 76)


def evaluate_predictions(y_true, probabilities, threshold=0.5):
    """
    Calculate classification metrics.

    The default 0.5 threshold is intentionally retained for the first
    baseline comparison. Threshold optimization comes later.
    """

    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
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

    if tn + fp > 0:
        metrics["specificity"] = float(tn / (tn + fp))
        metrics["false_positive_rate"] = float(fp / (tn + fp))
    else:
        metrics["specificity"] = np.nan
        metrics["false_positive_rate"] = np.nan

    return metrics


def participant_level_split(df):
    """
    Split by participant so that no participant occurs in both train and test.

    This is essential because many prediction rows belong to the same person
    and overlapping windows from the same hyperglycemic event are correlated.
    """

    participants = np.array(sorted(df[PERSON_ID].unique()))

    if len(participants) < 2:
        fail("Need at least two participants for a train/test split.")

    rng = np.random.default_rng(RANDOM_STATE)
    shuffled = participants.copy()
    rng.shuffle(shuffled)

    n_test = max(
        MIN_TEST_PARTICIPANTS,
        int(round(len(shuffled) * TEST_SIZE)),
    )

    n_test = min(n_test, len(shuffled) - 1)

    test_participants = set(shuffled[:n_test])
    train_participants = set(shuffled[n_test:])

    train_df = df[df[PERSON_ID].isin(train_participants)].copy()
    test_df = df[df[PERSON_ID].isin(test_participants)].copy()

    return (
        train_df,
        test_df,
        sorted(train_participants),
        sorted(test_participants),
    )


def build_model(feature_columns):
    """
    Leakage-safe preprocessing + logistic regression.

    Median imputation is included only as a defensive measure. It is fitted
    on training data through the Pipeline and therefore cannot use test data.

    Standardization is appropriate for logistic regression because the
    trajectory features are on very different numerical scales.
    """

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_columns,
            )
        ],
        remainder="drop",
    )

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("model", model),
        ]
    )


def run_model(
    model_name,
    feature_columns,
    train_df,
    test_df,
):
    print(f"\nTraining: {model_name}")
    print(f"Features: {len(feature_columns)}")

    X_train = train_df[feature_columns]
    y_train = train_df[TARGET].astype(int)

    X_test = test_df[feature_columns]
    y_test = test_df[TARGET].astype(int)

    model = build_model(feature_columns)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = evaluate_predictions(
        y_test,
        probabilities,
    )

    metrics["model"] = model_name
    metrics["feature_count"] = len(feature_columns)
    metrics["train_rows"] = len(train_df)
    metrics["test_rows"] = len(test_df)
    metrics["train_positive_rows"] = int(y_train.sum())
    metrics["test_positive_rows"] = int(y_test.sum())

    return model, metrics


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 76)
    print("Hall 2018 — Baseline Predictive Models")
    print("=" * 76)

    # ------------------------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------------------------

    print_section("LOADING FEATURE DATASET")

    if not FEATURE_PATH.exists():
        fail(
            f"Feature dataset not found:\n"
            f"{FEATURE_PATH.resolve()}"
        )

    df = pd.read_csv(FEATURE_PATH)

    print(f"Rows loaded:                 {len(df):,}")
    print(f"Participants:                {df[PERSON_ID].nunique():,}")

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
            + "\n".join(f"  - {column}" for column in missing_columns)
        )

    # ------------------------------------------------------------------------
    # TARGET VALIDATION
    # ------------------------------------------------------------------------

    print_section("TARGET")

    target_values = set(df[TARGET].dropna().unique())

    if not target_values.issubset({0, 1}):
        fail(
            f"Target contains unexpected values: {sorted(target_values)}"
        )

    if df[TARGET].isna().any():
        fail("Target contains missing values.")

    positive_count = int(df[TARGET].sum())
    negative_count = int((df[TARGET] == 0).sum())

    print(f"Positive rows:               {positive_count:,}")
    print(f"Negative rows:               {negative_count:,}")
    print(
        f"Positive rate:               "
        f"{positive_count / len(df) * 100:.2f}%"
    )

    # ------------------------------------------------------------------------
    # PARTICIPANT-LEVEL SPLIT
    # ------------------------------------------------------------------------

    print_section("PARTICIPANT-LEVEL TRAIN / TEST SPLIT")

    (
        train_df,
        test_df,
        train_participants,
        test_participants,
    ) = participant_level_split(df)

    overlap = set(train_participants) & set(test_participants)

    if overlap:
        fail(
            "Participant leakage detected. "
            f"Participants appear in both sets: {sorted(overlap)}"
        )

    print(f"Total participants:          {df[PERSON_ID].nunique():,}")
    print(f"Training participants:       {len(train_participants):,}")
    print(f"Test participants:           {len(test_participants):,}")

    print(f"\nTraining rows:               {len(train_df):,}")
    print(f"Test rows:                   {len(test_df):,}")

    print(
        f"Training positive rate:      "
        f"{train_df[TARGET].mean() * 100:.2f}%"
    )

    print(
        f"Test positive rate:          "
        f"{test_df[TARGET].mean() * 100:.2f}%"
    )

    print("\nTest participants:")
    for participant in test_participants:
        participant_rows = test_df[
            test_df[PERSON_ID] == participant
        ]

        participant_positive = int(
            participant_rows[TARGET].sum()
        )

        print(
            f"  {participant}: "
            f"{len(participant_rows):,} rows, "
            f"{participant_positive:,} positives"
        )

    # ------------------------------------------------------------------------
    # TRAIN MODELS
    # ------------------------------------------------------------------------

    print_section("MODEL TRAINING")

    baseline_model, baseline_metrics = run_model(
        "current_glucose_baseline",
        BASELINE_FEATURES,
        train_df,
        test_df,
    )

    trajectory_model, trajectory_metrics = run_model(
        "cgm_trajectory_logistic",
        TRAJECTORY_FEATURES,
        train_df,
        test_df,
    )

    results = pd.DataFrame(
        [
            baseline_metrics,
            trajectory_metrics,
        ]
    )

    # ------------------------------------------------------------------------
    # RESULTS
    # ------------------------------------------------------------------------

    print_section("MODEL COMPARISON")

    display_columns = [
        "model",
        "feature_count",
        "pr_auc",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "specificity",
        "false_positive_rate",
    ]

    print(
        results[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ------------------------------------------------------------------------
    # INTERPRETATION
    # ------------------------------------------------------------------------

    baseline_pr = baseline_metrics["pr_auc"]
    trajectory_pr = trajectory_metrics["pr_auc"]

    baseline_roc = baseline_metrics["roc_auc"]
    trajectory_roc = trajectory_metrics["roc_auc"]

    pr_delta = trajectory_pr - baseline_pr
    roc_delta = trajectory_roc - baseline_roc

    print_section("TRAJECTORY VS CURRENT-GLUCOSE BASELINE")

    print(
        f"PR-AUC change:               "
        f"{pr_delta:+.4f}"
    )

    print(
        f"ROC-AUC change:              "
        f"{roc_delta:+.4f}"
    )

    if pr_delta > 0:
        print(
            "\nTrajectory features improve PR-AUC over "
            "the current-glucose baseline."
        )
    elif pr_delta < 0:
        print(
            "\nTrajectory features do not improve PR-AUC "
            "over the current-glucose baseline."
        )
    else:
        print(
            "\nTrajectory features produce the same PR-AUC "
            "as the current-glucose baseline."
        )

    # ------------------------------------------------------------------------
    # SAVE RESULTS
    # ------------------------------------------------------------------------

    print_section("SAVING RESULTS")

    OUTPUT_RESULTS.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_RESULTS,
        index=False,
    )

    summary = {
        "dataset": {
            "feature_path": str(FEATURE_PATH),
            "rows": int(len(df)),
            "participants": int(df[PERSON_ID].nunique()),
            "positive_rows": positive_count,
            "negative_rows": negative_count,
            "positive_rate": float(df[TARGET].mean()),
        },
        "split": {
            "method": "participant_level",
            "random_state": RANDOM_STATE,
            "test_size_fraction": TEST_SIZE,
            "train_participants": train_participants,
            "test_participants": test_participants,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "participant_overlap": sorted(overlap),
        },
        "models": {
            "current_glucose_baseline": {
                "features": BASELINE_FEATURES,
                "metrics": baseline_metrics,
            },
            "cgm_trajectory_logistic": {
                "features": TRAJECTORY_FEATURES,
                "metrics": trajectory_metrics,
            },
        },
        "comparison": {
            "pr_auc_delta_trajectory_minus_baseline": float(pr_delta),
            "roc_auc_delta_trajectory_minus_baseline": float(roc_delta),
        },
        "methodology_notes": [
            "Primary target is pre_event_target.",
            "Participant-level splitting prevents participant leakage.",
            "Logistic regression uses class_weight='balanced'.",
            "No synthetic oversampling is used.",
            "Threshold-dependent metrics use probability threshold 0.5.",
            "PR-AUC is emphasized because the positive class is rare.",
            "Threshold optimization is intentionally deferred until after the baseline comparison.",
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
        f"Results CSV:                "
        f"{OUTPUT_RESULTS.resolve()}"
    )

    print(
        f"Summary JSON:               "
        f"{OUTPUT_SUMMARY.resolve()}"
    )

    print("\n" + "=" * 76)
    print("BASELINE MODELING COMPLETE")
    print("=" * 76)


if __name__ == "__main__":
    main()
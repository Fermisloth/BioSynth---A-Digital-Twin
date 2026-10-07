from __future__ import annotations

import json
import warnings
from pathlib import Path

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


FEATURE_PATH = Path("reports/synthetic_cgm_trajectory_features.csv")
OUTPUT_PREDICTIONS = Path("reports/synthetic_model_oof_predictions.csv")
OUTPUT_RESULTS = Path("reports/synthetic_cross_validation_results.csv")
OUTPUT_SUMMARY = Path("reports/synthetic_cross_validation_summary.json")

TARGET = "pre_event_target"
PERSON_ID = "person_id"
PHENOTYPE = "simulation_phenotype"
TIMESTAMP = "prediction_timestamp"
N_SPLITS = 5
RANDOM_STATE = 42

BASELINE_FEATURES = ["current_glucose_mg_dl"]

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

STRICT_EXCLUSIONS = {
    "simulation_phenotype",
    "person_id",
    "prediction_timestamp",
    "baseline_glucose_mg_dl",
    "circadian_amplitude",
    "noise_scale",
    "meal_response_scale",
    "recovery_parameter",
    "random_seed",
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
    "event_timestamp",
    "event_type",
    "event_magnitude",
    "event_duration_parameter",
}


def fail(message: str) -> None:
    raise RuntimeError(f"\nERROR: {message}")


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
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


def safe_pr_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, probabilities))


def safe_roc_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, probabilities))


def calculate_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int | None]:
    predictions = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) else None
    false_positive_rate = float(fp / (tn + fp)) if (tn + fp) else None

    return {
        "pr_auc": safe_pr_auc(y_true, probabilities),
        "roc_auc": safe_roc_auc(y_true, probabilities),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "specificity": specificity,
        "false_positive_rate": false_positive_rate,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def load_dataset() -> pd.DataFrame:
    if not FEATURE_PATH.exists():
        fail(f"Synthetic trajectory feature dataset not found: {FEATURE_PATH}")

    df = pd.read_csv(FEATURE_PATH)
    required = {PERSON_ID, TIMESTAMP, PHENOTYPE, TARGET, *BASELINE_FEATURES, *TRAJECTORY_FEATURES}
    missing = sorted(required - set(df.columns))
    if missing:
        fail("Required columns are missing:\n" + "\n".join(f"  - {column}" for column in missing))

    forbidden_baseline = sorted(set(BASELINE_FEATURES) & STRICT_EXCLUSIONS)
    forbidden_trajectory = sorted(set(TRAJECTORY_FEATURES) & STRICT_EXCLUSIONS)
    if forbidden_baseline or forbidden_trajectory:
        fail(
            "Forbidden columns entered model features: "
            f"{forbidden_baseline + forbidden_trajectory}"
        )

    df[PERSON_ID] = df[PERSON_ID].astype(str)
    df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP], errors="coerce")
    if df[TIMESTAMP].isna().any():
        fail("Invalid prediction timestamps found.")

    df[TARGET] = df[TARGET].astype(bool).astype(int)
    return df.reset_index(drop=True)


def subgroup_metrics(oof: pd.DataFrame) -> dict[str, dict[str, float | int | None]]:
    results = {}
    for phenotype, group in oof.groupby(PHENOTYPE):
        y = group[TARGET].astype(int).to_numpy()
        results[str(phenotype)] = {
            "participants": int(group[PERSON_ID].nunique()),
            "rows": int(len(group)),
            "positives": int(y.sum()),
            "positive_rate": float(y.mean()),
            "baseline_pr_auc": safe_pr_auc(y, group["baseline_probability"].to_numpy()),
            "trajectory_pr_auc": safe_pr_auc(y, group["trajectory_probability"].to_numpy()),
            "baseline_roc_auc": safe_roc_auc(y, group["baseline_probability"].to_numpy()),
            "trajectory_roc_auc": safe_roc_auc(y, group["trajectory_probability"].to_numpy()),
        }
    return results


def run_cross_validation(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    groups = df[PERSON_ID]
    cv = GroupKFold(n_splits=N_SPLITS)
    baseline_oof = np.full(len(df), np.nan)
    trajectory_oof = np.full(len(df), np.nan)
    fold_assignments = np.full(len(df), -1, dtype=int)
    fold_records = []
    leakage_pass = True

    for fold, (train_idx, val_idx) in enumerate(cv.split(df, df[TARGET], groups=groups), start=1):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        train_people = set(train_df[PERSON_ID])
        val_people = set(val_df[PERSON_ID])
        overlap = train_people & val_people
        if overlap:
            leakage_pass = False
            fail(f"Participant leakage in fold {fold}: {sorted(overlap)}")

        baseline_model = build_model()
        trajectory_model = build_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            baseline_model.fit(train_df[BASELINE_FEATURES], train_df[TARGET])
            trajectory_model.fit(train_df[TRAJECTORY_FEATURES], train_df[TARGET])

        baseline_prob = baseline_model.predict_proba(val_df[BASELINE_FEATURES])[:, 1]
        trajectory_prob = trajectory_model.predict_proba(val_df[TRAJECTORY_FEATURES])[:, 1]
        baseline_oof[val_idx] = baseline_prob
        trajectory_oof[val_idx] = trajectory_prob
        fold_assignments[val_idx] = fold

        y_val = val_df[TARGET].to_numpy()
        base_metrics = calculate_metrics(y_val, baseline_prob)
        traj_metrics = calculate_metrics(y_val, trajectory_prob)
        common = {
            "fold": fold,
            "validation_participants": int(len(val_people)),
            "validation_rows": int(len(val_df)),
            "validation_positives": int(y_val.sum()),
            "validation_positive_rate": float(y_val.mean()),
            "train_participants": int(len(train_people)),
        }
        fold_records.append({**common, "model": "current_glucose_baseline", **base_metrics})
        fold_records.append({**common, "model": "cgm_trajectory_logistic", **traj_metrics})

    if np.isnan(baseline_oof).any() or np.isnan(trajectory_oof).any() or (fold_assignments < 0).any():
        fail("Not every row received out-of-fold predictions.")

    oof = df[[PERSON_ID, TIMESTAMP, PHENOTYPE, TARGET]].copy()
    oof["baseline_probability"] = baseline_oof
    oof["trajectory_probability"] = trajectory_oof
    oof["fold"] = fold_assignments

    y = df[TARGET].to_numpy()
    baseline_overall = calculate_metrics(y, baseline_oof)
    trajectory_overall = calculate_metrics(y, trajectory_oof)

    results = pd.DataFrame(
        fold_records
        + [
            {
                "fold": "overall_oof",
                "model": "current_glucose_baseline",
                "validation_participants": int(df[PERSON_ID].nunique()),
                "validation_rows": int(len(df)),
                "validation_positives": int(y.sum()),
                "validation_positive_rate": float(y.mean()),
                "train_participants": None,
                **baseline_overall,
            },
            {
                "fold": "overall_oof",
                "model": "cgm_trajectory_logistic",
                "validation_participants": int(df[PERSON_ID].nunique()),
                "validation_rows": int(len(df)),
                "validation_positives": int(y.sum()),
                "validation_positive_rate": float(y.mean()),
                "train_participants": None,
                **trajectory_overall,
            },
        ]
    )

    fold_summaries = []
    pr_improved = 0
    roc_improved = 0
    for fold in range(1, N_SPLITS + 1):
        fold_df = results[results["fold"] == fold]
        base = fold_df[fold_df["model"] == "current_glucose_baseline"].iloc[0]
        traj = fold_df[fold_df["model"] == "cgm_trajectory_logistic"].iloc[0]
        pr_delta = float(traj["pr_auc"] - base["pr_auc"])
        roc_delta = float(traj["roc_auc"] - base["roc_auc"])
        pr_improved += int(pr_delta > 0)
        roc_improved += int(roc_delta > 0)
        fold_summaries.append(
            {
                "fold": fold,
                "train_participants": int(base["train_participants"]),
                "validation_participants": int(base["validation_participants"]),
                "validation_rows": int(base["validation_rows"]),
                "validation_positives": int(base["validation_positives"]),
                "validation_positive_rate": float(base["validation_positive_rate"]),
                "baseline_pr_auc": float(base["pr_auc"]),
                "trajectory_pr_auc": float(traj["pr_auc"]),
                "baseline_roc_auc": float(base["roc_auc"]),
                "trajectory_roc_auc": float(traj["roc_auc"]),
                "pr_auc_delta": pr_delta,
                "roc_auc_delta": roc_delta,
            }
        )

    pr_delta = float(trajectory_overall["pr_auc"] - baseline_overall["pr_auc"])
    roc_delta = float(trajectory_overall["roc_auc"] - baseline_overall["roc_auc"])
    relative_pr = float(pr_delta / baseline_overall["pr_auc"]) if baseline_overall["pr_auc"] else None
    status = (
        "SYNTHETIC MODEL STATUS: TRAJECTORY SIGNAL CONFIRMED"
        if pr_delta > 0 and pr_improved >= 4
        else "SYNTHETIC MODEL STATUS: TRAJECTORY SIGNAL NOT CONFIRMED"
    )

    summary = {
        "dataset": {
            "feature_path": str(FEATURE_PATH),
            "rows": int(len(df)),
            "participants": int(df[PERSON_ID].nunique()),
            "positive_rows": int(y.sum()),
            "negative_rows": int(len(df) - y.sum()),
            "positive_rate": float(y.mean()),
        },
        "cross_validation": {
            "method": "GroupKFold",
            "group_column": PERSON_ID,
            "n_splits": N_SPLITS,
            "participant_leakage_check": "PASS" if leakage_pass else "FAIL",
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
            "pr_auc_delta": pr_delta,
            "roc_auc_delta": roc_delta,
            "relative_pr_auc_change": relative_pr,
            "trajectory_pr_auc_improved_folds": int(pr_improved),
            "trajectory_roc_auc_improved_folds": int(roc_improved),
        },
        "folds": fold_summaries,
        "phenotype_oof_results": subgroup_metrics(oof),
        "feature_exclusion_notes": [
            "simulation_phenotype, person_id, and prediction_timestamp were not predictive features.",
            "Simulator-generation parameters were not predictive features.",
            "Simulator event ground truth and future/event-derived fields were not predictive features.",
            "simulation_phenotype was used only for OOF subgroup reporting.",
        ],
        "status": status,
    }

    return oof, results, summary


def fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "unavailable"
    return f"{value:.{digits}f}"


def print_terminal_summary(summary: dict[str, object]) -> None:
    data = summary["dataset"]
    folds = summary["folds"]
    baseline = summary["models"]["current_glucose_baseline"]["overall_oof_metrics"]
    trajectory = summary["models"]["cgm_trajectory_logistic"]["overall_oof_metrics"]
    comparison = summary["comparison"]

    print()
    print("Synthetic participant-separated model evaluation")
    print("=================================================")
    print(f"Rows: {data['rows']:,}")
    print(f"Participants: {data['participants']:,}")
    print(f"Positive target: {data['positive_rows']:,}")
    print(f"Positive rate: {data['positive_rate']:.2%}")

    for fold in folds:
        print()
        print(f"Fold {fold['fold']}:")
        print(f"  Train participants: {fold['train_participants']}")
        print(f"  Validation participants: {fold['validation_participants']}")
        print(f"  Validation rows: {fold['validation_rows']:,}")
        print(f"  Validation positives: {fold['validation_positives']:,}")
        print(f"  Baseline PR-AUC: {fold['baseline_pr_auc']:.4f}")
        print(f"  Trajectory PR-AUC: {fold['trajectory_pr_auc']:.4f}")
        print(f"  Baseline ROC-AUC: {fold['baseline_roc_auc']:.4f}")
        print(f"  Trajectory ROC-AUC: {fold['trajectory_roc_auc']:.4f}")

    print()
    print("Overall OOF")
    print("-----------")
    print("Current-glucose baseline:")
    print(f"  PR-AUC: {fmt(baseline['pr_auc'])}")
    print(f"  ROC-AUC: {fmt(baseline['roc_auc'])}")
    print(f"  Precision @0.50: {fmt(baseline['precision'])}")
    print(f"  Recall @0.50: {fmt(baseline['recall'])}")
    print(f"  F1 @0.50: {fmt(baseline['f1'])}")
    print(f"  Specificity @0.50: {fmt(baseline['specificity'])}")
    print(f"  FPR @0.50: {fmt(baseline['false_positive_rate'])}")
    print()
    print("Trajectory model:")
    print(f"  PR-AUC: {fmt(trajectory['pr_auc'])}")
    print(f"  ROC-AUC: {fmt(trajectory['roc_auc'])}")
    print(f"  Precision @0.50: {fmt(trajectory['precision'])}")
    print(f"  Recall @0.50: {fmt(trajectory['recall'])}")
    print(f"  F1 @0.50: {fmt(trajectory['f1'])}")
    print(f"  Specificity @0.50: {fmt(trajectory['specificity'])}")
    print(f"  FPR @0.50: {fmt(trajectory['false_positive_rate'])}")
    print()
    print("Trajectory improvement:")
    print(f"  PR-AUC absolute: {comparison['pr_auc_delta']:+.4f}")
    print(f"  PR-AUC relative: {comparison['relative_pr_auc_change']:+.2%}")
    print(f"  ROC-AUC absolute: {comparison['roc_auc_delta']:+.4f}")
    print(f"  PR-AUC improved folds: {comparison['trajectory_pr_auc_improved_folds']}/5")
    print(f"  ROC-AUC improved folds: {comparison['trajectory_roc_auc_improved_folds']}/5")
    print()
    print("Phenotype OOF results:")
    for phenotype, item in summary["phenotype_oof_results"].items():
        print(
            f"  {phenotype}: participants={item['participants']}, rows={item['rows']:,}, "
            f"positives={item['positives']:,}, positive_rate={item['positive_rate']:.2%}, "
            f"baseline_PR_AUC={fmt(item['baseline_pr_auc'])}, "
            f"trajectory_PR_AUC={fmt(item['trajectory_pr_auc'])}, "
            f"baseline_ROC_AUC={fmt(item['baseline_roc_auc'])}, "
            f"trajectory_ROC_AUC={fmt(item['trajectory_roc_auc'])}"
        )
    print()
    print(f"Participant leakage check: {summary['cross_validation']['participant_leakage_check']}")
    print(summary["status"])


def main() -> None:
    df = load_dataset()
    oof, results, summary = run_cross_validation(df)

    OUTPUT_PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    oof.to_csv(OUTPUT_PREDICTIONS, index=False)
    results.to_csv(OUTPUT_RESULTS, index=False)
    with OUTPUT_SUMMARY.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print_terminal_summary(summary)


if __name__ == "__main__":
    main()

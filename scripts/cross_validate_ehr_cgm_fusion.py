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


FEATURE_PATH = Path("reports/synthetic_ehr_cgm_fusion_features.csv")
FEATURE_MANIFEST_PATH = Path("reports/synthetic_ehr_cgm_fusion_summary.json")
OUTPUT_PREDICTIONS = Path("reports/ehr_cgm_fusion_oof_predictions.csv")
OUTPUT_RESULTS = Path("reports/ehr_cgm_fusion_cross_validation_results.csv")
OUTPUT_SUMMARY = Path("reports/ehr_cgm_fusion_cross_validation_summary.json")

TARGET = "pre_event_target"
PERSON_ID = "person_id"
PHENOTYPE = "simulation_phenotype"
TIMESTAMP = "prediction_timestamp"
N_SPLITS = 5
RANDOM_STATE = 42

BASELINE_FEATURES = ["current_glucose_mg_dl"]

STRICT_EXCLUSIONS = {
    PERSON_ID,
    TIMESTAMP,
    TARGET,
    PHENOTYPE,
    "synthetic_data_note",
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


def load_feature_manifest() -> tuple[list[str], list[str], list[str]]:
    if not FEATURE_MANIFEST_PATH.exists():
        fail(f"Fusion feature manifest not found: {FEATURE_MANIFEST_PATH}")

    with FEATURE_MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    manifest = summary.get("feature_manifest", {})
    cgm_features = manifest.get("CGM_FEATURES")
    ehr_features = manifest.get("EHR_FEATURES")
    fusion_features = manifest.get("FUSION_FEATURES")

    if not isinstance(cgm_features, list) or not isinstance(ehr_features, list):
        fail("Feature manifest is missing ordered CGM_FEATURES or EHR_FEATURES.")
    if not isinstance(fusion_features, list):
        fail("Feature manifest is missing ordered FUSION_FEATURES.")
    if fusion_features != [*cgm_features, *ehr_features]:
        fail("FUSION_FEATURES does not equal ordered CGM_FEATURES + EHR_FEATURES.")
    if set(fusion_features) & STRICT_EXCLUSIONS:
        fail(
            "Forbidden columns are present in FUSION_FEATURES: "
            + ", ".join(sorted(set(fusion_features) & STRICT_EXCLUSIONS))
        )
    if BASELINE_FEATURES[0] not in cgm_features:
        fail("Current-glucose baseline feature is missing from CGM_FEATURES.")

    return cgm_features, ehr_features, fusion_features


def load_dataset(
    cgm_features: list[str], ehr_features: list[str], fusion_features: list[str]
) -> pd.DataFrame:
    if not FEATURE_PATH.exists():
        fail(f"Fusion feature dataset not found: {FEATURE_PATH}")

    df = pd.read_csv(FEATURE_PATH)
    required = {
        PERSON_ID,
        TIMESTAMP,
        PHENOTYPE,
        TARGET,
        *BASELINE_FEATURES,
        *cgm_features,
        *ehr_features,
        *fusion_features,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        fail("Required columns are missing:\n" + "\n".join(f"  - {column}" for column in missing))

    for name, features in {
        "baseline": BASELINE_FEATURES,
        "cgm_trajectory": cgm_features,
        "ehr_cgm_fusion": fusion_features,
    }.items():
        forbidden = sorted(set(features) & STRICT_EXCLUSIONS)
        if forbidden:
            fail(f"Forbidden columns entered {name} model features: {forbidden}")
        non_numeric = [
            column
            for column in features
            if not pd.api.types.is_numeric_dtype(pd.to_numeric(df[column], errors="coerce"))
        ]
        if non_numeric:
            fail(f"Non-numeric predictive features found for {name}: {non_numeric}")

    df[PERSON_ID] = df[PERSON_ID].astype(str)
    df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP], errors="coerce")
    if df[TIMESTAMP].isna().any():
        fail("Invalid prediction timestamps found.")

    df[TARGET] = df[TARGET].astype(bool).astype(int)
    if df[PERSON_ID].nunique() != 100:
        fail(f"Expected 100 participants; found {df[PERSON_ID].nunique()}.")
    if len(df) != 198000:
        fail(f"Expected 198,000 rows; found {len(df)}.")
    if df[[PERSON_ID, TIMESTAMP]].duplicated().any():
        fail("Duplicate person_id + prediction_timestamp rows found.")
    if df[fusion_features].isna().any().any():
        fail("Missing values found in raw fusion feature matrix before fold-local imputation.")

    return df.reset_index(drop=True)


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


def subgroup_metrics(oof: pd.DataFrame) -> dict[str, dict[str, float | int | None]]:
    results = {}
    for phenotype, group in oof.groupby(PHENOTYPE):
        y = group[TARGET].astype(int).to_numpy()
        results[str(phenotype)] = {
            "participants": int(group[PERSON_ID].nunique()),
            "rows": int(len(group)),
            "positives": int(y.sum()),
            "positive_rate": float(y.mean()),
            "baseline_pr_auc": safe_pr_auc(y, group["current_glucose_baseline_probability"].to_numpy()),
            "trajectory_pr_auc": safe_pr_auc(y, group["cgm_trajectory_probability"].to_numpy()),
            "fusion_pr_auc": safe_pr_auc(y, group["ehr_cgm_fusion_probability"].to_numpy()),
            "baseline_roc_auc": safe_roc_auc(y, group["current_glucose_baseline_probability"].to_numpy()),
            "trajectory_roc_auc": safe_roc_auc(y, group["cgm_trajectory_probability"].to_numpy()),
            "fusion_roc_auc": safe_roc_auc(y, group["ehr_cgm_fusion_probability"].to_numpy()),
        }
    return results


def classify_result(pr_delta: float, roc_delta: float, pr_improved_folds: int) -> str:
    if pr_delta > 0 and pr_improved_folds >= 4:
        return "FUSION IMPROVES SYNTHETIC OOF PERFORMANCE"
    if pr_delta <= 0 and roc_delta <= 0 and pr_improved_folds <= 2:
        return "FUSION DOES NOT IMPROVE SYNTHETIC OOF PERFORMANCE"
    return "MIXED RESULTS"


def run_cross_validation(
    df: pd.DataFrame,
    cgm_features: list[str],
    ehr_features: list[str],
    fusion_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    model_specs = {
        "current_glucose_baseline": BASELINE_FEATURES,
        "cgm_trajectory": cgm_features,
        "ehr_cgm_fusion": fusion_features,
    }
    groups = df[PERSON_ID]
    cv = GroupKFold(n_splits=N_SPLITS)
    probabilities = {name: np.full(len(df), np.nan) for name in model_specs}
    fold_assignments = np.full(len(df), -1, dtype=int)
    fold_records = []
    fold_summaries = []
    leakage_pass = True
    preprocessing_is_fold_local = True

    for fold, (train_idx, val_idx) in enumerate(cv.split(df, df[TARGET], groups=groups), start=1):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        train_people = set(train_df[PERSON_ID])
        val_people = set(val_df[PERSON_ID])
        overlap = train_people & val_people
        if overlap:
            leakage_pass = False
            fail(f"Participant leakage in fold {fold}: {sorted(overlap)}")

        y_val = val_df[TARGET].to_numpy()
        common = {
            "fold": fold,
            "validation_participants": int(len(val_people)),
            "validation_rows": int(len(val_df)),
            "validation_positives": int(y_val.sum()),
            "validation_positive_rate": float(y_val.mean()),
            "train_participants": int(len(train_people)),
        }
        fold_model_metrics = {}

        for model_name, feature_columns in model_specs.items():
            model = build_model()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(train_df[feature_columns], train_df[TARGET])
            prob = model.predict_proba(val_df[feature_columns])[:, 1]
            probabilities[model_name][val_idx] = prob
            metrics = calculate_metrics(y_val, prob)
            fold_model_metrics[model_name] = metrics
            fold_records.append({**common, "model": model_name, "feature_count": len(feature_columns), **metrics})

        fold_assignments[val_idx] = fold
        traj = fold_model_metrics["cgm_trajectory"]
        fusion = fold_model_metrics["ehr_cgm_fusion"]
        fold_summaries.append(
            {
                **common,
                "trajectory_pr_auc": traj["pr_auc"],
                "fusion_pr_auc": fusion["pr_auc"],
                "trajectory_roc_auc": traj["roc_auc"],
                "fusion_roc_auc": fusion["roc_auc"],
                "fusion_minus_trajectory_pr_auc": float(fusion["pr_auc"] - traj["pr_auc"]),
                "fusion_minus_trajectory_roc_auc": float(fusion["roc_auc"] - traj["roc_auc"]),
            }
        )

    if (fold_assignments < 0).any() or any(np.isnan(values).any() for values in probabilities.values()):
        fail("Not every row received out-of-fold predictions.")

    oof = df[[PERSON_ID, TIMESTAMP, PHENOTYPE, TARGET]].copy()
    for model_name, values in probabilities.items():
        oof[f"{model_name}_probability"] = values
    oof["fold"] = fold_assignments

    y = df[TARGET].to_numpy()
    overall_metrics = {
        model_name: calculate_metrics(y, values)
        for model_name, values in probabilities.items()
    }

    for model_name, metrics in overall_metrics.items():
        fold_records.append(
            {
                "fold": "overall_oof",
                "model": model_name,
                "feature_count": len(model_specs[model_name]),
                "validation_participants": int(df[PERSON_ID].nunique()),
                "validation_rows": int(len(df)),
                "validation_positives": int(y.sum()),
                "validation_positive_rate": float(y.mean()),
                "train_participants": None,
                **metrics,
            }
        )

    results = pd.DataFrame(fold_records)

    trajectory_overall = overall_metrics["cgm_trajectory"]
    fusion_overall = overall_metrics["ehr_cgm_fusion"]
    pr_delta = float(fusion_overall["pr_auc"] - trajectory_overall["pr_auc"])
    roc_delta = float(fusion_overall["roc_auc"] - trajectory_overall["roc_auc"])
    relative_pr = float(pr_delta / trajectory_overall["pr_auc"]) if trajectory_overall["pr_auc"] else None
    pr_improved = sum(1 for fold in fold_summaries if fold["fusion_minus_trajectory_pr_auc"] > 0)
    roc_improved = sum(1 for fold in fold_summaries if fold["fusion_minus_trajectory_roc_auc"] > 0)
    conclusion = classify_result(pr_delta, roc_delta, pr_improved)

    summary = {
        "dataset": {
            "feature_path": str(FEATURE_PATH),
            "feature_manifest_path": str(FEATURE_MANIFEST_PATH),
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
            "identical_participant_folds_for_all_models": True,
            "participant_leakage_check": "PASS" if leakage_pass else "FAIL",
            "all_rows_have_oof_predictions": True,
            "preprocessing_fit_inside_each_training_fold": preprocessing_is_fold_local,
            "training_predictions_used_for_evaluation": False,
        },
        "models": {
            "current_glucose_baseline": {
                "features": BASELINE_FEATURES,
                "feature_count": len(BASELINE_FEATURES),
                "overall_oof_metrics": overall_metrics["current_glucose_baseline"],
            },
            "cgm_trajectory": {
                "features": cgm_features,
                "feature_count": len(cgm_features),
                "overall_oof_metrics": overall_metrics["cgm_trajectory"],
            },
            "ehr_cgm_fusion": {
                "features": fusion_features,
                "cgm_feature_count": len(cgm_features),
                "ehr_feature_count": len(ehr_features),
                "feature_count": len(fusion_features),
                "overall_oof_metrics": overall_metrics["ehr_cgm_fusion"],
            },
        },
        "comparison": {
            "fusion_minus_trajectory_pr_auc": pr_delta,
            "fusion_minus_trajectory_roc_auc": roc_delta,
            "relative_pr_auc_change_vs_trajectory": relative_pr,
            "fusion_pr_auc_improved_folds_vs_trajectory": int(pr_improved),
            "fusion_roc_auc_improved_folds_vs_trajectory": int(roc_improved),
        },
        "folds": fold_summaries,
        "phenotype_oof_results": subgroup_metrics(oof),
        "leakage_protection": {
            "no_participant_overlap_between_train_and_validation": "PASS" if leakage_pass else "FAIL",
            "no_future_cgm_predictive_features": "PASS",
            "no_event_labels_as_predictive_features": "PASS",
            "simulation_phenotype_not_model_input": "PASS",
            "person_id_not_model_input": "PASS",
            "no_simulator_ground_truth_parameters_as_model_inputs": "PASS",
            "preprocessing_trained_within_each_fold_only": "PASS",
        },
        "feature_exclusion_notes": [
            "simulation_phenotype, person_id, prediction_timestamp, and pre_event_target were not predictive features.",
            "Simulator-generation parameters were not predictive features.",
            "Simulator event ground truth and future/event-derived fields were not predictive features.",
            "simulation_phenotype was used only for OOF subgroup reporting.",
        ],
        "scientific_limitation": (
            "Synthetic EHR values were generated using simulation phenotype metadata, and "
            "synthetic CGM trajectories were also influenced by phenotype. Fusion performance "
            "therefore reflects a synthetic two-stream POC and is not independent clinical "
            "validation or clinically validated personalization."
        ),
        "status": conclusion,
    }

    return oof, results, summary


def fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "unavailable"
    return f"{value:.{digits}f}"


def print_terminal_summary(summary: dict[str, object]) -> None:
    data = summary["dataset"]
    models = summary["models"]
    comparison = summary["comparison"]

    print()
    print("Synthetic EHR + CGM fusion participant-separated evaluation")
    print("===========================================================")
    print(f"Rows: {data['rows']:,}")
    print(f"Participants: {data['participants']:,}")
    print(f"Positive target: {data['positive_rows']:,}")
    print(f"Positive rate: {data['positive_rate']:.2%}")

    print()
    print("Overall OOF")
    print("-----------")
    for model_name, model in models.items():
        metrics = model["overall_oof_metrics"]
        print(f"{model_name} ({model['feature_count']} features):")
        print(f"  PR-AUC: {fmt(metrics['pr_auc'])}")
        print(f"  ROC-AUC: {fmt(metrics['roc_auc'])}")
        print(f"  Precision @0.50: {fmt(metrics['precision'])}")
        print(f"  Recall @0.50: {fmt(metrics['recall'])}")
        print(f"  F1 @0.50: {fmt(metrics['f1'])}")
        print(f"  Specificity @0.50: {fmt(metrics['specificity'])}")
        print(f"  FPR @0.50: {fmt(metrics['false_positive_rate'])}")
        print()

    print("Fusion vs CGM trajectory:")
    print(f"  PR-AUC delta: {comparison['fusion_minus_trajectory_pr_auc']:+.4f}")
    print(f"  ROC-AUC delta: {comparison['fusion_minus_trajectory_roc_auc']:+.4f}")
    print(f"  Relative PR-AUC change: {comparison['relative_pr_auc_change_vs_trajectory']:+.2%}")
    print(f"  PR-AUC improved folds: {comparison['fusion_pr_auc_improved_folds_vs_trajectory']}/5")
    print(f"  ROC-AUC improved folds: {comparison['fusion_roc_auc_improved_folds_vs_trajectory']}/5")
    print()

    print("Fold-level fusion comparison:")
    for fold in summary["folds"]:
        print(
            f"  Fold {fold['fold']}: "
            f"trajectory_PR_AUC={fold['trajectory_pr_auc']:.4f}, "
            f"fusion_PR_AUC={fold['fusion_pr_auc']:.4f}, "
            f"delta={fold['fusion_minus_trajectory_pr_auc']:+.4f}; "
            f"trajectory_ROC_AUC={fold['trajectory_roc_auc']:.4f}, "
            f"fusion_ROC_AUC={fold['fusion_roc_auc']:.4f}, "
            f"delta={fold['fusion_minus_trajectory_roc_auc']:+.4f}"
        )

    print()
    print("Leakage checks:")
    for check, status in summary["leakage_protection"].items():
        print(f"  {check}: {status}")
    print()
    print(summary["status"])


def main() -> None:
    cgm_features, ehr_features, fusion_features = load_feature_manifest()
    df = load_dataset(cgm_features, ehr_features, fusion_features)
    oof, results, summary = run_cross_validation(df, cgm_features, ehr_features, fusion_features)

    OUTPUT_PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    oof.to_csv(OUTPUT_PREDICTIONS, index=False)
    results.to_csv(OUTPUT_RESULTS, index=False)
    with OUTPUT_SUMMARY.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print_terminal_summary(summary)


if __name__ == "__main__":
    main()

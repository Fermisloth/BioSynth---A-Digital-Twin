from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SYNTHETIC_ROOT = PROJECT_ROOT / "data" / "02_synthetic"
CGM_DIR = SYNTHETIC_ROOT / "cgm"
METADATA_DIR = SYNTHETIC_ROOT / "metadata"
REPORT_DIR = PROJECT_ROOT / "reports"

PARTICIPANT_METADATA_PATH = METADATA_DIR / "synthetic_participants.csv"
EVENTS_PATH = METADATA_DIR / "synthetic_events.csv"
SUMMARY_PATH = REPORT_DIR / "synthetic_cgm_generation_summary.json"

MASTER_RANDOM_SEED = 20261007
PARTICIPANT_COUNT = 100
SIMULATION_DAYS = 7
SAMPLING_INTERVAL_MINUTES = 5
OBSERVATIONS_PER_DAY = int(24 * 60 / SAMPLING_INTERVAL_MINUTES)
OBSERVATIONS_PER_PARTICIPANT = SIMULATION_DAYS * OBSERVATIONS_PER_DAY
GLUCOSE_MIN_MG_DL = 40.0
GLUCOSE_MAX_MG_DL = 400.0
HYPERGLYCEMIA_THRESHOLD_MG_DL = 180.0
START_TIMESTAMP = pd.Timestamp("2026-01-01 00:00:00")


PHENOTYPE_COUNTS = {
    "No diabetes-like": 40,
    "Prediabetes-like": 30,
    "T2D-like": 30,
}


@dataclass(frozen=True)
class PhenotypeProfile:
    baseline_mean: float
    baseline_sd: float
    circadian_range: tuple[float, float]
    noise_range: tuple[float, float]
    meal_response_range: tuple[float, float]
    recovery_range: tuple[float, float]
    excursion_probability: float
    excursion_range: tuple[float, float]


@dataclass(frozen=True)
class ParticipantParameters:
    person_id: str
    simulation_phenotype: str
    baseline_glucose_mg_dl: float
    circadian_amplitude: float
    noise_scale: float
    meal_response_scale: float
    recovery_parameter: float
    excursion_probability: float
    random_seed: int
    observation_count: int
    simulation_days: int


PHENOTYPE_PROFILES = {
    # These are simulation phenotypes for engineering demonstrations, not diagnoses.
    "No diabetes-like": PhenotypeProfile(
        baseline_mean=92,
        baseline_sd=7,
        circadian_range=(4, 10),
        noise_range=(1.8, 4.0),
        meal_response_range=(22, 42),
        recovery_range=(0.050, 0.075),
        excursion_probability=0.04,
        excursion_range=(20, 45),
    ),
    "Prediabetes-like": PhenotypeProfile(
        baseline_mean=112,
        baseline_sd=9,
        circadian_range=(6, 14),
        noise_range=(3.0, 6.0),
        meal_response_range=(38, 68),
        recovery_range=(0.035, 0.058),
        excursion_probability=0.16,
        excursion_range=(35, 75),
    ),
    "T2D-like": PhenotypeProfile(
        baseline_mean=142,
        baseline_sd=16,
        circadian_range=(8, 20),
        noise_range=(4.0, 9.0),
        meal_response_range=(58, 105),
        recovery_range=(0.022, 0.045),
        excursion_probability=0.34,
        excursion_range=(50, 120),
    ),
}


def build_output_directories() -> None:
    for directory in (CGM_DIR, METADATA_DIR, REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def build_population(master_rng: np.random.Generator) -> list[ParticipantParameters]:
    population: list[ParticipantParameters] = []
    participant_index = 1

    for phenotype, count in PHENOTYPE_COUNTS.items():
        profile = PHENOTYPE_PROFILES[phenotype]

        for _ in range(count):
            seed = int(master_rng.integers(1, np.iinfo(np.int32).max))
            rng = np.random.default_rng(seed)

            population.append(
                ParticipantParameters(
                    person_id=f"SYN-{participant_index:04d}",
                    simulation_phenotype=phenotype,
                    baseline_glucose_mg_dl=round(
                        float(rng.normal(profile.baseline_mean, profile.baseline_sd)),
                        3,
                    ),
                    circadian_amplitude=round(
                        float(rng.uniform(*profile.circadian_range)),
                        3,
                    ),
                    noise_scale=round(float(rng.uniform(*profile.noise_range)), 3),
                    meal_response_scale=round(
                        float(rng.uniform(*profile.meal_response_range)),
                        3,
                    ),
                    recovery_parameter=round(
                        float(rng.uniform(*profile.recovery_range)),
                        5,
                    ),
                    excursion_probability=profile.excursion_probability,
                    random_seed=seed,
                    observation_count=OBSERVATIONS_PER_PARTICIPANT,
                    simulation_days=SIMULATION_DAYS,
                )
            )
            participant_index += 1

    return population


def daily_meal_minutes(rng: np.random.Generator, day_index: int) -> list[int]:
    day_offset = day_index * 24 * 60
    meal_centers = [8 * 60, 13 * 60, 19 * 60]
    jitters = rng.normal(0, 35, size=len(meal_centers)).astype(int)
    return [day_offset + center + jitter for center, jitter in zip(meal_centers, jitters)]


def meal_effect(
    minutes_since_event: np.ndarray,
    magnitude: float,
    recovery_parameter: float,
) -> np.ndarray:
    response = minutes_since_event / 35.0
    effect = magnitude * response * np.exp(1 - response)
    effect[minutes_since_event < 0] = 0
    effect[minutes_since_event > (1 / recovery_parameter) * 7.0] = 0
    return effect


def generate_events(
    parameters: ParticipantParameters,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []

    for day_index in range(SIMULATION_DAYS):
        for meal_number, event_minute in enumerate(daily_meal_minutes(rng, day_index), start=1):
            magnitude = float(
                rng.normal(parameters.meal_response_scale, parameters.meal_response_scale * 0.18)
            )

            if rng.random() < parameters.excursion_probability:
                profile = PHENOTYPE_PROFILES[parameters.simulation_phenotype]
                magnitude += float(rng.uniform(*profile.excursion_range))
                event_type = "meal_plus_hyperglycemic_excursion"
            else:
                event_type = "meal_disturbance"

            events.append(
                {
                    "person_id": parameters.person_id,
                    "event_timestamp": START_TIMESTAMP
                    + pd.Timedelta(minutes=int(event_minute)),
                    "event_type": event_type,
                    "event_magnitude": round(max(5.0, magnitude), 3),
                    "event_duration_parameter": parameters.recovery_parameter,
                    "meal_number": meal_number,
                }
            )

    return events


def generate_participant_cgm(
    parameters: ParticipantParameters,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    rng = np.random.default_rng(parameters.random_seed)
    timestamps = pd.date_range(
        start=START_TIMESTAMP,
        periods=OBSERVATIONS_PER_PARTICIPANT,
        freq=f"{SAMPLING_INTERVAL_MINUTES}min",
    )
    elapsed_minutes = np.arange(OBSERVATIONS_PER_PARTICIPANT) * SAMPLING_INTERVAL_MINUTES
    elapsed_days = elapsed_minutes / (24 * 60)

    circadian = parameters.circadian_amplitude * np.sin(
        2 * np.pi * elapsed_days - np.pi / 2
    )

    events = generate_events(parameters, rng)
    event_signal = np.zeros(OBSERVATIONS_PER_PARTICIPANT)

    for event in events:
        event_minute = (
            pd.Timestamp(event["event_timestamp"]) - START_TIMESTAMP
        ).total_seconds() / 60
        event_signal += meal_effect(
            elapsed_minutes - event_minute,
            float(event["event_magnitude"]),
            parameters.recovery_parameter,
        )

    autoregressive_noise = np.zeros(OBSERVATIONS_PER_PARTICIPANT)
    noise_innovations = rng.normal(
        0,
        parameters.noise_scale,
        size=OBSERVATIONS_PER_PARTICIPANT,
    )
    for idx in range(1, OBSERVATIONS_PER_PARTICIPANT):
        autoregressive_noise[idx] = (
            0.82 * autoregressive_noise[idx - 1] + noise_innovations[idx]
        )

    glucose = (
        parameters.baseline_glucose_mg_dl
        + circadian
        + event_signal
        + autoregressive_noise
    )
    glucose = np.clip(glucose, GLUCOSE_MIN_MG_DL, GLUCOSE_MAX_MG_DL)

    cgm = pd.DataFrame(
        {
            "timestamp": timestamps,
            "glucose_value_mg_dl": np.round(glucose, 1),
        }
    )

    return cgm, events


def summarize_generation(
    participants: pd.DataFrame,
    events: pd.DataFrame,
    all_cgm: pd.DataFrame,
) -> dict[str, object]:
    high_mask = all_cgm["glucose_value_mg_dl"] > HYPERGLYCEMIA_THRESHOLD_MG_DL
    participant_high = (
        all_cgm.loc[high_mask, "person_id"]
        .drop_duplicates()
        .shape[0]
    )
    phenotype_high_counts = (
        all_cgm.loc[high_mask]
        .merge(participants[["person_id", "simulation_phenotype"]], on="person_id")
        ["simulation_phenotype"]
        .value_counts()
        .to_dict()
    )

    return {
        "random_seed": MASTER_RANDOM_SEED,
        "participant_count": int(len(participants)),
        "participant_counts_by_phenotype": participants["simulation_phenotype"]
        .value_counts()
        .to_dict(),
        "total_observations": int(len(all_cgm)),
        "simulation_duration": f"{SIMULATION_DAYS} days",
        "sampling_interval": f"{SAMPLING_INTERVAL_MINUTES} minutes",
        "overall_glucose": {
            "min": float(all_cgm["glucose_value_mg_dl"].min()),
            "max": float(all_cgm["glucose_value_mg_dl"].max()),
            "mean": float(all_cgm["glucose_value_mg_dl"].mean()),
            "median": float(all_cgm["glucose_value_mg_dl"].median()),
        },
        "observations_above_180_mg_dl": int(high_mask.sum()),
        "observation_percent_above_180_mg_dl": float(high_mask.mean() * 100),
        "participants_with_at_least_one_observation_above_180": int(participant_high),
        "counts_above_180_by_phenotype": {
            phenotype: int(phenotype_high_counts.get(phenotype, 0))
            for phenotype in PHENOTYPE_COUNTS
        },
        "event_counts_by_type": events["event_type"].value_counts().to_dict(),
        "output_paths": {
            "synthetic_cgm_directory": str(CGM_DIR.relative_to(PROJECT_ROOT)),
            "participant_metadata": str(PARTICIPANT_METADATA_PATH.relative_to(PROJECT_ROOT)),
            "event_metadata": str(EVENTS_PATH.relative_to(PROJECT_ROOT)),
            "summary_json": str(SUMMARY_PATH.relative_to(PROJECT_ROOT)),
        },
        "clinical_validation_note": (
            "Synthetic POC data generated for hackathon demonstration and model "
            "development only; it is not clinically validated and does not represent "
            "clinical diagnoses."
        ),
    }


def print_summary(summary: dict[str, object]) -> None:
    glucose = summary["overall_glucose"]
    print()
    print("Synthetic CGM generation complete")
    print("=" * 40)
    print(f"Participants generated: {summary['participant_count']}")
    print(f"Total CGM observations: {summary['total_observations']:,}")
    print(f"Phenotype distribution: {summary['participant_counts_by_phenotype']}")
    print(
        "Glucose range: "
        f"{glucose['min']:.1f}-{glucose['max']:.1f} mg/dL "
        f"(mean {glucose['mean']:.2f}, median {glucose['median']:.2f})"
    )
    print(
        "Observations >180 mg/dL: "
        f"{summary['observations_above_180_mg_dl']:,} "
        f"({summary['observation_percent_above_180_mg_dl']:.2f}%)"
    )
    print(
        "Participants with hyperglycemic observations: "
        f"{summary['participants_with_at_least_one_observation_above_180']}"
    )
    print(f"Output CGM directory: {CGM_DIR}")
    print(f"Output metadata directory: {METADATA_DIR}")
    print(f"Summary JSON: {SUMMARY_PATH}")


def main() -> int:
    try:
        build_output_directories()
        master_rng = np.random.default_rng(MASTER_RANDOM_SEED)
        population = build_population(master_rng)

        participant_rows = []
        event_rows = []
        cgm_frames = []

        for parameters in population:
            cgm, events = generate_participant_cgm(parameters)
            cgm.to_csv(CGM_DIR / f"{parameters.person_id}.csv", index=False)

            participant_rows.append(asdict(parameters))
            event_rows.extend(events)
            cgm_frames.append(cgm.assign(person_id=parameters.person_id))

        participants = pd.DataFrame(participant_rows)
        events = pd.DataFrame(event_rows)
        all_cgm = pd.concat(cgm_frames, ignore_index=True)

        participants.to_csv(PARTICIPANT_METADATA_PATH, index=False)
        events.to_csv(EVENTS_PATH, index=False)

        summary = summarize_generation(participants, events, all_cgm)
        with SUMMARY_PATH.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        print_summary(summary)
        return 0

    except Exception as exc:
        print(f"[ERROR] Synthetic CGM generation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

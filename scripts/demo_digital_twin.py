from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.biosynth_twin import BioSynthTwin  # noqa: E402


CGM_DIR = PROJECT_ROOT / "data" / "02_synthetic" / "cgm"
EVENTS_PATH = PROJECT_ROOT / "reports" / "synthetic_detected_hyperglycemic_events.csv"


def fail(message: str) -> None:
    raise RuntimeError(f"\nERROR: {message}")


def choose_demo_case(twin: BioSynthTwin) -> tuple[str, pd.Timestamp, dict]:
    if not EVENTS_PATH.exists():
        fail(f"Synthetic detected event file not found: {EVENTS_PATH}")

    events = pd.read_csv(EVENTS_PATH)
    required = {"person_id", "start_time", "peak_glucose_mg_dl"}
    missing = required - set(events.columns)
    if missing:
        fail(f"Event file is missing columns: {sorted(missing)}")

    events["start_time"] = pd.to_datetime(events["start_time"], errors="coerce")
    events = events.dropna(subset=["start_time"]).sort_values(
        ["peak_glucose_mg_dl", "start_time"],
        ascending=[False, True],
    )
    if events.empty:
        fail("No synthetic hyperglycemic events available for demo selection.")

    for event in events.head(75).itertuples(index=False):
        cgm_path = CGM_DIR / f"{event.person_id}.csv"
        if not cgm_path.exists():
            continue

        cgm = pd.read_csv(cgm_path)
        cgm["timestamp"] = pd.to_datetime(cgm["timestamp"], errors="coerce")
        best_state = None
        best_time = None

        for lead_minutes in range(120, 10, -5):
            prediction_time = event.start_time - pd.Timedelta(minutes=lead_minutes)
            historical_cgm = cgm[cgm["timestamp"] <= prediction_time].copy()
            if historical_cgm.empty:
                continue
            try:
                state = twin.predict_state(historical_cgm, prediction_time)
            except ValueError:
                continue
            if best_state is None or state["hyperglycemia_risk"] > best_state["hyperglycemia_risk"]:
                best_state = state
                best_time = prediction_time

        if best_state is not None and best_state["hyperglycemia_risk"] >= 0.30:
            return str(event.person_id), best_time, best_state

    fail("Could not find a demo participant with raw CGM and an event.")


def format_change(value: float) -> str:
    if pd.isna(value):
        return "unavailable"
    return f"{value:+.1f} mg/dL"


def main() -> None:
    twin = BioSynthTwin()
    person_id, prediction_time, state = choose_demo_case(twin)
    cgm_path = CGM_DIR / f"{person_id}.csv"
    cgm = pd.read_csv(cgm_path)
    cgm["timestamp"] = pd.to_datetime(cgm["timestamp"], errors="coerce")

    historical_cgm = cgm[cgm["timestamp"] <= prediction_time].copy()
    if historical_cgm.empty:
        fail("Selected demo timestamp has no historical CGM.")

    if (historical_cgm["timestamp"] > prediction_time).any():
        fail("Future CGM leaked into demo runtime input.")

    if not 0.0 <= state["hyperglycemia_risk"] <= 1.0:
        fail("Runtime returned a risk outside [0, 1].")
    if state["feature_count"] != 18:
        fail("Runtime did not provide exactly 18 model features.")

    print("=" * 52)
    print("BIOSYNTH DIGITAL TWIN")
    print("=" * 52)
    print(f"Participant:           {person_id}")
    print(f"Prediction time:       {state['prediction_timestamp']}")
    print(f"History available:     {state['trajectory_actual_history_minutes']:.0f} min")
    print()
    print("CURRENT STATE")
    print(f"Current glucose:       {state['current_glucose_mg_dl']:.1f} mg/dL")
    print(f"Glucose state:         {state['glucose_state']}")
    print(f"Trend:                 {state['trend']}")
    print(f"15-min change:         {format_change(state['glucose_change_15m'])}")
    print(f"30-min change:         {format_change(state['glucose_change_30m'])}")
    print(f"60-min change:         {format_change(state['glucose_change_60m'])}")
    print(f"Slope:                 {state['glucose_slope_per_hour']:+.1f} mg/dL/hour")
    print()
    print("PREDICTION")
    print(f"Hyperglycemia risk:    {state['hyperglycemia_risk']:.1%}")
    print(f"Risk state:            {state['risk_state']}")
    print(f"Prediction horizon:    {state['prediction_horizon_minutes']} min")
    print()
    print("POC NOTICE")
    print("Synthetic virtual participant.")
    print("Not clinically validated.")
    print("=" * 52)


if __name__ == "__main__":
    main()

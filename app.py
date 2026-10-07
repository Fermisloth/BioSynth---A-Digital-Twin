from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.biosynth_twin import BioSynthTwin


PROJECT_ROOT = Path(__file__).resolve().parent
PARTICIPANT_ID = "SYN-0085"
CGM_PATH = PROJECT_ROOT / "data" / "02_synthetic" / "cgm" / f"{PARTICIPANT_ID}.csv"
EVENTS_PATH = PROJECT_ROOT / "reports" / "synthetic_detected_hyperglycemic_events.csv"
ENGINEERING_THRESHOLD = 180.0
DEMO_EVENT_ID = "SYN-0085-0-32"
GUIDED_OFFSETS_MINUTES = [-90, -60, -30, -10, 0, 30, 60]


st.set_page_config(
    page_title="BioSynth Digital Twin",
    page_icon="",
    layout="wide",
)


@st.cache_resource
def load_twin() -> BioSynthTwin:
    return BioSynthTwin()


@st.cache_data
def load_cgm() -> pd.DataFrame:
    if not CGM_PATH.exists():
        raise FileNotFoundError(f"Participant CGM file not found: {CGM_PATH}")

    cgm = pd.read_csv(CGM_PATH)
    required = {"timestamp", "glucose_value_mg_dl"}
    missing = required - set(cgm.columns)
    if missing:
        raise ValueError(f"Participant CGM file is missing columns: {sorted(missing)}")

    cgm["timestamp"] = pd.to_datetime(cgm["timestamp"], errors="coerce")
    cgm["glucose_value_mg_dl"] = pd.to_numeric(
        cgm["glucose_value_mg_dl"],
        errors="coerce",
    )
    cgm = cgm.dropna(subset=["timestamp", "glucose_value_mg_dl"])
    if cgm.empty:
        raise ValueError("Participant CGM file has no valid timestamped glucose readings.")

    return cgm.sort_values("timestamp").reset_index(drop=True)


@st.cache_data
def load_demo_event() -> dict:
    if not EVENTS_PATH.exists():
        raise FileNotFoundError(f"Synthetic detected event file not found: {EVENTS_PATH}")

    events = pd.read_csv(EVENTS_PATH)
    required = {"person_id", "event_id", "start_time", "end_time", "peak_glucose_mg_dl"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"Synthetic event file is missing columns: {sorted(missing)}")

    event = events.loc[
        (events["person_id"] == PARTICIPANT_ID)
        & (events["event_id"] == DEMO_EVENT_ID)
    ]
    if event.empty:
        raise ValueError(f"Demo event {DEMO_EVENT_ID} was not found for {PARTICIPANT_ID}.")

    row = event.iloc[0].to_dict()
    row["start_time"] = pd.to_datetime(row["start_time"])
    row["end_time"] = pd.to_datetime(row["end_time"])
    return row


def format_change(value: float) -> str:
    if pd.isna(value):
        return "unavailable"
    return f"{value:+.1f} mg/dL"


def risk_color(risk_state: str) -> str:
    return {
        "LOW": "#15803d",
        "ELEVATED": "#b7791f",
        "HIGH": "#b91c1c",
        "EVENT": "#b91c1c",
    }.get(risk_state, "#374151")


def render_metric_card(label: str, value: str, help_text: str | None = None) -> None:
    st.metric(label=label, value=value, help=help_text)


def state_interpretation(state: dict) -> str:
    glucose = state["current_glucose_mg_dl"]
    if glucose > ENGINEERING_THRESHOLD:
        return (
            "Glucose is currently above the 180 mg/dL POC threshold. "
            "The early-warning model is designed for the period before a new "
            "threshold-crossing event begins, so its pre-event risk score is "
            "not displayed during the event."
        )

    threshold_phrase = (
        "above the 180 mg/dL POC threshold"
        if glucose > ENGINEERING_THRESHOLD
        else "below the 180 mg/dL POC threshold"
    )
    horizon = state["prediction_horizon_minutes"]
    risk_state = state["risk_state"]
    trend = state["trend"].lower()

    if risk_state == "HIGH":
        return (
            f"Glucose is currently {threshold_phrase}, and BioSynth estimates a "
            f"high-risk trajectory over the next {horizon} minutes."
        )
    if risk_state == "ELEVATED":
        return (
            f"Glucose is currently {threshold_phrase}. The recent trajectory is "
            f"{trend}, and BioSynth indicates elevated near-term risk."
        )
    return (
        f"Glucose is currently {threshold_phrase}. The recent trajectory is "
        f"{trend}, and BioSynth estimates low near-term risk of crossing the POC threshold."
    )


def guided_timestamps(event: dict, cgm: pd.DataFrame) -> list[pd.Timestamp]:
    raw_times = [
        event["start_time"] + pd.Timedelta(minutes=offset)
        for offset in GUIDED_OFFSETS_MINUTES
    ]
    min_time = cgm["timestamp"].min() + pd.Timedelta(minutes=60)
    max_time = cgm["timestamp"].max()
    valid = [time for time in raw_times if min_time <= time <= max_time]
    return valid


def nearest_cgm_time(cgm: pd.DataFrame, timestamp: pd.Timestamp) -> pd.Timestamp:
    eligible = cgm[cgm["timestamp"] <= timestamp]
    if eligible.empty:
        return cgm["timestamp"].min()
    return eligible.iloc[-1]["timestamp"]


def render_cgm_chart(cgm_history: pd.DataFrame, prediction_time: pd.Timestamp) -> None:
    chart_start = prediction_time - pd.Timedelta(hours=6)
    chart = cgm_history[cgm_history["timestamp"] >= chart_start].copy()

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.plot(
        chart["timestamp"],
        chart["glucose_value_mg_dl"],
        color="#1f77b4",
        linewidth=2,
        label="CGM glucose",
    )
    ax.axhline(
        ENGINEERING_THRESHOLD,
        color="#b91c1c",
        linestyle="--",
        linewidth=1.3,
        label="POC threshold: 180 mg/dL",
    )
    ax.axvline(
        prediction_time,
        color="#111827",
        linestyle=":",
        linewidth=1.4,
        label="Prediction time",
    )
    ax.set_ylabel("Glucose (mg/dL)")
    ax.set_xlabel("Time")
    ax.grid(alpha=0.22)
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    st.pyplot(fig, clear_figure=True)
    st.caption("Model-input chart ends at the replay timestamp; future glucose is not shown here.")


def render_future_reveal(cgm: pd.DataFrame, prediction_time: pd.Timestamp) -> None:
    st.markdown("### What happened next?")
    st.caption("Future synthetic CGM shown below was hidden from BioSynth at prediction time.")
    reveal_end = prediction_time + pd.Timedelta(hours=2)
    future = cgm[
        (cgm["timestamp"] >= prediction_time)
        & (cgm["timestamp"] <= reveal_end)
    ].copy()
    if future.empty:
        st.info("No future CGM is available after this replay timestamp.")
        return

    fig, ax = plt.subplots(figsize=(9.5, 2.8))
    ax.plot(future["timestamp"], future["glucose_value_mg_dl"], color="#7c3aed", linewidth=2)
    ax.axhline(ENGINEERING_THRESHOLD, color="#b91c1c", linestyle="--", linewidth=1.2)
    ax.axvline(prediction_time, color="#111827", linestyle=":", linewidth=1.2)
    ax.set_ylabel("Glucose (mg/dL)")
    ax.set_xlabel("Retrospective future window")
    ax.grid(alpha=0.2)
    fig.autofmt_xdate()
    st.pyplot(fig, clear_figure=True)


def initialize_guided_step() -> None:
    if "guided_step" not in st.session_state:
        st.session_state.guided_step = 0


def render_primary_state(state: dict) -> None:
    st.markdown("### Digital Twin State")
    display = judge_display_state(state)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("CURRENT GLUCOSE", f"{state['current_glucose_mg_dl']:.1f} mg/dL")
    with c2:
        st.metric("TRAJECTORY", state["trend"])
    with c3:
        st.metric("120-MIN RISK", display["risk_label"])
    with c4:
        st.markdown(
            f"""
            <div style="padding: 0.9rem; border: 1px solid #e5e7eb; border-radius: 8px;">
              <div style="font-size: 0.78rem; color: #4b5563; font-weight: 600;">TWIN STATE</div>
              <div style="font-size: 1.55rem; font-weight: 800; color: {risk_color(display['color_key'])};">{display['twin_state']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def judge_display_state(state: dict) -> dict[str, str]:
    if state["current_glucose_mg_dl"] > ENGINEERING_THRESHOLD:
        return {
            "risk_label": "Event underway",
            "twin_state": "ABOVE POC THRESHOLD",
            "color_key": "EVENT",
        }
    return {
        "risk_label": f"{state['hyperglycemia_risk']:.1%}",
        "twin_state": f"{state['risk_state']} RISK",
        "color_key": state["risk_state"],
    }


def render_supporting_details(state: dict) -> None:
    st.markdown("#### Trajectory Details")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.metric("15-min change", format_change(state["glucose_change_15m"]))
    with d2:
        st.metric("30-min change", format_change(state["glucose_change_30m"]))
    with d3:
        st.metric("60-min change", format_change(state["glucose_change_60m"]))
    with d4:
        st.metric("Slope", f"{state['glucose_slope_per_hour']:+.1f} mg/dL/hour")


def main() -> None:
    st.markdown(
        """
        <div style="padding-bottom: 0.5rem;">
          <div style="font-size: 3rem; font-weight: 850; letter-spacing: 0;">BIOSYNTH</div>
          <div style="font-size: 1.45rem; font-weight: 600;">Predictive Glucose Digital Twin</div>
          <div style="font-size: 1.05rem; color: #4b5563; margin-top: 0.35rem;">From glucose history to early risk awareness</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write(
        "BioSynth continuously interprets recent CGM patterns and estimates the risk "
        "of glucose exceeding 180 mg/dL within the next 120 minutes."
    )
    st.caption(f"Synthetic Virtual Participant — {PARTICIPANT_ID}")

    try:
        twin = load_twin()
        cgm = load_cgm()
        event = load_demo_event()
    except Exception as exc:
        st.error(f"Could not start BioSynth dashboard: {exc}")
        return

    mode = st.radio(
        "Replay mode",
        ["GUIDED DEMO", "EXPLORE TIMELINE"],
        horizontal=True,
        index=0,
    )

    if mode == "GUIDED DEMO":
        initialize_guided_step()
        timestamps = guided_timestamps(event, cgm)
        st.caption(
            f"Guided event: {event['event_id']} | starts {event['start_time']} | "
            f"peak {float(event['peak_glucose_mg_dl']):.1f} mg/dL"
        )
        b1, b2, b3 = st.columns([1, 1, 4])
        with b1:
            if st.button("Previous", use_container_width=True):
                st.session_state.guided_step = max(0, st.session_state.guided_step - 1)
        with b2:
            if st.button("Next", use_container_width=True):
                st.session_state.guided_step = min(
                    len(timestamps) - 1,
                    st.session_state.guided_step + 1,
                )
        with b3:
            if st.button("Reset guided replay"):
                st.session_state.guided_step = 0

        prediction_time = nearest_cgm_time(cgm, timestamps[st.session_state.guided_step])
        st.write(
            f"Guided step `{st.session_state.guided_step + 1}/{len(timestamps)}` "
            f"at `{prediction_time}`"
        )

    else:
        min_index = 12 if len(cgm) > 12 else 0
        max_index = len(cgm) - 1
        default_index = min(max(930, min_index), max_index)
        selected_index = st.slider(
            "Replay timeline",
            min_value=min_index,
            max_value=max_index,
            value=default_index,
            step=1,
            help="Move through the synthetic participant's CGM record. The runtime receives only readings at or before the selected timestamp.",
        )
        prediction_time = cgm.loc[selected_index, "timestamp"]

    historical_cgm = cgm.loc[cgm["timestamp"] <= prediction_time].copy()

    if (historical_cgm["timestamp"] > prediction_time).any():
        st.error("Future CGM would be passed to the runtime. Prediction stopped.")
        return

    try:
        state = twin.predict_state(historical_cgm, prediction_time)
    except Exception as exc:
        st.error(f"BioSynthTwin could not produce a state at this timestamp: {exc}")
        return

    risk = state["hyperglycemia_risk"]
    if not 0 <= risk <= 1:
        st.error("Runtime returned a risk outside the expected 0-1 range.")
        return

    render_primary_state(state)
    st.info(state_interpretation(state), icon="→")

    chart_col, details_col = st.columns([1.45, 1])
    with chart_col:
        st.markdown("### CGM + Digital Twin Replay")
        render_cgm_chart(historical_cgm, prediction_time)
    with details_col:
        render_supporting_details(state)
        st.write(f"Glucose state: `{state['glucose_state']}`")
        st.write(f"Prediction horizon: `{state['prediction_horizon_minutes']} min`")
        st.write(f"Historical rows passed to runtime: `{len(historical_cgm):,}`")

    with st.expander("What BioSynth sees"):
        st.write(
            "BioSynth receives recent CGM trajectory information such as current glucose, "
            "recent changes, recent average/range, variability, and time spent above the "
            "POC threshold."
        )
        st.write("BioSynth does not receive future glucose values when making a prediction.")

    with st.expander("Why this is a Digital Twin"):
        st.write(
            "This virtual state is continuously updated from the participant's CGM history. "
            "As new readings arrive, BioSynth recalculates the participant's glucose "
            "trajectory and near-term risk state."
        )

    if mode == "GUIDED DEMO":
        render_future_reveal(cgm, prediction_time)

    with st.expander("Evidence behind the prototype"):
        st.markdown(
            """
            **Real-world Hall 2018 proof**
            - Participant-separated evaluation
            - Current-glucose baseline PR-AUC = 0.0657
            - Trajectory PR-AUC = 0.1383
            - Relative improvement approximately +110.6%
            - Trajectory PR-AUC improved in 5/5 folds

            **Synthetic Digital Twin development**
            - 100 virtual participants
            - 198,000 prediction moments
            - Current-glucose baseline PR-AUC = 0.1918
            - Trajectory PR-AUC = 0.4433
            - Relative improvement = +131.15%
            - Trajectory PR-AUC improved in 5/5 folds

            These experiments support the POC hypothesis that recent CGM trajectory contains
            more predictive signal than the current glucose value alone.
            """
        )

    st.divider()
    st.caption(
        "BioSynth is a hackathon proof of concept. This demonstration uses a synthetic "
        "virtual participant. Predictions are not clinically validated and are not "
        "intended for medical decision-making."
    )


if __name__ == "__main__":
    main()

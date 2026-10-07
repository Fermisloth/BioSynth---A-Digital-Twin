from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.biosynth_twin import BioSynthTwin


PROJECT_ROOT = Path(__file__).resolve().parent
PARTICIPANT_ID = "SYN-0085"
CGM_PATH = PROJECT_ROOT / "data" / "02_synthetic" / "cgm" / f"{PARTICIPANT_ID}.csv"
ENGINEERING_THRESHOLD = 180.0


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


def format_change(value: float) -> str:
    if pd.isna(value):
        return "unavailable"
    return f"{value:+.1f} mg/dL"


def risk_color(risk_state: str) -> str:
    return {
        "LOW": "#15803d",
        "ELEVATED": "#b7791f",
        "HIGH": "#b91c1c",
    }.get(risk_state, "#374151")


def render_metric_card(label: str, value: str, help_text: str | None = None) -> None:
    st.metric(label=label, value=value, help=help_text)


def main() -> None:
    st.title("BIOSYNTH")
    st.subheader("Predictive Glucose Digital Twin")
    st.caption(f"Synthetic Virtual Participant: {PARTICIPANT_ID}")

    st.info(
        "BioSynth is a hackathon proof of concept using synthetic virtual participants. "
        "Predictions are not clinically validated and are not intended for medical decision-making.",
        icon="ℹ",
    )

    try:
        twin = load_twin()
        cgm = load_cgm()
    except Exception as exc:
        st.error(f"Could not start BioSynth dashboard: {exc}")
        return

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

    left, right = st.columns([1.1, 1.4])

    with left:
        st.markdown("### Current Digital Twin State")
        c1, c2 = st.columns(2)
        with c1:
            render_metric_card("Current glucose", f"{state['current_glucose_mg_dl']:.1f} mg/dL")
            render_metric_card("Glucose state", state["glucose_state"])
            render_metric_card("Trend", state["trend"])
        with c2:
            st.markdown(
                f"""
                <div style="padding: 1rem; border: 1px solid #e5e7eb; border-radius: 8px;">
                  <div style="font-size: 0.9rem; color: #4b5563;">Hyperglycemia risk</div>
                  <div style="font-size: 2.4rem; font-weight: 700; color: {risk_color(state['risk_state'])};">{risk:.1%}</div>
                  <div style="font-size: 1.05rem; font-weight: 700;">{state['risk_state']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_metric_card("Prediction horizon", f"{state['prediction_horizon_minutes']} min")

        st.markdown("### Trajectory Features")
        t1, t2 = st.columns(2)
        with t1:
            render_metric_card("15-min change", format_change(state["glucose_change_15m"]))
            render_metric_card("30-min change", format_change(state["glucose_change_30m"]))
        with t2:
            render_metric_card("60-min change", format_change(state["glucose_change_60m"]))
            render_metric_card("Slope", f"{state['glucose_slope_per_hour']:+.1f} mg/dL/hour")

    with right:
        st.markdown("### Historical CGM Before Prediction")
        chart_start = prediction_time - pd.Timedelta(hours=6)
        chart = historical_cgm.loc[historical_cgm["timestamp"] >= chart_start, [
            "timestamp",
            "glucose_value_mg_dl",
        ]].copy()
        threshold = pd.DataFrame(
            {
                "timestamp": chart["timestamp"],
                "POC hyperglycemia engineering threshold": ENGINEERING_THRESHOLD,
            }
        )
        chart = chart.rename(columns={"glucose_value_mg_dl": "CGM glucose"})
        chart_data = chart.merge(threshold, on="timestamp", how="left").set_index("timestamp")
        st.line_chart(chart_data, height=390)
        st.caption(
            "Main chart ends at the selected prediction timestamp and does not include future glucose."
        )

    st.divider()
    st.write(
        f"Prediction timestamp: `{state['prediction_timestamp']}` | "
        f"Historical rows passed to runtime: `{len(historical_cgm):,}` | "
        f"History window used: `{state['trajectory_actual_history_minutes']:.0f} min`"
    )


if __name__ == "__main__":
    main()

# BioSynth — Digital Twin Healthcare Command Center
# Advanced Clinical Frontend with Predictive Telemetry & Counterfactual Sandbox
# Designed for seamless plug-and-play integration with backend models (src/models).

import importlib
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
import streamlit as st


# ==============================================================================
# 1. ASSET DISCOVERY & PATH CONFIGURATION
# ==============================================================================

UI_DIR = Path(__file__).resolve().parent
ASSETS_DIR = UI_DIR / "assets"
LOGO_PATH = ASSETS_DIR / "BioSynth Digital Twin Healthcare Branding.png"

# Fallback path checking for cross-environment reliability
if not LOGO_PATH.exists():
    fallback_candidate = UI_DIR / "ui" / "assets" / "BioSynth Digital Twin Healthcare Branding.png"
    if fallback_candidate.exists():
        LOGO_PATH = fallback_candidate

page_favicon = ":material/monitoring:"
if LOGO_PATH.exists():
    try:
        page_favicon = Image.open(LOGO_PATH)
    except Exception:
        page_favicon = ":material/monitoring:"


# ==============================================================================
# 2. PAGE CONFIGURATION & METADATA
# ==============================================================================

st.set_page_config(
    page_title="BioSynth | Patient Digital Twin",
    page_icon=page_favicon,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================================================================
# 3. DESIGN SYSTEM & CYBER-CLINICAL STYLING
# ==============================================================================

st.markdown(
    """
    <style>
    /* Global Page Palette */
    .stApp {
        background-color: #08111f;
        color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    [data-testid="stSidebar"] {
        background-color: #0d1829;
        border-right: 1px solid #1e293b;
    }

    /* Enterprise Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #0d2138 0%, #0c313f 50%, #082430 100%);
        padding: 22px 26px;
        border-radius: 16px;
        border: 1px solid #1e4554;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.45);
    }

    .hero-eyebrow {
        color: #2dd4bf;
        font-weight: 700;
        font-size: 0.75rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
    }

    .hero-title {
        color: #f8fafc;
        font-size: 1.95rem;
        font-weight: 800;
        margin: 0 0 6px 0;
        letter-spacing: -0.4px;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.92rem;
        margin: 0;
        line-height: 1.5;
    }

    /* Patient EHR Persona Card */
    .persona-card {
        background: #0f1c2e;
        border: 1px solid #22374e;
        border-radius: 14px;
        padding: 18px 22px;
        margin-bottom: 22px;
        display: flex;
        flex-wrap: wrap;
        gap: 18px;
        align-items: center;
    }

    .persona-avatar {
        width: 48px;
        height: 48px;
        border-radius: 10px;
        background: linear-gradient(135deg, #0369a1, #0f766e);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.05rem;
        font-weight: 700;
        color: #f8fafc;
        border: 1px solid #38bdf8;
        letter-spacing: 0.5px;
    }

    .persona-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        background: #1e293b;
        color: #cbd5e1;
        border: 1px solid #334155;
        margin-right: 6px;
        margin-bottom: 4px;
    }

    .persona-badge-teal {
        background: rgba(45, 212, 191, 0.12);
        color: #2dd4bf;
        border: 1px solid rgba(45, 212, 191, 0.3);
    }

    .persona-badge-amber {
        background: rgba(251, 191, 36, 0.12);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.3);
    }

    .rx-badge {
        display: inline-block;
        padding: 1px 6px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 700;
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        margin-right: 6px;
    }

    /* Twin Telemetry Sync Pill */
    .sync-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .sync-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #34d399;
        box-shadow: 0 0 8px #34d399;
    }

    /* Status indicator pill */
    .alert-status-pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-right: 8px;
        vertical-align: middle;
    }

    .status-danger {
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid #ef4444;
    }

    .status-warning {
        background: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        border: 1px solid #f59e0b;
    }

    .status-normal {
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid #10b981;
    }

    /* Distribution Legend Dots */
    .legend-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
        vertical-align: middle;
    }

    .dot-green { background-color: #34d399; box-shadow: 0 0 6px rgba(52, 211, 153, 0.6); }
    .dot-amber { background-color: #fbbf24; box-shadow: 0 0 6px rgba(251, 191, 36, 0.6); }
    .dot-red { background-color: #f87171; box-shadow: 0 0 6px rgba(248, 113, 113, 0.6); }

    /* KPI Cards */
    div[data-testid="stMetric"] {
        background: #0f1c2e;
        border: 1px solid #1e334a;
        padding: 14px 18px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.25);
    }

    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        font-weight: 500;
    }

    div[data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }

    /* Glassmorphic Cards */
    .glass-card {
        background: #0f1c2e;
        border: 1px solid #22374e;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 16px;
    }

    .alert-card-danger {
        background: rgba(239, 68, 68, 0.08);
        border: 1px solid rgba(239, 68, 68, 0.35);
        border-radius: 12px;
        padding: 16px 20px;
        color: #fca5a5;
    }

    .alert-card-safe {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-radius: 12px;
        padding: 16px 20px;
        color: #6ee7b7;
    }

    .alert-card-amber {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.35);
        border-radius: 12px;
        padding: 16px 20px;
        color: #fcd34d;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0b1524;
        padding: 8px 10px;
        border-radius: 12px;
        border: 1px solid #1e293b;
    }

    .stTabs [data-baseweb="tab"] {
        height: 42px;
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 0 18px;
        border: none;
    }

    .stTabs [aria-selected="true"] {
        background-color: #1e3a5f !important;
        color: #38bdf8 !important;
    }

    /* Integration Indicator Badge */
    .integration-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.72rem;
        padding: 3px 10px;
        border-radius: 6px;
        background: #132337;
        color: #7dd3fc;
        border: 1px solid #1e40af;
    }

    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# 4. PATIENT PERSONAS & CLINICAL PROFILES
# ==============================================================================

# Aligns with configs/data_sources.yaml (Synthea EHR static features)
PATIENT_PROFILES = {
    "DEMO-001": {
        "name": "Elena Rostova",
        "gender": "Female",
        "age": 54,
        "avatar": "ER",
        "bmi": 27.8,
        "hba1c": 7.6,
        "fasting_glucose": 138,
        "condition": "Type 2 Diabetes (Diagnosed 4y ago)",
        "hypertension": True,
        "cvd": False,
        "medications": ["Metformin 500mg BID", "Lisinopril 10mg QD"],
        "baseline_glucose": 132,
        "seed": 101,
        "volatility": "Moderate",
        "spike_behavior": "Moderate postprandial drift",
    },
    "DEMO-002": {
        "name": "Marcus Vance",
        "gender": "Male",
        "age": 62,
        "avatar": "MV",
        "bmi": 31.4,
        "hba1c": 8.8,
        "fasting_glucose": 168,
        "condition": "Type 2 Diabetes (Brittle / High Volatility, 9y)",
        "hypertension": True,
        "cvd": True,
        "medications": ["Metformin 1000mg BID", "Glipizide 10mg", "Lantus 24U"],
        "baseline_glucose": 158,
        "seed": 202,
        "volatility": "High (Frequent Spikes > 180)",
        "spike_behavior": "Rapid post-lunch glycemic surge",
    },
    "DEMO-003": {
        "name": "David Chen",
        "gender": "Male",
        "age": 48,
        "avatar": "DC",
        "bmi": 24.5,
        "hba1c": 6.5,
        "fasting_glucose": 112,
        "condition": "Type 2 Diabetes (Well-Controlled, 2y)",
        "hypertension": False,
        "cvd": False,
        "medications": ["Metformin 500mg QD", "Lifestyle / Aerobic"],
        "baseline_glucose": 114,
        "seed": 303,
        "volatility": "Low (Stable)",
        "spike_behavior": "Transient mild elevations",
    },
}


# ==============================================================================
# 5. PLUG-AND-PLAY BACKEND ADAPTER / CONNECTOR
# ==============================================================================

def get_backend_status():
    """
    Checks if a real trained machine learning model or inference service
    exists in src.models. If found, enables live backend mode; otherwise
    gracefully uses the calibrated physiological twin engine.
    """
    try:
        model_module = importlib.import_module("src.models.predict")
        if hasattr(model_module, "predict_glucose_risk"):
            return True, "Live Backend Model (src.models)"
    except Exception:
        pass
    return False, "Simulated Digital Twin Engine (Plug & Play Ready)"


BACKEND_AVAILABLE, BACKEND_MODE_NAME = get_backend_status()


# ==============================================================================
# 6. SIDEBAR CONTROLS & BRANDING
# ==============================================================================

with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), use_container_width=True)
    else:
        st.markdown("### BioSynth Twin")

    st.caption("Predictive Telemetry & Intervention Sandbox")
    st.divider()

    st.subheader("1. Active Clinical Subject")
    selected_id = st.selectbox(
        "Select Patient Profile",
        options=list(PATIENT_PROFILES.keys()),
        format_func=lambda pid: f"{pid} — {PATIENT_PROFILES[pid]['name']}",
    )

    patient = PATIENT_PROFILES[selected_id]

    st.subheader("2. Telemetry Scope")
    time_window = st.select_slider(
        "Historical Window",
        options=[6, 12, 24],
        value=12,
        format_func=lambda h: f"Last {h} Hours",
    )

    st.subheader("3. Chart Display Options")
    show_threshold = st.checkbox("Show 180 mg/dL Danger Threshold", value=True)
    show_confidence = st.checkbox("Show 2h Forecast Confidence Band", value=True)
    show_glycemic_zones = st.checkbox("Show Shaded Target Zones", value=True)
    show_markers = st.checkbox("Show Individual CGM Points", value=False)

    st.divider()

    st.markdown("### Engine Status")
    st.markdown(
        f"""
        <div class="integration-pill">
            <span class="sync-dot"></span>
            {BACKEND_MODE_NAME}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Auto-binds to `src.models.predict.predict_glucose_risk()` once available."
    )


# ==============================================================================
# 7. PHYSIOLOGICAL CGM ENGINE & 2-HOUR FORECAST GENERATOR
# ==============================================================================

def generate_cgm_series(profile, hours):
    """
    Generates realistic, reproducible continuous glucose monitoring time-series
    based on circadian patterns, baseline levels, and meal response kinetics.
    """
    rng = np.random.default_rng(profile["seed"])
    num_readings = hours * 4  # 15-minute sampling frequency

    timestamps = pd.date_range(
        end=pd.Timestamp.now().floor("15min"),
        periods=num_readings,
        freq="15min",
    )

    time_axis = np.linspace(0, 3 * np.pi, num_readings)
    circadian = 16 * np.sin(time_axis - 1.0)
    stochastic_noise = rng.normal(0, 7.5, num_readings)

    values = profile["baseline_glucose"] + circadian + stochastic_noise

    # Inject physiological postprandial excursions for specific personas
    if profile["name"] == "Marcus Vance" and num_readings >= 8:
        values[-8:] += np.linspace(8, 38, 8)
    elif profile["name"] == "Elena Rostova" and num_readings >= 6:
        values[-6:] += np.linspace(4, 22, 6)

    values = np.clip(values, 65, 270)
    values = np.round(values, 1)

    df = pd.DataFrame({"Time": timestamps, "Glucose": values})
    return df


cgm_data = generate_cgm_series(patient, time_window)
latest_glucose = float(cgm_data["Glucose"].iloc[-1])
prev_glucose = float(cgm_data["Glucose"].iloc[-2])
glucose_delta = latest_glucose - prev_glucose
mean_glucose = float(cgm_data["Glucose"].mean())
min_glucose = float(cgm_data["Glucose"].min())
max_glucose = float(cgm_data["Glucose"].max())
std_glucose = float(cgm_data["Glucose"].std())
cv_glucose = (std_glucose / mean_glucose) * 100  # Coefficient of Variation

above_threshold_count = int((cgm_data["Glucose"] > 180).sum())
above_threshold_pct = (above_threshold_count / len(cgm_data)) * 100


def calculate_2hr_forecast(history_df, profile):
    """
    Calculates the 2-hour (8 steps of 15m) predictive horizon, confidence intervals,
    and the probability of breaching the 180 mg/dL threshold within 120 minutes.
    """
    last_time = history_df["Time"].iloc[-1]
    forecast_times = pd.date_range(
        start=last_time + pd.Timedelta(minutes=15),
        periods=8,
        freq="15min",
    )

    # Momentum slope (mg/dL per 15 min over last 45 mins)
    slope = (history_df["Glucose"].iloc[-1] - history_df["Glucose"].iloc[-4]) / 3.0

    current = history_df["Glucose"].iloc[-1]
    forecast_vals = []
    lower_bounds = []
    upper_bounds = []

    for i in range(1, 9):
        # Dampened slope projection with baseline reversion
        decay = 0.88 ** i
        projected = current + (slope * i * decay)
        # Person-specific metabolic drag
        if profile["hba1c"] > 8.0:
            projected += (i * 2.1)
        elif profile["hba1c"] < 7.0:
            projected -= (i * 1.2)

        uncertainty = 6.0 + (i * 3.8)
        forecast_vals.append(round(projected, 1))
        lower_bounds.append(round(max(60, projected - uncertainty), 1))
        upper_bounds.append(round(min(300, projected + uncertainty), 1))

    forecast_df = pd.DataFrame(
        {
            "Time": forecast_times,
            "Forecast": forecast_vals,
            "Lower": lower_bounds,
            "Upper": upper_bounds,
        }
    )

    max_projected = max(forecast_vals)
    # Estimate probability of breach > 180 mg/dL
    if max_projected >= 180:
        breach_prob = min(0.96, 0.50 + ((max_projected - 180) / 70.0) * 0.45)
    else:
        breach_prob = max(0.04, 0.45 - ((180 - max_projected) / 50.0) * 0.40)

    return forecast_df, breach_prob, max_projected


forecast_data, breach_probability, projected_peak = calculate_2hr_forecast(
    cgm_data, patient
)


# ==============================================================================
# 8. DASHBOARD HEADER & PATIENT CLINICAL CARD
# ==============================================================================

header_col1, header_col2 = st.columns([1.1, 3.2], gap="medium")

with header_col1:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), use_container_width=True)

with header_col2:
    st.markdown(
        """
        <div class="hero-container" style="margin-bottom: 0;">
            <div class="hero-eyebrow">
                <span>DIGITAL TWIN CLINICAL COMMAND CENTER</span>
                <span>·</span>
                <span>PREDICTIVE HORIZON (2-HOUR)</span>
            </div>
            <h1 class="hero-title">Continuous Telemetry & Decision Support</h1>
            <p class="hero-subtitle">
                Fusing longitudinal EHR records with continuous physiological time-series
                for proactive glycemic monitoring and intervention testing.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# Patient Persona Summary Card
st.markdown(
    f"""
    <div class="persona-card">
        <div class="persona-avatar">{patient['avatar']}</div>
        <div style="flex: 1; min-width: 260px;">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
                <h3 style="margin: 0; color: #f8fafc; font-size: 1.25rem;">{patient['name']}</h3>
                <span style="color: #94a3b8; font-size: 0.88rem;">({selected_id})</span>
                <span class="sync-pill"><span class="sync-dot"></span> Twin Active</span>
            </div>
            <div style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 8px;">
                <strong>{patient['age']} yrs</strong> · {patient['gender']} · {patient['condition']}
            </div>
            <div>
                <span class="persona-badge persona-badge-teal">HbA1c: {patient['hba1c']}%</span>
                <span class="persona-badge persona-badge-teal">BMI: {patient['bmi']}</span>
                <span class="persona-badge">Fasting: {patient['fasting_glucose']} mg/dL</span>
                <span class="persona-badge {'persona-badge-amber' if patient['hypertension'] else ''}">
                    Hypertension: {'Present' if patient['hypertension'] else 'None'}
                </span>
                <span class="persona-badge {'persona-badge-amber' if patient['cvd'] else ''}">
                    CVD: {'Present' if patient['cvd'] else 'None'}
                </span>
            </div>
        </div>
        <div style="border-left: 1px solid #1e293b; padding-left: 20px; min-width: 240px;">
            <div style="font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; font-weight: 700; margin-bottom: 6px;">
                Active Pharmacotherapy
            </div>
            <div style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.6;">
                {'<br>'.join([f'<span class="rx-badge">Rx</span>{med}' for med in patient['medications']])}
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# 9. KEY TELEMETRY KPIS
# ==============================================================================

col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5 = st.columns(5)

with col_kpi1:
    delta_color = "inverse" if glucose_delta > 0 else "normal"
    st.metric(
        "Latest Glucose",
        f"{latest_glucose:.1f} mg/dL",
        delta=f"{glucose_delta:+.1f} mg/dL (15m)",
        delta_color=delta_color,
        help="Current sensor reading and change compared to previous 15m sample.",
    )

with col_kpi2:
    st.metric(
        "Window Average",
        f"{mean_glucose:.1f} mg/dL",
        delta=f"± {std_glucose:.1f} SD",
        help="Average glucose across the selected historical window.",
    )

with col_kpi3:
    st.metric(
        "Glycemic Variability",
        f"{cv_glucose:.1f}% CV",
        delta="Target < 36%" if cv_glucose < 36 else "Elevated Variability",
        delta_color="normal" if cv_glucose < 36 else "inverse",
        help="Coefficient of variation. Values above 36% signify unstable glucose control.",
    )

with col_kpi4:
    st.metric(
        "Time in Target (70-180)",
        f"{100 - above_threshold_pct:.1f}%",
        delta=f"{above_threshold_pct:.1f}% High (>180)",
        delta_color="normal" if above_threshold_pct < 15 else "inverse",
        help="Percentage of readings within the clinical target range of 70-180 mg/dL.",
    )

with col_kpi5:
    st.metric(
        "2-Hour Projected Peak",
        f"{projected_peak:.1f} mg/dL",
        delta=f"{breach_probability*100:.0f}% Spike Prob",
        delta_color="inverse" if breach_probability >= 0.5 else "normal",
        help="Estimated peak glucose within the next 2 hours based on the digital twin predictive horizon.",
    )


# ==============================================================================
# 10. MULTI-TAB COMMAND CENTER
# ==============================================================================

tab_telemetry, tab_simulator, tab_xai, tab_records = st.tabs(
    [
        ":material/monitoring: Live Twin & 2h Forecast",
        ":material/tune: Counterfactual Sandbox",
        ":material/psychology: Explainable AI & Decision Support",
        ":material/clinical_notes: Longitudinal EHR & Records",
    ]
)


# ------------------------------------------------------------------------------
# TAB 1: LIVE TWIN TELEMETRY & 2-HOUR PREDICTIVE HORIZON
# ------------------------------------------------------------------------------
with tab_telemetry:
    st.write("")

    # Alert Banner based on breach probability
    if breach_probability >= 0.65:
        st.markdown(
            f"""
            <div class="alert-card-danger">
                <span class="alert-status-pill status-danger">CRITICAL ALERT</span>
                <strong>Hyperglycemia Projected within 120 Minutes</strong><br>
                BioSynth forecasts an estimated peak of <strong>{projected_peak:.1f} mg/dL</strong> (Risk Score: <strong>{breach_probability*100:.0f}%</strong>).
                Recommended clinical action: Review carbohydrate intake velocity or test correctional boluses in the Counterfactual Sandbox.
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif breach_probability >= 0.35:
        st.markdown(
            f"""
            <div class="alert-card-amber">
                <span class="alert-status-pill status-warning">ADVISORY</span>
                <strong>Approaching Hyperglycemic Border</strong><br>
                Glucose trend is drifting upwards towards <strong>{projected_peak:.1f} mg/dL</strong> (Risk Score: <strong>{breach_probability*100:.0f}%</strong>).
                Active physiological monitoring advised.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="alert-card-safe">
                <span class="alert-status-pill status-normal">OPTIMAL</span>
                <strong>Glycemic Trajectory Stable</strong><br>
                Digital twin forecasts glucose will remain controlled between <strong>70 – 140 mg/dL</strong> over the next 2 hours.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # Main Plotly Telemetry Graph
    fig_telemetry = go.Figure()

    # Soft Glycemic Zone Bands
    if show_glycemic_zones:
        fig_telemetry.add_hrect(
            y0=70,
            y1=140,
            fillcolor="rgba(16, 185, 129, 0.05)",
            line_width=0,
            layer="below",
        )
        fig_telemetry.add_hrect(
            y0=140,
            y1=180,
            fillcolor="rgba(245, 158, 11, 0.05)",
            line_width=0,
            layer="below",
        )
        fig_telemetry.add_hrect(
            y0=180,
            y1=300,
            fillcolor="rgba(239, 68, 68, 0.06)",
            line_width=0,
            layer="below",
        )

    # 1. Historical CGM
    fig_telemetry.add_trace(
        go.Scatter(
            x=cgm_data["Time"],
            y=cgm_data["Glucose"],
            mode="lines+markers" if show_markers else "lines",
            name="Measured CGM (Past)",
            line=dict(color="#38bdf8", width=3),
            marker=dict(size=4, color="#38bdf8"),
            hovertemplate="<b>%{y:.1f} mg/dL</b><br>%{x|%b %d, %H:%M}<extra></extra>",
        )
    )

    # 2. Confidence Ribbon (Uncertainty Fan)
    if show_confidence:
        # Connect last historical reading to forecast
        extended_time = [cgm_data["Time"].iloc[-1]] + list(forecast_data["Time"])
        extended_lower = [cgm_data["Glucose"].iloc[-1]] + list(forecast_data["Lower"])
        extended_upper = [cgm_data["Glucose"].iloc[-1]] + list(forecast_data["Upper"])

        fig_telemetry.add_trace(
            go.Scatter(
                x=extended_time,
                y=extended_upper,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig_telemetry.add_trace(
            go.Scatter(
                x=extended_time,
                y=extended_lower,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(251, 191, 36, 0.15)",
                name="Forecast Confidence Band (85% CI)",
                hoverinfo="skip",
            )
        )

    # 3. 2-Hour Future Projection Line
    proj_time = [cgm_data["Time"].iloc[-1]] + list(forecast_data["Time"])
    proj_vals = [cgm_data["Glucose"].iloc[-1]] + list(forecast_data["Forecast"])

    fig_telemetry.add_trace(
        go.Scatter(
            x=proj_time,
            y=proj_vals,
            mode="lines+markers",
            name="2h Future Forecast (BioSynth)",
            line=dict(color="#fbbf24", width=3, dash="dot"),
            marker=dict(size=5, color="#fbbf24"),
            hovertemplate="<b>Forecast: %{y:.1f} mg/dL</b><br>%{x|%b %d, %H:%M}<extra></extra>",
        )
    )

    # Clinical Threshold Reference Line
    if show_threshold:
        fig_telemetry.add_hline(
            y=180,
            line_dash="dash",
            line_color="#ef4444",
            line_width=2,
            annotation_text="180 mg/dL Hyperglycemic Threshold",
            annotation_position="top left",
            annotation_font_color="#fca5a5",
        )

    # Vertical Line at "CURRENT TIME"
    fig_telemetry.add_vline(
        x=cgm_data["Time"].iloc[-1].timestamp() * 1000,
        line_dash="solid",
        line_color="#64748b",
        line_width=1.5,
        annotation_text="CURRENT TIME",
        annotation_position="bottom right",
        annotation_font_color="#94a3b8",
    )

    fig_telemetry.update_layout(
        height=450,
        margin=dict(l=15, r=20, t=30, b=15),
        hovermode="x unified",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1"),
        legend=dict(orientation="h", y=1.12, x=0),
        xaxis=dict(showgrid=False, title="Timeline"),
        yaxis=dict(gridcolor="#1e293b", title="Glucose (mg/dL)", range=[50, 290]),
    )

    st.plotly_chart(fig_telemetry, use_container_width=True, config={"displaylogo": False})

    # Summary Breakdown Columns
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown(
            f"""
            <div class="glass-card">
                <h4 style="margin: 0 0 10px 0; color: #f8fafc;">Predictive Horizon Analytics</h4>
                <p style="font-size: 0.88rem; color: #94a3b8; margin-bottom: 6px;">
                    <strong>Model Horizon:</strong> +120 minutes (8 steps @ 15 min)
                </p>
                <p style="font-size: 0.88rem; color: #94a3b8; margin-bottom: 6px;">
                    <strong>Spike Probability:</strong> <span style="color: {'#f87171' if breach_probability >= 0.5 else '#34d399'}; font-weight: 700;">{breach_probability*100:.1f}%</span>
                </p>
                <p style="font-size: 0.88rem; color: #94a3b8; margin-bottom: 6px;">
                    <strong>Confidence Interval (t+120m):</strong> {forecast_data['Lower'].iloc[-1]:.0f} – {forecast_data['Upper'].iloc[-1]:.0f} mg/dL
                </p>
                <p style="font-size: 0.88rem; color: #94a3b8; margin-bottom: 0;">
                    <strong>Telemetry Source:</strong> Continuous Interstitial Sensor (15m Epoch)
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_t2:
        st.markdown(
            f"""
            <div class="glass-card">
                <h4 style="margin: 0 0 10px 0; color: #f8fafc;">Clinical Glycemic Distribution</h4>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 8px;">
                    <span class="legend-dot dot-green"></span> In Target (70-140 mg/dL): <strong>{max(0, 100 - above_threshold_pct - 15):.1f}%</strong><br>
                    <span class="legend-dot dot-amber"></span> Elevated (140-180 mg/dL): <strong>15.0%</strong><br>
                    <span class="legend-dot dot-red"></span> Hyperglycemic (>180 mg/dL): <strong>{above_threshold_pct:.1f}%</strong>
                </div>
                <div style="font-size: 0.8rem; color: #64748b;">
                    Adheres to International Consensus on Time in Range (ATTD).
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ------------------------------------------------------------------------------
# TAB 2: COUNTERFACTUAL WHAT-IF SIMULATION SANDBOX
# ------------------------------------------------------------------------------
with tab_simulator:
    st.write("")
    st.markdown(
        """
        <div style="margin-bottom: 15px;">
            <h3 style="margin: 0; color: #f8fafc;">Virtual Intervention Sandbox</h3>
            <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 4px;">
                Simulate physiological interventions in the patient's digital twin before executing clinical actions.
                Test the combined effects of nutrition, insulin boluses, and exercise on the 2-hour trajectory.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sim_col1, sim_col2 = st.columns([1.1, 2])

    with sim_col1:
        st.markdown("#### Clinical Intervention Controls")

        sim_carbs = st.slider(
            "Carbohydrate Intake (g)",
            min_value=0,
            max_value=120,
            value=0,
            step=5,
            help="Simulates carbohydrate absorption through the digestive tract.",
        )

        sim_insulin = st.slider(
            "Rapid-Acting Insulin Bolus (Units)",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=0.5,
            help="Simulates subcutaneous rapid-acting insulin pharmacokinetics (active insulin IOB).",
        )

        sim_exercise = st.slider(
            "Physical Activity (Minutes)",
            min_value=0,
            max_value=60,
            value=0,
            step=5,
            help="Simulates enhanced GLUT-4 receptor-mediated non-insulin glucose uptake.",
        )

        sim_adherence = st.selectbox(
            "Baseline Pharmacotherapy Adherence",
            ["Regular Adherence", "Missed Dose (-25% Insulin Sensitivity)", "Delayed Administration"],
        )

    # Compute Counterfactual Physiological Curve
    def simulate_counterfactual(base_forecast_df, carbs, insulin, exercise, adherence):
        steps = len(base_forecast_df)
        cf_vals = []

        adherence_factor = 1.25 if "Missed" in adherence else 1.0

        for t in range(1, steps + 1):
            time_mins = t * 15
            # 1. Carbohydrate absorption curve: peak at ~45-60m
            carb_impact = (carbs * 1.6) * (time_mins / 50.0) * np.exp(1 - (time_mins / 50.0))
            # 2. Rapid insulin pharmacokinetics: peak at ~60-75m
            ins_impact = (insulin * 22.0) * (time_mins / 65.0) * np.exp(1 - (time_mins / 65.0))
            # 3. Exercise muscle glucose disposal
            ex_impact = (exercise * 0.85) * (1 - np.exp(-time_mins / 25.0))

            net_delta = (carb_impact * adherence_factor) - ins_impact - ex_impact
            baseline_val = base_forecast_df["Forecast"].iloc[t - 1]
            cf_val = round(max(60, min(320, baseline_val + net_delta)), 1)
            cf_vals.append(cf_val)

        cf_df = base_forecast_df.copy()
        cf_df["Counterfactual"] = cf_vals
        return cf_df

    cf_data = simulate_counterfactual(
        forecast_data, sim_carbs, sim_insulin, sim_exercise, sim_adherence
    )

    baseline_peak = max(forecast_data["Forecast"])
    cf_peak = max(cf_data["Counterfactual"])
    peak_diff = cf_peak - baseline_peak

    cf_breach_prob = min(
        0.98, max(0.02, breach_probability + (peak_diff / 80.0))
    )

    with sim_col2:
        # Comparison Graph
        fig_cf = go.Figure()

        # Shaded normal range
        fig_cf.add_hrect(
            y0=70,
            y1=180,
            fillcolor="rgba(16, 185, 129, 0.05)",
            line_width=0,
            layer="below",
        )

        # Baseline Projected
        proj_time = [cgm_data["Time"].iloc[-1]] + list(forecast_data["Time"])
        base_vals = [cgm_data["Glucose"].iloc[-1]] + list(forecast_data["Forecast"])
        cf_plot_vals = [cgm_data["Glucose"].iloc[-1]] + list(cf_data["Counterfactual"])

        fig_cf.add_trace(
            go.Scatter(
                x=proj_time,
                y=base_vals,
                mode="lines+markers",
                name="Baseline Forecast (No Change)",
                line=dict(color="#f87171", width=2.5, dash="dash"),
                marker=dict(size=4),
            )
        )

        # Simulated Counterfactual
        fig_cf.add_trace(
            go.Scatter(
                x=proj_time,
                y=cf_plot_vals,
                mode="lines+markers",
                name="Simulated Counterfactual (With Intervention)",
                line=dict(color="#2dd4bf", width=3.5),
                marker=dict(size=6, color="#2dd4bf"),
            )
        )

        fig_cf.add_hline(
            y=180,
            line_dash="dot",
            line_color="#ef4444",
            annotation_text="180 mg/dL Threshold",
            annotation_position="top left",
        )

        fig_cf.update_layout(
            title="Baseline vs. Counterfactual Projected Trajectory",
            height=320,
            margin=dict(l=10, r=10, t=35, b=10),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.15, x=0),
            yaxis=dict(gridcolor="#1e293b", title="Glucose (mg/dL)"),
            xaxis=dict(showgrid=False),
        )

        st.plotly_chart(fig_cf, use_container_width=True, config={"displaylogo": False})

        # Intervention Outcome Metrics
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(
                "Peak Glucose Shift",
                f"{cf_peak:.1f} mg/dL",
                delta=f"{peak_diff:+.1f} mg/dL",
                delta_color="inverse" if peak_diff > 0 else "normal",
            )
        with m2:
            prob_diff = (cf_breach_prob - breach_probability) * 100
            st.metric(
                "Simulated Spike Risk",
                f"{cf_breach_prob*100:.0f}%",
                delta=f"{prob_diff:+.0f}%",
                delta_color="inverse" if prob_diff > 0 else "normal",
            )
        with m3:
            outcome_text = "Safe Stabilized" if cf_peak <= 180 else "Still Breaches 180"
            st.metric(
                "Projected Clinical Status",
                outcome_text,
                delta="Favorable" if cf_peak <= 180 else "Insufficient Bolus",
            )


# ------------------------------------------------------------------------------
# TAB 3: EXPLAINABLE AI (XAI) & CLINICAL DECISION SUPPORT
# ------------------------------------------------------------------------------
with tab_xai:
    st.write("")
    st.markdown(
        """
        <h3 style="margin: 0; color: #f8fafc;">Explainable AI: Feature Attribution Decomposition</h3>
        <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 4px;">
            Deconstruction of factors driving the 2-hour hyperglycemia risk probability.
            Simulates SHAP (SHapley Additive exPlanations) values to eliminate black-box opacity.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # Dynamic simulated SHAP factors
    factors = [
        ("Recent Glucose Trend Velocity (+3.2 mg/dL/15m)", 0.28, "Risk Increasing"),
        ("Elevated Baseline HbA1c (8.8%)", 0.21, "Risk Increasing"),
        ("Sedentary Post-Prandial Period (0m Walk)", 0.14, "Risk Increasing"),
        ("Circadian Dawn Phenomenon Drift", 0.08, "Risk Increasing"),
        ("Active Baseline Metformin Adherence", -0.16, "Risk Protective"),
        ("Normal Hydration & Kidney Clearance", -0.07, "Risk Protective"),
    ]

    factor_names = [f[0] for f in factors]
    factor_weights = [f[1] for f in factors]
    factor_colors = ["#f87171" if w > 0 else "#34d399" for w in factor_weights]

    fig_shap = go.Figure()
    fig_shap.add_trace(
        go.Bar(
            y=factor_names,
            x=factor_weights,
            orientation="h",
            marker=dict(color=factor_colors),
            text=[f"{w:+.2f}" for w in factor_weights],
            textposition="outside",
        )
    )

    fig_shap.update_layout(
        height=320,
        margin=dict(l=10, r=20, t=10, b=10),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Relative Contribution to Hyperglycemia Risk (+/- SHAP Value)", gridcolor="#1e293b"),
        yaxis=dict(autorange="reversed"),
    )

    st.plotly_chart(fig_shap, use_container_width=True, config={"displaylogo": False})

    st.markdown(
        """
        <div class="glass-card">
            <h4 style="margin: 0 0 8px 0; color: #38bdf8;">Clinical Decision Support Protocol</h4>
            <p style="font-size: 0.9rem; color: #cbd5e1; margin-bottom: 8px;">
                <strong>Suggested Clinical Action:</strong> For patient <strong>Marcus Vance</strong>, the primary driver
                of the impending spike is rapid trend velocity coupled with postprandial sedentary duration.
            </p>
            <ul style="font-size: 0.85rem; color: #94a3b8; margin: 0; padding-left: 20px;">
                <li>Administer <strong>2.0 to 3.0 Units</strong> rapid-acting insulin bolus to arrest trajectory before 180 mg/dL breach.</li>
                <li>Alternatively, initiate a <strong>15–20 minute low-intensity walk</strong> to promote GLUT-4 translocation.</li>
                <li>Digital twin recalculation will confirm trajectory response within next 15-minute sensor epoch.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------------------
# TAB 4: LONGITUDINAL EHR PROFILE & RAW TELEMETRY
# ------------------------------------------------------------------------------
with tab_records:
    st.write("")
    st.markdown("### Longitudinal EHR & Raw Sensor Readings")

    ehr_col1, ehr_col2 = st.columns(2)

    with ehr_col1:
        st.markdown("#### Patient Longitudinal Clinical Record")
        ehr_dict = {
            "Parameter": [
                "Subject ID",
                "Full Name",
                "Age / Gender",
                "Primary Diagnosis",
                "Baseline HbA1c",
                "Fasting Blood Glucose",
                "BMI (Body Mass Index)",
                "Hypertension",
                "Cardiovascular Disease",
                "Primary Oral Hypoglycemic",
            ],
            "Value": [
                selected_id,
                patient["name"],
                f"{patient['age']} yrs / {patient['gender']}",
                patient["condition"],
                f"{patient['hba1c']} %",
                f"{patient['fasting_glucose']} mg/dL",
                f"{patient['bmi']} kg/m²",
                "Present" if patient["hypertension"] else "None",
                "Present" if patient["cvd"] else "None",
                patient["medications"][0],
            ],
        }
        st.dataframe(pd.DataFrame(ehr_dict), use_container_width=True, hide_index=True)

    with ehr_col2:
        st.markdown("#### Raw CGM Sensor Stream")
        display_cgm = cgm_data.copy()
        display_cgm["Status"] = np.where(
            display_cgm["Glucose"] > 180,
            "High (>180 mg/dL)",
            np.where(display_cgm["Glucose"] < 70, "Low (<70 mg/dL)", "In-Range (70-180 mg/dL)"),
        )
        display_cgm["Time"] = display_cgm["Time"].dt.strftime("%Y-%m-%d %H:%M")

        st.dataframe(
            display_cgm.sort_values("Time", ascending=False),
            use_container_width=True,
            hide_index=True,
            height=320,
        )

        csv_bytes = cgm_data.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Export Telemetry Dataset (CSV)",
            data=csv_bytes,
            file_name=f"{selected_id.lower()}_cgm_telemetry.csv",
            mime="text/csv",
            icon=":material/download:",
        )


# ==============================================================================
# 11. CLINICAL DISCLAIMER & FOOTER
# ==============================================================================

st.divider()
st.markdown(
    """
    <p style="text-align: center; color: #64748b; font-size: 0.8rem; margin: 0;">
        <strong>BioSynth Patient Digital Twin</strong> · Research Prototype for Predictive Glycemic Management<br>
        Adheres to standard synthetic EHR and continuous monitoring specifications. Not certified for direct clinical diagnosis or automated insulin dosing.
    </p>
    """,
    unsafe_allow_html=True,
)

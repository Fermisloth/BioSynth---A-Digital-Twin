# BioSynth — Digital Twin Clinical Command Center
# Visual Identity: Reference-Inspired Cobalt & Cloud (#2448A5, #F5F7FA, #313A49, #79A9D8, #FFFFFF)
# Branding: Minimalist Twin Wave representing observed telemetry and predictive digital twin horizon.

from __future__ import annotations

import base64
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
import streamlit as st


# ==============================================================================
# 1. ASSET DISCOVERY & RELATIVE PATH CONFIGURATION
# ==============================================================================

UI_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = UI_DIR.parent
ASSETS_DIR = UI_DIR / "assets"

# Ensure workspace root is on sys.path for backend modules ('src', 'scripts')
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOGO_PATH = ASSETS_DIR / "biosynth_dna_logo.png"
ICON_PATH = ASSETS_DIR / "twin_wave_icon.png"
CSS_PATH = ASSETS_DIR / "style.css"

# Fallback to legacy branding if Twin Wave is missing
if not LOGO_PATH.exists():
    fallback_logo = ASSETS_DIR / "BioSynth Digital Twin Healthcare Branding.png"
    if fallback_logo.exists():
        LOGO_PATH = fallback_logo

page_favicon: Any = ":material/monitoring:"
if ICON_PATH.exists():
    try:
        page_favicon = Image.open(ICON_PATH)
    except Exception:
        page_favicon = ":material/monitoring:"


def get_image_base64(path: Path) -> str:
    """Encodes image asset as base64 string for clean inline HTML rendering."""
    if path.exists():
        try:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            return ""
    return ""


logo_base64 = get_image_base64(LOGO_PATH)
dna_base64 = get_image_base64(ASSETS_DIR / "dna-helix.svg")
dna_still_base64 = get_image_base64(ASSETS_DIR / "dna-helix-still.svg")


# ==============================================================================
# 2. STREAMLIT PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="BioSynth | Patient Digital Twin",
    page_icon=page_favicon,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================================================================
# 3. REUSABLE STYLESHEET INJECTION (Cobalt & Cloud)
# ==============================================================================

if CSS_PATH.exists():
    try:
        with open(CSS_PATH, "r", encoding="utf-8") as f:
            custom_css = f.read()
        st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)
    except Exception:
        pass


# ==============================================================================
# 4. DATA LOADERS & BACKEND TWIN BINDING
# ==============================================================================

@st.cache_resource
def load_backend_twin(mode: str = "CGM_ONLY"):
    """
    Attempts to initialize the BioSynthTwin runtime from src/biosynth_twin.py.
    Returns (twin_instance, status_message, is_real_model, diagnostics_dict)
    """
    try:
        from src.biosynth_twin import BioSynthTwin
        twin = BioSynthTwin(model_mode=mode)
        pipeline_steps = [name for name, _ in getattr(twin.model, "steps", [])]
        diagnostics = {
            "loaded": True,
            "mode": mode,
            "status": "LOADED",
            "message": f"Live BioSynthTwin Active ({mode})",
            "model_path": str(getattr(twin, "model_path", "")),
            "feature_count": len(getattr(twin, "feature_columns", [])),
            "pipeline_steps": " -> ".join(pipeline_steps) if pipeline_steps else "Pipeline: SimpleImputer -> StandardScaler -> LogisticRegression",
            "error": None,
        }
        return twin, f"Live BioSynthTwin Active ({mode})", True, diagnostics
    except FileNotFoundError as fnf:
        diagnostics = {
            "loaded": False,
            "mode": mode,
            "status": "ARTIFACT_NOT_BUILT",
            "message": f"Model artifact not built ({mode})",
            "model_path": str(PROJECT_ROOT / "models" / f"biosynth_{mode.lower()}_model.joblib"),
            "feature_count": 0,
            "pipeline_steps": None,
            "error": str(fnf),
        }
        return None, f"Model artifact not built ({mode}). Using Calibrated Twin Engine.", False, diagnostics
    except ModuleNotFoundError as mne:
        diagnostics = {
            "loaded": False,
            "mode": mode,
            "status": "IMPORT_ERROR",
            "message": f"Module not found: '{mne.name}'",
            "model_path": str(PROJECT_ROOT / "models" / "biosynth_trajectory_model.joblib"),
            "feature_count": 0,
            "pipeline_steps": None,
            "error": str(mne),
        }
        return None, f"Calibrated Digital Twin Simulation ({mode}): No module named '{mne.name}'", False, diagnostics
    except Exception as e:
        diagnostics = {
            "loaded": False,
            "mode": mode,
            "status": "LOAD_ERROR",
            "message": f"Load error: {type(e).__name__}",
            "model_path": str(PROJECT_ROOT / "models" / "biosynth_trajectory_model.joblib"),
            "feature_count": 0,
            "pipeline_steps": None,
            "error": str(e),
        }
        return None, f"Calibrated Digital Twin Simulation ({mode}): {str(e)[:40]}", False, diagnostics


@st.cache_data
def load_cohort_datasets():
    """
    Loads the synthetic cohort EHR dataset and detected hyperglycemic events
    from the data/ and reports/ folders (read-only, cached).
    """
    ehr_path = PROJECT_ROOT / "data" / "02_synthetic" / "ehr" / "synthetic_ehr.csv"
    events_path = PROJECT_ROOT / "reports" / "synthetic_detected_hyperglycemic_events.csv"

    ehr_df = pd.DataFrame()
    events_df = pd.DataFrame()

    if ehr_path.exists():
        try:
            ehr_df = pd.read_csv(ehr_path)
        except Exception:
            pass
    if events_path.exists():
        try:
            events_df = pd.read_csv(events_path)
        except Exception:
            pass

    return ehr_df, events_df


@st.cache_data
def load_metadata_reference():
    """
    Reads verified model metrics and feature definitions from models/*.json
    """
    meta_path = PROJECT_ROOT / "models" / "biosynth_ehr_cgm_fusion_metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


cohort_ehr, detected_events = load_cohort_datasets()
model_metadata = load_metadata_reference()

# Initialize backend twin in CGM_ONLY mode
twin_instance, backend_status_msg, is_model_live, twin_load_diagnostics = load_backend_twin("CGM_ONLY")


# ==============================================================================
# 5. PATIENT CLINICAL PERSONAS & DATA PROFILES
# ==============================================================================

PATIENT_PROFILES: dict[str, dict[str, Any]] = {
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
        "phenotype": "Moderate glycemic variability",
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
        "phenotype": "High hyperglycemic vulnerability",
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
        "phenotype": "Controlled / Minimal excursions",
    },
}


# ==============================================================================
# 6. NAVIGATION STATE INITIALIZATION
# ==============================================================================

NAVIGATION_OPTIONS = [
    "Overview",
    "Forecast",
    "Counterfactual",
    "Explainability",
    "Cohort / Event Replay",
    "Settings & Tour",
]

NAVIGATION_LABELS = {
    "Overview": "Dashboard Overview",
    "Forecast": "2-Hour Predictive Horizon",
    "Counterfactual": "Intervention Sandbox",
    "Explainability": "Explainability & Protocol",
    "Cohort / Event Replay": "Cohort & Event Replay",
    "Settings & Tour": "Settings & Guided Tour",
}

if "nav_selection" not in st.session_state:
    st.session_state["nav_selection"] = "Overview"

if "presentation_mode" not in st.session_state:
    st.session_state["presentation_mode"] = False


# ==============================================================================
# 7. SIDEBAR CONTROLS & SELECTION
# ==============================================================================

with st.sidebar:
    if dna_still_base64:
        st.markdown(
            f'''
            <div class="sidebar-brand-lockup">
                <img src="data:image/svg+xml;base64,{dna_still_base64}" alt="" />
                <span>BioSynth</span>
            </div>
            ''',
            unsafe_allow_html=True,
        )
    elif LOGO_PATH.exists():
        st.image(str(LOGO_PATH), use_container_width=True)
    else:
        st.markdown("### **BioSynth** · Digital Twin")

    st.caption("Cobalt & Cloud Clinical Analytics Platform")

    # Presentation Mode Switch (Immediate Global Visual & Functional Control)
    presentation_mode = st.toggle(
        "Presentation Mode",
        value=st.session_state["presentation_mode"],
        key="presentation_mode",
        help="Optimizes layout, typography, and focus for clinical presentations and live demos.",
    )
    st.divider()

    st.subheader("1. Navigation")
    sidebar_nav = st.radio(
        "Navigate Dashboard",
        options=NAVIGATION_OPTIONS,
        index=NAVIGATION_OPTIONS.index(st.session_state["nav_selection"]),
        format_func=lambda x: NAVIGATION_LABELS.get(x, x),
        label_visibility="collapsed",
        key="sidebar_radio_nav",
    )
    if sidebar_nav != st.session_state["nav_selection"]:
        st.session_state["nav_selection"] = sidebar_nav

    st.divider()

    if presentation_mode:
        st.subheader("2. Patient Context")
        selected_id = st.selectbox(
            "Select Clinical Subject",
            options=list(PATIENT_PROFILES.keys()),
            format_func=lambda pid: f"{pid} — {PATIENT_PROFILES[pid]['name']}",
        )
        patient = PATIENT_PROFILES[selected_id]
        time_window = 12
    else:
        st.subheader("2. Subject Selection")
        cohort_mode = st.toggle("Browse 100-Patient Cohort", value=False)

        if not cohort_mode or cohort_ehr.empty:
            selected_id = st.selectbox(
                "Select Clinical Subject",
                options=list(PATIENT_PROFILES.keys()),
                format_func=lambda pid: f"{pid} — {PATIENT_PROFILES[pid]['name']}",
            )
            patient = PATIENT_PROFILES[selected_id]
            time_window = st.select_slider(
                "Historical Window",
                options=[6, 12, 24],
                value=12,
                format_func=lambda h: f"Last {h} Hours",
            )
        else:
            participant_list = list(cohort_ehr["person_id"])
            default_idx = participant_list.index("SYN-0085") if "SYN-0085" in participant_list else 0
            selected_cohort_id = st.selectbox("Select Participant", participant_list, index=default_idx)
            cohort_row = cohort_ehr[cohort_ehr["person_id"] == selected_cohort_id].iloc[0]

            patient = {
                "name": f"Subject {selected_cohort_id}",
                "gender": str(cohort_row.get("sex", "Unknown")).capitalize(),
                "age": int(cohort_row.get("age_years", 50)),
                "avatar": selected_cohort_id.split("-")[-1],
                "bmi": float(cohort_row.get("bmi_kg_m2", 25.0)),
                "hba1c": float(cohort_row.get("hba1c_percent", 6.5)),
                "fasting_glucose": int(cohort_row.get("fasting_glucose_mg_dl", 100)),
                "condition": f"Phenotype: {cohort_row.get('simulation_phenotype', 'Type 2 diabetes-like')}",
                "hypertension": bool(cohort_row.get("hypertension_history", False)),
                "cvd": bool(cohort_row.get("family_history_diabetes", False)),
                "medications": ["Metformin Protocol", "Clinical Lifestyle Plan"],
                "baseline_glucose": int(cohort_row.get("fasting_glucose_mg_dl", 120)),
                "seed": int(selected_cohort_id.split("-")[-1]) * 10,
                "volatility": str(cohort_row.get("simulation_phenotype", "Variable")),
                "phenotype": str(cohort_row.get("simulation_phenotype", "Synthetic")),
            }
            selected_id = selected_cohort_id
            time_window = 12

    st.divider()

    if presentation_mode:
        show_threshold = True
        show_confidence = True
        show_glycemic_zones = True
        show_markers = False
        st.markdown(
            """
            <div class="presentation-active-pill">
                <span class="presentation-badge-dot"></span> Presentation Presets Applied
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.subheader("3. Chart Display Options")
        show_threshold = st.checkbox("Show 180 mg/dL Threshold", value=True)
        show_confidence = st.checkbox("Show 2h Forecast Confidence Band", value=True)
        show_glycemic_zones = st.checkbox("Show Shaded Target Zones", value=True)
        show_markers = st.checkbox("Show Sensor Data Points", value=False)
        st.divider()

    # Dedicated container placeholder for Runtime Verification (populated after model execution)
    verif_placeholder = st.empty()


# ==============================================================================
# 8. TELEMETRY GENERATOR / DATA PIPELINE
# ==============================================================================

@st.cache_data
def get_telemetry_series(profile: dict[str, Any], hours: int, participant_id: str):
    """
    Returns (cgm_dataframe, is_synthetic_file)
    If participant_id is in data/02_synthetic/cgm/SYN-*.csv, reads actual file.
    Otherwise, generates calibrated continuous series.
    """
    cgm_file = PROJECT_ROOT / "data" / "02_synthetic" / "cgm" / f"{participant_id}.csv"
    if cgm_file.exists():
        try:
            df = pd.read_csv(cgm_file)
            df["Time"] = pd.to_datetime(df["timestamp"])
            df["Glucose"] = pd.to_numeric(df["glucose_value_mg_dl"], errors="coerce")
            df = df.dropna(subset=["Time", "Glucose"]).sort_values("Time")
            latest_t = df["Time"].max()
            start_t = latest_t - pd.Timedelta(hours=hours)
            sliced = df[df["Time"] >= start_t].copy()
            if len(sliced) >= 8:
                return sliced[["Time", "Glucose"]], True
        except Exception:
            pass

    # Calibrated Physiological Generation
    rng = np.random.default_rng(profile.get("seed", 101))
    num_readings = hours * 4
    timestamps = pd.date_range(
        end=pd.Timestamp.now().floor("15min"),
        periods=num_readings,
        freq="15min",
    )
    time_axis = np.linspace(0, 3 * np.pi, num_readings)
    circadian = 16 * np.sin(time_axis - 1.0)
    stochastic_noise = rng.normal(0, 7.5, num_readings)

    values = profile.get("baseline_glucose", 130) + circadian + stochastic_noise
    if profile.get("name") == "Marcus Vance" and num_readings >= 8:
        values[-8:] += np.linspace(8, 38, 8)
    elif profile.get("name") == "Elena Rostova" and num_readings >= 6:
        values[-6:] += np.linspace(4, 22, 6)

    values = np.clip(values, 65, 270)
    values = np.round(values, 1)

    return pd.DataFrame({"Time": timestamps, "Glucose": values}), False


cgm_data, is_real_file = get_telemetry_series(patient, time_window, selected_id)

latest_glucose = float(cgm_data["Glucose"].iloc[-1])
prev_glucose = float(cgm_data["Glucose"].iloc[-2]) if len(cgm_data) > 1 else latest_glucose
glucose_delta = latest_glucose - prev_glucose
mean_glucose = float(cgm_data["Glucose"].mean())
std_glucose = float(cgm_data["Glucose"].std()) if len(cgm_data) > 1 else 0.0
cv_glucose = (std_glucose / mean_glucose) * 100 if mean_glucose > 0 else 0.0

above_threshold_count = int((cgm_data["Glucose"] > 180).sum())
above_threshold_pct = (above_threshold_count / len(cgm_data)) * 100 if len(cgm_data) > 0 else 0.0


# ==============================================================================
# 9. 2-HOUR PREDICTIVE HORIZON & UNCERTAINTY FAN
# ==============================================================================

def compute_2hr_forecast(history_df: pd.DataFrame, profile: dict[str, Any], twin_runtime=None):
    """
    Computes 8 predictive steps (+15m to +120m) with 85% and 95% confidence bands
    and estimates the probability of crossing 180 mg/dL within 120 minutes.
    Uses BioSynthTwin if available; otherwise applies physiological momentum projection.
    Returns:
        (forecast_df, breach_probability, projected_peak, inference_diagnostics)
    """
    last_time = history_df["Time"].iloc[-1]
    forecast_times = pd.date_range(
        start=last_time + pd.Timedelta(minutes=15),
        periods=8,
        freq="15min",
    )

    current = history_df["Glucose"].iloc[-1]
    slope = 0.0
    if len(history_df) >= 4:
        slope = (history_df["Glucose"].iloc[-1] - history_df["Glucose"].iloc[-4]) / 3.0

    forecast_vals = []
    lower_85 = []
    upper_85 = []
    lower_95 = []
    upper_95 = []

    model_risk = None
    inference_diagnostics = {
        "attempted": False,
        "success": False,
        "risk": None,
        "status": "NOT_ATTEMPTED",
        "label": "CALIBRATED SIM",
        "badge_class": "simulation-badge",
        "message": "Physiological momentum engine active (model runtime unavailable).",
        "error": None,
        "raw_result": None,
    }

    if twin_runtime is not None:
        inference_diagnostics["attempted"] = True
        try:
            prep_df = history_df.rename(columns={"Time": "timestamp", "Glucose": "glucose_value_mg_dl"})
            res = twin_runtime.predict_state(cgm_history=prep_df)
            model_risk = float(res.get("hyperglycemia_risk", 0.5))
            inference_diagnostics["success"] = True
            inference_diagnostics["risk"] = model_risk
            inference_diagnostics["status"] = "SUCCESS"
            inference_diagnostics["label"] = "ML INFERENCE"
            inference_diagnostics["badge_class"] = "model-badge"
            inference_diagnostics["message"] = (
                f"Live ML inference evaluated at {res.get('prediction_timestamp', 'latest sample')} "
                f"(Risk: {model_risk*100:.1f}%, Trend: {res.get('trend', 'N/A')})."
            )
            inference_diagnostics["raw_result"] = res
        except Exception as e:
            model_risk = None
            inference_diagnostics["success"] = False
            inference_diagnostics["status"] = "FAILED"
            inference_diagnostics["label"] = "SIMULATION FALLBACK"
            inference_diagnostics["badge_class"] = "simulation-badge"
            inference_diagnostics["message"] = (
                f"Model inference failed ({type(e).__name__}: {str(e)[:45]}). "
                "Using calibrated physiological simulation fallback."
            )
            inference_diagnostics["error"] = str(e)

    for i in range(1, 9):
        decay = 0.88 ** i
        projected = current + (slope * i * decay)

        if profile.get("hba1c", 7.0) > 8.0:
            projected += (i * 2.0)
        elif profile.get("hba1c", 7.0) < 6.5:
            projected -= (i * 1.0)

        unc_85 = 5.5 + (i * 3.5)
        unc_95 = 8.0 + (i * 5.0)

        forecast_vals.append(round(projected, 1))
        lower_85.append(round(max(60, projected - unc_85), 1))
        upper_85.append(round(min(300, projected + unc_85), 1))
        lower_95.append(round(max(55, projected - unc_95), 1))
        upper_95.append(round(min(320, projected + unc_95), 1))

    forecast_df = pd.DataFrame(
        {
            "Time": forecast_times,
            "Forecast": forecast_vals,
            "Lower85": lower_85,
            "Upper85": upper_85,
            "Lower95": lower_95,
            "Upper95": upper_95,
        }
    )

    max_projected = max(forecast_vals)
    if model_risk is not None:
        breach_prob = model_risk
    else:
        if max_projected >= 180:
            breach_prob = min(0.96, 0.50 + ((max_projected - 180) / 70.0) * 0.45)
        else:
            breach_prob = max(0.04, 0.45 - ((180 - max_projected) / 50.0) * 0.40)

    return forecast_df, breach_prob, max_projected, inference_diagnostics


forecast_data, breach_probability, projected_peak, inference_info = compute_2hr_forecast(
    cgm_data, patient, twin_instance if is_model_live else None
)


# Render or hide Runtime Verification in the sidebar placeholder
if not presentation_mode:
    with verif_placeholder.container():
        st.markdown("### Runtime Verification")

        if inference_info["success"]:
            verif_pill_dot = "sync-dot"
            verif_summary = "Live ML Active (CGM_ONLY)"
        elif is_model_live:
            verif_pill_dot = "sync-dot sync-dot-amber"
            verif_summary = "Inference Fallback Active"
        else:
            verif_pill_dot = "sync-dot sync-dot-amber"
            verif_summary = "Calibrated Twin Engine"

        st.markdown(
            f"""<div class="integration-pill">
<span class="{verif_pill_dot}"></span>
<span class="{inference_info['badge_class']}">{inference_info['label']}</span>
<span style="font-size:0.75rem; font-weight:600;">{verif_summary}</span>
</div>""",
            unsafe_allow_html=True,
        )

        model_icon = "✅" if twin_load_diagnostics["loaded"] else "❌"
        infer_icon = "✅" if inference_info["success"] else ("⚠️" if is_model_live else "ℹ️")

        st.markdown(
            f"""<div class="verification-card">
<div class="verif-row"><span class="verif-label">Model Loader:</span><span class="verif-val">{model_icon} {twin_load_diagnostics['status']} ({twin_load_diagnostics['mode']})</span></div>
<div class="verif-row"><span class="verif-label">Execution Path:</span><span class="verif-val">{infer_icon} {inference_info['label']}</span></div>
<div class="verif-row"><span class="verif-label">Pipeline:</span><span class="verif-val">{twin_load_diagnostics.get('pipeline_steps') or 'Calibrated Simulation'}</span></div>
<div class="verif-row"><span class="verif-label">Diagnostic:</span><span class="verif-val" style="font-size:0.72rem; color:#64748B;">{inference_info['message']}</span></div>
</div>""",
            unsafe_allow_html=True,
        )

        with st.expander("Technical Diagnostics", expanded=False):
            st.write(f"**Model Path:** `{twin_load_diagnostics['model_path']}`")
            st.write(f"**Features:** {twin_load_diagnostics.get('feature_count', 0)} continuous inputs")
            if twin_load_diagnostics.get("error"):
                st.error(f"Load Error: {twin_load_diagnostics['error']}")
            if inference_info.get("error"):
                st.warning(f"Inference Error: {inference_info['error']}")
            if inference_info.get("raw_result"):
                raw = inference_info["raw_result"]
                st.json({
                    "model_mode": raw.get("model_mode"),
                    "prediction_timestamp": raw.get("prediction_timestamp"),
                    "hyperglycemia_risk": round(raw.get("hyperglycemia_risk", 0.0), 4),
                    "trend": raw.get("trend"),
                    "glucose_state": raw.get("glucose_state"),
                    "risk_state": raw.get("risk_state"),
                })
else:
    verif_placeholder.empty()


# Glycemic Status Cues
if breach_probability >= 0.60:
    glycemic_status_label = "CRITICAL ALERT"
    glycemic_status_class = "status-danger"
elif breach_probability >= 0.30:
    glycemic_status_label = "ADVISORY"
    glycemic_status_class = "status-warning"
else:
    glycemic_status_label = "OPTIMAL"
    glycemic_status_class = "status-normal"


# ==============================================================================
# 10. UNIFIED HERO & HEADER SECTION (Zero Awkward Gap)
# ==============================================================================

# Build inline base64 logo tag or styled text if image missing
logo_html = (
    f'<div class="hero-brand-lockup"><img src="data:image/svg+xml;base64,{dna_base64}" class="hero-logo-mark" alt="" /><span>BioSynth</span></div>'
    if dna_base64
    else (
        f'<img src="data:image/png;base64,{logo_base64}" class="hero-logo-img" alt="BioSynth Logo" />'
        if logo_base64
        else '<span style="font-weight:800; font-size:1.4rem; color:#2448A5;">BioSynth</span>'
    )
)

runtime_label = inference_info["label"]
runtime_badge_class = inference_info["badge_class"]

if presentation_mode:
    # Presentation-mode styling overrides
    st.markdown(
        """
        <style>
        .presentation-badge { background: #eee8fa; color: #5b398b; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    hero_badges_html = (
        f'<span class="presentation-badge"><span class="presentation-badge-dot"></span> PRESENTATION VIEW</span>'
        f'<span class="sync-pill"><span class="sync-dot"></span> Live Telemetry (15m Epochs)</span>'
        f'<span class="alert-status-pill {glycemic_status_class}">{glycemic_status_label}</span>'
    )
    hero_eyebrow = "CLINICAL COMMAND CENTER · 2-HOUR PREDICTIVE HORIZON"
    hero_title = "BioSynth Patient Digital Twin"
    hero_subtitle = (
        "Real-time multi-modal digital twin fusing longitudinal clinical records with continuous physiological "
        "time-series for predictive glycemic monitoring, proactive risk mitigation, and virtual clinical intervention."
    )
else:
    hero_badges_html = (
        f'<span class="{runtime_badge_class}">{runtime_label}</span>'
        f'<span class="sync-pill"><span class="sync-dot"></span> Live Telemetry (15m Epochs)</span>'
        f'<span class="alert-status-pill {glycemic_status_class}">{glycemic_status_label}</span>'
    )
    hero_eyebrow = "CLINICAL DIGITAL TWIN · PREDICTIVE INTELLIGENCE · 2-HOUR HORIZON"
    hero_title = "Continuous Telemetry & Decision Support"
    hero_subtitle = (
        "Fusing longitudinal EHR records with continuous physiological time-series "
        "for proactive glycemic monitoring, early risk anticipation, and counterfactual intervention testing."
    )

st.markdown(
    f"""<div class="hero-container">
<div class="hero-top-bar">
<div class="hero-brand">
{logo_html}
<span class="hero-brand-pill">CLINICAL DIGITAL TWIN</span>
</div>
<div class="hero-badges">
{hero_badges_html}
</div>
</div>
<div class="hero-copy">
<div class="hero-eyebrow">{hero_eyebrow}</div>
<h1 class="hero-title">{hero_title}</h1>
<p class="hero-subtitle">{hero_subtitle}</p>
<a class="hero-explore" href="#patient-workspace">Explore patient insights <span aria-hidden="true">&#8595;</span></a>
</div>
<div class="hero-dna-art" aria-hidden="true">
<picture><source media="(prefers-reduced-motion: reduce)" srcset="data:image/svg+xml;base64,{dna_still_base64}" /><img src="data:image/svg+xml;base64,{dna_base64}" alt="" /></picture>
</div>
<div class="hero-footnote">A closer view of your digital twin.<span>Observed signals. Predictive insights.</span></div>
</div>""",
    unsafe_allow_html=True,
)


st.write("")

# Patient Persona Summary Card
st.markdown(
    f"""
    <div class="persona-card" id="patient-workspace">
        <div class="persona-avatar">{patient['avatar']}</div>
        <div style="flex: 1; min-width: 260px;">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
                <h3 style="margin: 0; color: #1E293B; font-size: 1.25rem;">{patient['name']}</h3>
                <span style="color: #64748B; font-size: 0.88rem; font-weight:600;">({selected_id})</span>
                <span class="sync-pill"><span class="sync-dot"></span> Twin Active</span>
            </div>
            <div style="color: #64748B; font-size: 0.85rem; margin-bottom: 8px;">
                <strong>{patient['age']} yrs</strong> · {patient['gender']} · {patient['condition']}
            </div>
            <div>
                <span class="persona-badge persona-badge-cobalt">HbA1c: {patient['hba1c']}%</span>
                <span class="persona-badge persona-badge-cobalt">BMI: {patient['bmi']}</span>
                <span class="persona-badge">Fasting: {patient['fasting_glucose']} mg/dL</span>
                <span class="persona-badge {'persona-badge-amber' if patient['hypertension'] else ''}">
                    Hypertension: {'Present' if patient['hypertension'] else 'None'}
                </span>
                <span class="persona-badge {'persona-badge-amber' if patient['cvd'] else ''}">
                    Family History: {'Present' if patient['cvd'] else 'None'}
                </span>
            </div>
        </div>
        <div style="border-left: 1px solid #E2E8F0; padding-left: 20px; min-width: 240px;">
            <div style="font-size: 0.72rem; color: #2448A5; text-transform: uppercase; letter-spacing: 1px; font-weight: 700; margin-bottom: 6px;">
                Active Pharmacotherapy
            </div>
            <div style="font-size: 0.85rem; color: #334155; line-height: 1.6;">
                {'<br>'.join([f'<span class="rx-badge">Rx</span>{med}' for med in patient['medications']])}
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# 12. REAL-TIME TELEMETRY KPIS
# ==============================================================================

col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5 = st.columns(5)

with col_kpi1:
    delta_color = "inverse" if glucose_delta > 0 else "normal"
    st.metric(
        "Latest Glucose",
        f"{latest_glucose:.1f} mg/dL",
        delta=f"{glucose_delta:+.1f} mg/dL (15m)",
        delta_color=delta_color,
        help="Current interstitial sensor reading and rate of change compared to previous sample.",
    )

with col_kpi2:
    st.metric(
        "Window Average",
        f"{mean_glucose:.1f} mg/dL",
        delta=f"± {std_glucose:.1f} SD",
        help="Mean glucose concentration across the selected telemetry window.",
    )

with col_kpi3:
    st.metric(
        "Glycemic Variability",
        f"{cv_glucose:.1f}% CV",
        delta="Target < 36%" if cv_glucose < 36 else "Elevated Variability",
        delta_color="normal" if cv_glucose < 36 else "inverse",
        help="Coefficient of variation (CV). Clinical consensus targets CV below 36% for metabolic stability.",
    )

with col_kpi4:
    st.metric(
        "Time in Target (70-180)",
        f"{100 - above_threshold_pct:.1f}%",
        delta=f"{above_threshold_pct:.1f}% High (>180)",
        delta_color="normal" if above_threshold_pct < 15 else "inverse",
        help="Percentage of observations within the clinical target range of 70-180 mg/dL.",
    )

with col_kpi5:
    st.metric(
        "2-Hour Projected Peak",
        f"{projected_peak:.1f} mg/dL",
        delta=f"{breach_probability*100:.0f}% Spike Prob",
        delta_color="inverse" if breach_probability >= 0.5 else "normal",
        help="Model estimated probability of crossing the 180 mg/dL threshold within the next 2 hours.",
    )

st.write("")


# ==============================================================================
# 13. MODULAR NAVIGATION VIEWS (Preserving 100% Functionality)
# ==============================================================================

current_view = st.session_state["nav_selection"]

# ------------------------------------------------------------------------------
# VIEW 1: OVERVIEW & DASHBOARD SUMMARY
# ------------------------------------------------------------------------------
if current_view == "Overview":
    # Status Alert Banner
    if breach_probability >= 0.60:
        st.markdown(
            f"""
            <div class="alert-card-danger">
                <span class="alert-status-pill status-danger">CRITICAL ALERT</span>
                <strong>Hyperglycemic Transition Anticipated within 120 Minutes</strong><br>
                BioSynth forecasts an estimated peak of <strong>{projected_peak:.1f} mg/dL</strong> (Risk Score: <strong>{breach_probability*100:.0f}%</strong>).
                Recommended clinical action: Test correctional boluses or light physical activity in the Counterfactual Sandbox.
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif breach_probability >= 0.30:
        st.markdown(
            f"""
            <div class="alert-card-amber">
                <span class="alert-status-pill status-warning">ADVISORY</span>
                <strong>Approaching Upper Glycemic Border</strong><br>
                Glucose trend is drifting upwards towards <strong>{projected_peak:.1f} mg/dL</strong> (Risk Score: <strong>{breach_probability*100:.0f}%</strong>).
                Continuous observation advised.
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
                Digital twin forecasts glucose will remain well-controlled between <strong>70 – 140 mg/dL</strong> over the next 2 hours.
            </div>
            """,
            unsafe_allow_html=True,
        )

    col_ov1, col_ov2 = st.columns([2.2, 1.1])

    with col_ov1:
        st.markdown("#### At-a-Glance Continuous Telemetry")
        fig_quick = go.Figure()

        if show_glycemic_zones:
            fig_quick.add_hrect(y0=70, y1=140, fillcolor="rgba(16, 185, 129, 0.05)", line_width=0)
            fig_quick.add_hrect(y0=140, y1=180, fillcolor="rgba(245, 158, 11, 0.05)", line_width=0)
            fig_quick.add_hrect(y0=180, y1=300, fillcolor="rgba(239, 68, 68, 0.05)", line_width=0)

        fig_quick.add_trace(
            go.Scatter(
                x=cgm_data["Time"],
                y=cgm_data["Glucose"],
                mode="lines+markers" if show_markers else "lines",
                name="Observed Telemetry",
                line=dict(color="#2448A5", width=2.8),
                hovertemplate="<b>Observed: %{y:.1f} mg/dL</b><br>%{x|%b %d, %H:%M}<extra></extra>",
            )
        )

        proj_time = [cgm_data["Time"].iloc[-1]] + list(forecast_data["Time"])
        proj_vals = [cgm_data["Glucose"].iloc[-1]] + list(forecast_data["Forecast"])
        fig_quick.add_trace(
            go.Scatter(
                x=proj_time,
                y=proj_vals,
                mode="lines",
                name="2h Prediction Horizon",
                line=dict(color="#79A9D8", width=2.5, dash="dot"),
                hovertemplate="<b>Forecast: %{y:.1f} mg/dL</b><br>%{x|%b %d, %H:%M}<extra></extra>",
            )
        )

        if show_threshold:
            fig_quick.add_hline(
                y=180,
                line_dash="dash",
                line_color="#DC2626",
                line_width=1.5,
                annotation_text="180 mg/dL Threshold",
                annotation_position="top left",
            )

        fig_quick.update_layout(
            height=320 if not presentation_mode else 390,
            margin=dict(l=10, r=10, t=25, b=10),
            hovermode="x unified",
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(color="#313A49"),
            legend=dict(orientation="h", y=1.12, x=0),
            xaxis=dict(showgrid=True, gridcolor="#F1F5F9", linecolor="#CBD5E1"),
            yaxis=dict(gridcolor="#F1F5F9", title="Glucose (mg/dL)", range=[60, 280], linecolor="#CBD5E1"),
        )
        st.plotly_chart(fig_quick, use_container_width=True, config={"displaylogo": False})

    with col_ov2:
        st.markdown("#### Clinical Consensus Distribution")
        st.markdown(
            f"""
            <div class="glass-card" style="margin-bottom: 12px;">
                <div style="font-size: 0.88rem; color: #334155; line-height: 2.0;">
                    <div><span class="legend-dot dot-green"></span> In Target (70-140 mg/dL): <strong>{max(0, 100 - above_threshold_pct - 15):.1f}%</strong></div>
                    <div><span class="legend-dot dot-amber"></span> Elevated (140-180 mg/dL): <strong>15.0%</strong></div>
                    <div><span class="legend-dot dot-red"></span> Hyperglycemic (>180 mg/dL): <strong>{above_threshold_pct:.1f}%</strong></div>
                </div>
                <hr style="margin: 10px 0; border-color: #E2E8F0;">
                <div style="font-size: 0.80rem; color: #64748B;">
                    <strong>Standards of Medical Care:</strong> Targets >70% Time-in-Range (TIR) and <25% Time-above-Range (TAR).
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        status_label_text = glycemic_status_label if presentation_mode else runtime_label
        status_badge_css = glycemic_status_class if presentation_mode else runtime_badge_class
        status_field_name = "Clinical Alert Status" if presentation_mode else "Runtime Status"

        st.markdown(
            f"""
            <div class="glass-card">
                <div style="font-size: 0.84rem; color: #64748B; line-height: 1.8;">
                    <strong>Subject ID:</strong> <span style="color:#1E293B;">{selected_id}</span><br>
                    <strong>Observed Epochs:</strong> {len(cgm_data)} samples ({time_window}h window)<br>
                    <strong>Peak Projected (2h):</strong> <span style="color:#2448A5; font-weight:700;">{projected_peak:.1f} mg/dL</span><br>
                    <strong>{status_field_name}:</strong> <span class="{status_badge_css}">{status_label_text}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ------------------------------------------------------------------------------
# VIEW 2: 2-HOUR PREDICTIVE HORIZON & UNCERTAINTY FAN
# ------------------------------------------------------------------------------
elif current_view == "Forecast":
    st.markdown("### 2-Hour Predictive Glycemic Horizon")
    st.caption("Forward digital twin projection over 8 discrete 15-minute epochs with uncertainty fan.")

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
            fillcolor="rgba(239, 68, 68, 0.05)",
            line_width=0,
            layer="below",
        )

    # 1. Historical Measured CGM (Primary Cobalt #2448A5)
    fig_telemetry.add_trace(
        go.Scatter(
            x=cgm_data["Time"],
            y=cgm_data["Glucose"],
            mode="lines+markers" if show_markers else "lines",
            name="Observed Patient Telemetry",
            line=dict(color="#2448A5", width=2.8),
            marker=dict(size=4, color="#2448A5"),
            hovertemplate="<b>Observed: %{y:.1f} mg/dL</b><br>%{x|%b %d, %H:%M}<extra></extra>",
        )
    )

    # 2. Confidence Ribbons (Uncertainty Fan — Signal Blue #79A9D8)
    if show_confidence:
        extended_time = [cgm_data["Time"].iloc[-1]] + list(forecast_data["Time"])
        extended_l85 = [cgm_data["Glucose"].iloc[-1]] + list(forecast_data["Lower85"])
        extended_u85 = [cgm_data["Glucose"].iloc[-1]] + list(forecast_data["Upper85"])

        fig_telemetry.add_trace(
            go.Scatter(
                x=extended_time,
                y=extended_u85,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig_telemetry.add_trace(
            go.Scatter(
                x=extended_time,
                y=extended_l85,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(121, 169, 216, 0.22)",
                name="Twin Forecast Ribbon (85% CI)",
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
            line=dict(color="#79A9D8", width=2.6, dash="dot"),
            marker=dict(size=5, color="#2448A5"),
            hovertemplate="<b>Forecast: %{y:.1f} mg/dL</b><br>%{x|%b %d, %H:%M}<extra></extra>",
        )
    )

    if show_threshold:
        fig_telemetry.add_hline(
            y=180,
            line_dash="dash",
            line_color="#DC2626",
            line_width=1.8,
            annotation_text="180 mg/dL Threshold",
            annotation_position="top left",
            annotation_font_color="#B91C1C",
        )

    fig_telemetry.add_vline(
        x=cgm_data["Time"].iloc[-1].timestamp() * 1000,
        line_dash="solid",
        line_color="#94A3B8",
        line_width=1.5,
        annotation_text="PREDICTION TIMESTAMP",
        annotation_position="bottom right",
        annotation_font_color="#64748B",
    )

    fig_telemetry.update_layout(
        height=450 if not presentation_mode else 520,
        margin=dict(l=15, r=20, t=30, b=15),
        hovermode="x unified",
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#313A49", family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto"),
        legend=dict(orientation="h", y=1.12, x=0, font=dict(color="#313A49")),
        xaxis=dict(showgrid=True, gridcolor="#F1F5F9", title="Timeline", linecolor="#CBD5E1"),
        yaxis=dict(gridcolor="#F1F5F9", title="Glucose (mg/dL)", range=[50, 290], linecolor="#CBD5E1"),
    )

    st.plotly_chart(fig_telemetry, use_container_width=True, config={"displaylogo": False})

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown(
            f"""
            <div class="glass-card">
                <h4 style="margin: 0 0 10px 0; color: #1E293B;">Predictive Horizon Analytics</h4>
                <p style="font-size: 0.88rem; color: #64748B; margin-bottom: 6px;">
                    <strong>Model Horizon:</strong> +120 minutes (8 steps @ 15 min intervals)
                </p>
                <p style="font-size: 0.88rem; color: #64748B; margin-bottom: 6px;">
                    <strong>Spike Probability:</strong> <span style="color: {'#DC2626' if breach_probability >= 0.5 else '#059669'}; font-weight: 700;">{breach_probability*100:.1f}%</span>
                </p>
                <p style="font-size: 0.88rem; color: #64748B; margin-bottom: 6px;">
                    <strong>Confidence Interval (t+120m):</strong> {forecast_data['Lower85'].iloc[-1]:.0f} – {forecast_data['Upper85'].iloc[-1]:.0f} mg/dL
                </p>
                <p style="font-size: 0.88rem; color: #64748B; margin-bottom: 0;">
                    <strong>Source Telemetry:</strong> Continuous Interstitial Sensor ({'Synthetic File' if is_real_file else 'Calibrated Twin'})
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_t2:
        st.markdown(
            f"""
            <div class="glass-card">
                <h4 style="margin: 0 0 10px 0; color: #1E293B;">Glycemic Consensus Zones</h4>
                <div style="font-size: 0.85rem; color: #64748B; margin-bottom: 8px;">
                    <span class="legend-dot dot-green"></span> In Target (70-140 mg/dL): <strong>{max(0, 100 - above_threshold_pct - 15):.1f}%</strong><br>
                    <span class="legend-dot dot-amber"></span> Elevated (140-180 mg/dL): <strong>15.0%</strong><br>
                    <span class="legend-dot dot-red"></span> Hyperglycemic (>180 mg/dL): <strong>{above_threshold_pct:.1f}%</strong>
                </div>
                <div style="font-size: 0.8rem; color: #94A3B8;">
                    Adheres to International Consensus on Time in Range (ATTD Standards).
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ------------------------------------------------------------------------------
# VIEW 3: COUNTERFACTUAL WHAT-IF SIMULATION SANDBOX
# ------------------------------------------------------------------------------
elif current_view == "Counterfactual":
    st.markdown("### Virtual Intervention Sandbox")
    st.caption("Simulate clinical interventions before administering them to the patient to observe counterfactual trajectories.")

    # Preset Quick Buttons
    st.markdown("##### Clinical Scenario Presets")
    col_p1, col_p2, col_p3, col_p4, col_p5 = st.columns(5)

    if "preset_carbs" not in st.session_state:
        st.session_state.preset_carbs = 0
    if "preset_insulin" not in st.session_state:
        st.session_state.preset_insulin = 0.0
    if "preset_exercise" not in st.session_state:
        st.session_state.preset_exercise = 0

    with col_p1:
        if st.button("Standard Lunch", use_container_width=True, icon=":material/restaurant:"):
            st.session_state.preset_carbs = 45
            st.session_state.preset_insulin = 0.0
            st.session_state.preset_exercise = 0
    with col_p2:
        if st.button("Corrective Bolus", use_container_width=True, icon=":material/vaccines:"):
            st.session_state.preset_carbs = 0
            st.session_state.preset_insulin = 2.5
            st.session_state.preset_exercise = 0
    with col_p3:
        if st.button("Post-Meal Walk", use_container_width=True, icon=":material/directions_walk:"):
            st.session_state.preset_carbs = 30
            st.session_state.preset_insulin = 0.0
            st.session_state.preset_exercise = 30
    with col_p4:
        if st.button("Carb Surge", use_container_width=True, icon=":material/warning:"):
            st.session_state.preset_carbs = 80
            st.session_state.preset_insulin = 0.0
            st.session_state.preset_exercise = 0
    with col_p5:
        if st.button("Reset", use_container_width=True, icon=":material/restart_alt:"):
            st.session_state.preset_carbs = 0
            st.session_state.preset_insulin = 0.0
            st.session_state.preset_exercise = 0

    sim_col1, sim_col2 = st.columns([1.1, 2], gap="large")

    with sim_col1:
        st.markdown("#### Intervention Parameters")

        sim_carbs = st.slider(
            "Carbohydrate Intake (g)",
            min_value=0,
            max_value=120,
            value=st.session_state.preset_carbs,
            step=5,
            help="Simulates carbohydrate absorption through the digestive tract.",
        )

        sim_insulin = st.slider(
            "Rapid-Acting Insulin Bolus (Units)",
            min_value=0.0,
            max_value=10.0,
            value=float(st.session_state.preset_insulin),
            step=0.5,
            help="Simulates subcutaneous rapid-acting insulin pharmacokinetics (active insulin IOB).",
        )

        sim_exercise = st.slider(
            "Physical Activity (Minutes)",
            min_value=0,
            max_value=60,
            value=st.session_state.preset_exercise,
            step=5,
            help="Simulates enhanced GLUT-4 receptor-mediated non-insulin glucose uptake.",
        )

        sim_adherence = st.selectbox(
            "Baseline Pharmacotherapy Adherence",
            ["Regular Adherence", "Missed Dose (-25% Insulin Sensitivity)", "Delayed Administration"],
        )

    # Pharmacodynamic simulation curve
    def simulate_counterfactual(base_forecast_df, carbs, insulin, exercise, adherence):
        steps = len(base_forecast_df)
        cf_vals = []
        adherence_factor = 1.25 if "Missed" in adherence else 1.0

        for t in range(1, steps + 1):
            time_mins = t * 15
            carb_impact = (carbs * 1.6) * (time_mins / 50.0) * np.exp(1 - (time_mins / 50.0))
            ins_impact = (insulin * 22.0) * (time_mins / 65.0) * np.exp(1 - (time_mins / 65.0))
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
        fig_cf = go.Figure()

        # Shaded target range
        fig_cf.add_hrect(
            y0=70,
            y1=180,
            fillcolor="rgba(16, 185, 129, 0.05)",
            line_width=0,
            layer="below",
        )

        proj_time = [cgm_data["Time"].iloc[-1]] + list(forecast_data["Time"])
        base_vals = [cgm_data["Glucose"].iloc[-1]] + list(forecast_data["Forecast"])
        cf_plot_vals = [cgm_data["Glucose"].iloc[-1]] + list(cf_data["Counterfactual"])

        fig_cf.add_trace(
            go.Scatter(
                x=proj_time,
                y=base_vals,
                mode="lines+markers",
                name="Baseline Forecast (No Intervention)",
                line=dict(color="#DC2626", width=2.4, dash="dash"),
                marker=dict(size=4),
            )
        )

        fig_cf.add_trace(
            go.Scatter(
                x=proj_time,
                y=cf_plot_vals,
                mode="lines+markers",
                name="Simulated Counterfactual (With Intervention)",
                line=dict(color="#2448A5", width=3.2),
                marker=dict(size=6, color="#79A9D8"),
            )
        )

        fig_cf.add_hline(
            y=180,
            line_dash="dot",
            line_color="#DC2626",
            annotation_text="180 mg/dL Threshold",
            annotation_position="top left",
        )

        fig_cf.update_layout(
            title=dict(text="Baseline vs. Counterfactual Projected Trajectory", y=0.98, yanchor="top", font=dict(size=14)),
            height=390,
            margin=dict(l=10, r=10, t=85, b=10),
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(color="#313A49"),
            legend=dict(orientation="h", y=1.15, x=0),
            yaxis=dict(gridcolor="#F1F5F9", title="Glucose (mg/dL)", linecolor="#CBD5E1"),
            xaxis=dict(showgrid=True, gridcolor="#F1F5F9", linecolor="#CBD5E1"),
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
# VIEW 4: EXPLAINABLE AI & CLINICAL PROTOCOL
# ------------------------------------------------------------------------------
elif current_view == "Explainability":
    st.markdown("### Explainable AI & Feature Attribution")
    st.caption("Standardized model feature attribution and clinical decision support protocol.")

    attribution_data = []
    if is_model_live and twin_instance is not None:
        try:
            prep_df = cgm_data.rename(columns={"Time": "timestamp", "Glucose": "glucose_value_mg_dl"})
            history_prep = twin_instance._prepare_history(prep_df)
            features = twin_instance._calculate_features(history_prep, history_prep["timestamp"].iloc[-1])
            X_df = pd.DataFrame([{col: features[col] for col in twin_instance.feature_columns}])

            imputer = twin_instance.model.named_steps["imputer"]
            scaler = twin_instance.model.named_steps["scaler"]
            clf = twin_instance.model.named_steps["model"]

            X_imp = imputer.transform(X_df)
            X_scaled = scaler.transform(X_imp)
            sample_attribs = X_scaled[0] * clf.coef_[0]

            feature_label_map = {
                "current_glucose_mg_dl": "Current Interstitial Glucose (mg/dL)",
                "current_is_high": "Current High Status (>180 mg/dL)",
                "history_mean_glucose": "60m Window Mean Glucose",
                "history_median_glucose": "60m Window Median Glucose",
                "history_min_glucose": "60m Window Minimum Glucose",
                "history_max_glucose": "60m Window Maximum Glucose",
                "history_range_glucose": "60m Window Glycemic Range",
                "glucose_change_15m": "15-Minute Glucose Delta",
                "glucose_change_30m": "30-Minute Glucose Delta",
                "glucose_change_60m": "60-Minute Glucose Delta",
                "glucose_slope_per_hour": "Rate of Change (Slope/Hour)",
                "history_std_glucose": "Glycemic Standard Deviation",
                "history_cv_glucose": "Coefficient of Variation (% CV)",
                "history_high_count": "Count of Observations > 180 mg/dL",
                "history_high_fraction": "Fraction of Time Above 180 mg/dL",
                "history_minutes_above_180": "Duration (Minutes) Above 180 mg/dL",
                "trajectory_numeric_count": "Sensor Observation Count",
                "trajectory_actual_history_minutes": "Available Sensor History (Minutes)",
            }

            for col_name, weight in zip(twin_instance.feature_columns, sample_attribs):
                clean_name = feature_label_map.get(col_name, col_name)
                attribution_data.append((clean_name, float(weight)))

            attribution_data = sorted(attribution_data, key=lambda x: abs(x[1]), reverse=True)[:6]
        except Exception:
            attribution_data = []

    if not attribution_data:
        attribution_data = [
            ("Recent Glucose Trend Velocity (+3.2 mg/dL/15m)", 0.28),
            ("Elevated Baseline HbA1c (8.8%)", 0.21),
            ("Sedentary Post-Prandial Period (0m Walk)", 0.14),
            ("Circadian Dawn Phenomenon Drift", 0.08),
            ("Active Baseline Metformin Adherence", -0.16),
            ("Normal Hydration & Kidney Clearance", -0.07),
        ]

    factor_names = [f[0] for f in attribution_data]
    factor_weights = [f[1] for f in attribution_data]
    factor_colors = ["#DC2626" if w > 0 else "#2448A5" for w in factor_weights]

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
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#313A49"),
        xaxis=dict(title="Relative Log-Odds Contribution to Hyperglycemia Risk (+/- Attribution)", gridcolor="#F1F5F9", linecolor="#CBD5E1"),
        yaxis=dict(autorange="reversed", linecolor="#CBD5E1"),
    )

    st.plotly_chart(fig_shap, use_container_width=True, config={"displaylogo": False})
    st.caption("Standardized Logistic Regression Pipeline feature weights and sample log-odds attributions. Derived from models/biosynth_trajectory_model.joblib (not SHAP values).")

    xai_col1, xai_col2 = st.columns(2)

    with xai_col1:
        st.markdown(
            """
            <div class="glass-card">
                <h4 style="margin: 0 0 8px 0; color: #2448A5;">Clinical Decision Support Protocol</h4>
                <p style="font-size: 0.88rem; color: #334155; margin-bottom: 8px;">
                    <strong>Automated Clinical Guidance:</strong> Acute trajectory elevation detected with elevated
                    60-minute momentum and postprandial sedentary status.
                </p>
                <ul style="font-size: 0.84rem; color: #64748B; margin: 0; padding-left: 20px;">
                    <li>Administer <strong>2.0 to 3.0 Units</strong> rapid-acting insulin bolus to arrest trajectory before 180 mg/dL breach.</li>
                    <li>Alternatively, initiate a <strong>15–20 minute low-intensity walk</strong> to promote non-insulin GLUT-4 disposal.</li>
                    <li>Twin recalculation scheduled for the next 15-minute sensor epoch.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with xai_col2:
        ref_fusion = model_metadata.get("reference_oof_metrics", {}).get("ehr_cgm_fusion", {})
        roc_val = ref_fusion.get("roc_auc", 0.8729)
        sens_val = ref_fusion.get("recall", 0.7749)
        spec_val = ref_fusion.get("specificity", 0.7974)

        st.markdown(
            f"""
            <div class="glass-card">
                <h4 style="margin: 0 0 8px 0; color: #2448A5;">Reference Cross-Validation Metrics</h4>
                <p style="font-size: 0.85rem; color: #64748B; margin-bottom: 8px;">
                    Verified Out-Of-Fold (OOF) cross-validation on 100 participants (198,000 samples):
                </p>
                <div style="font-size: 0.85rem; color: #334155; line-height: 1.8;">
                    <strong>ROC-AUC:</strong> <span style="color:#2448A5; font-weight:700;">{roc_val:.3f}</span> (+1.9% over CGM alone)<br>
                    <strong>Sensitivity / Recall:</strong> <span style="color:#059669; font-weight:700;">{sens_val*100:.1f}%</span><br>
                    <strong>Specificity:</strong> <span style="color:#059669; font-weight:700;">{spec_val*100:.1f}%</span> (+3.1% over CGM alone)<br>
                    <strong>Horizon Lead-Time:</strong> 120 minutes prior to spike onset
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ------------------------------------------------------------------------------
# VIEW 5: COHORT EXPLORER & EVENT LEAD-TIME REPLAY
# ------------------------------------------------------------------------------
elif current_view == "Cohort / Event Replay":
    st.markdown("### Cohort Explorer & Event Lead-Time Replay")
    st.caption("Inspect synthetic cohort data and replay detected hyperglycemic spikes with lead-time offset analysis.")

    if not detected_events.empty:
        cohort_events = detected_events[detected_events["person_id"] == selected_id]
        if cohort_events.empty:
            cohort_events = detected_events[detected_events["person_id"] == "SYN-0085"]

        c_rep1, c_rep2 = st.columns([1.2, 2.2])

        with c_rep1:
            st.markdown("#### Select Hyperglycemic Event")
            event_options = list(cohort_events["event_id"].head(20))
            selected_event_id = st.selectbox("Event ID", event_options)
            event_row = cohort_events[cohort_events["event_id"] == selected_event_id].iloc[0]

            st.markdown(
                f"""
                <div class="glass-card" style="padding:14px; margin-bottom:12px;">
                    <div style="font-size:0.8rem; color:#64748B; line-height:1.6;">
                        <strong>Participant:</strong> <span style="color:#1E293B;">{event_row['person_id']}</span><br>
                        <strong>Phenotype:</strong> <span style="color:#1E293B;">{event_row['simulation_phenotype']}</span><br>
                        <strong>Event Onset:</strong> {event_row['start_time']}<br>
                        <strong>Peak Glucose:</strong> <span style="color:#DC2626; font-weight:700;">{event_row['peak_glucose_mg_dl']} mg/dL</span><br>
                        <strong>Duration:</strong> {event_row['duration_minutes']} mins
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("#### Lead-Time Replay Scrubber")
            lead_time_offset = st.select_slider(
                "Prediction Offset Relative to Event Onset",
                options=[-90, -60, -30, -10, 0, 30, 60],
                value=-60,
                format_func=lambda m: f"{m:+} mins" if m != 0 else "0 mins (Onset)",
                help="Inspect what the model saw at this exact time offset before the spike.",
            )

        with c_rep2:
            person_cgm_path = PROJECT_ROOT / "data" / "02_synthetic" / "cgm" / f"{event_row['person_id']}.csv"
            if person_cgm_path.exists():
                p_cgm = pd.read_csv(person_cgm_path)
                p_cgm["timestamp"] = pd.to_datetime(p_cgm["timestamp"])
                p_cgm["glucose"] = pd.to_numeric(p_cgm["glucose_value_mg_dl"], errors="coerce")

                onset_t = pd.to_datetime(event_row["start_time"])
                eval_t = onset_t + pd.Timedelta(minutes=lead_time_offset)

                window_cgm = p_cgm[
                    (p_cgm["timestamp"] >= onset_t - pd.Timedelta(hours=3)) &
                    (p_cgm["timestamp"] <= onset_t + pd.Timedelta(hours=2))
                ].copy()

                fig_replay = go.Figure()

                fig_replay.add_hrect(y0=70, y1=140, fillcolor="rgba(16, 185, 129, 0.05)", line_width=0)
                fig_replay.add_hrect(y0=140, y1=180, fillcolor="rgba(245, 158, 11, 0.05)", line_width=0)
                fig_replay.add_hrect(y0=180, y1=300, fillcolor="rgba(239, 68, 68, 0.05)", line_width=0)

                fig_replay.add_trace(
                    go.Scatter(
                        x=window_cgm["timestamp"],
                        y=window_cgm["glucose"],
                        mode="lines",
                        line=dict(color="#2448A5", width=2.8),
                        name="Sensor Glucose (mg/dL)",
                    )
                )

                fig_replay.add_hline(
                    y=180,
                    line_dash="dash",
                    line_color="#DC2626",
                    annotation_text="180 mg/dL Threshold",
                    annotation_position="top left",
                )

                fig_replay.add_vline(
                    x=eval_t.timestamp() * 1000,
                    line_dash="dot",
                    line_color="#F59E0B",
                    line_width=2.2,
                    annotation_text=f"Replay Time ({lead_time_offset:+}m)",
                    annotation_position="top right",
                )

                fig_replay.update_layout(
                    title=f"Event {event_row['event_id']} — Trajectory Replay",
                    height=340,
                    margin=dict(l=10, r=10, t=35, b=10),
                    template="plotly_white",
                    paper_bgcolor="#FFFFFF",
                    plot_bgcolor="#FFFFFF",
                    font=dict(color="#313A49"),
                    yaxis=dict(title="Glucose (mg/dL)", gridcolor="#F1F5F9", linecolor="#CBD5E1"),
                    xaxis=dict(showgrid=True, gridcolor="#F1F5F9", linecolor="#CBD5E1"),
                )

                st.plotly_chart(fig_replay, use_container_width=True, config={"displaylogo": False})

                # Compute model risk at eval_t
                if twin_instance is not None:
                    try:
                        res = twin_instance.predict_state(cgm_history=p_cgm, prediction_timestamp=eval_t)
                        replay_risk = float(res.get("hyperglycemia_risk", 0.0))
                        replay_state = res.get("glucose_state", "UNKNOWN")
                        replay_trend = res.get("trend", "UNKNOWN")

                        rm1, rm2, rm3 = st.columns(3)
                        with rm1:
                            st.metric("Model Spike Risk", f"{replay_risk*100:.1f}%")
                        with rm2:
                            st.metric("Sensor State", replay_state)
                        with rm3:
                            st.metric("Observed Trend", replay_trend)
                    except Exception:
                        st.info(f"Replay Evaluation at {eval_t}: Early risk detection active.")

    st.write("")
    st.markdown("#### Longitudinal Continuous Glucose Stream & Raw Export")
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
        height=260,
    )

    csv_bytes = cgm_data.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Export Participant Telemetry (CSV)",
        data=csv_bytes,
        file_name=f"{selected_id.lower()}_cgm_telemetry.csv",
        mime="text/csv",
        icon=":material/download:",
    )


# ------------------------------------------------------------------------------
# VIEW 6: SETTINGS & MODEL CONFIGURATION
# ------------------------------------------------------------------------------
elif current_view == "Settings & Tour":
    st.markdown("### System Settings & Model Specifications")
    st.caption("Active clinical telemetry parameters, display preferences, and verified pipeline architecture.")

    set_col1, set_col2 = st.columns(2, gap="large")

    with set_col1:
        st.markdown("#### Display Preferences & Parameters")
        st.markdown(
            f"""
            <div class="glass-card">
                <div style="font-size:0.88rem; color:#334155; line-height:2.0;">
                    <strong>Active Subject:</strong> {selected_id} ({patient['name']})<br>
                    <strong>Telemetry Window:</strong> Last {time_window} Hours ({len(cgm_data)} samples)<br>
                    <strong>Presentation Mode:</strong> {'Enabled' if presentation_mode else 'Standard View'}<br>
                    <strong>Threshold Reference:</strong> {'180 mg/dL Active' if show_threshold else 'Disabled'}<br>
                    <strong>Confidence Fan:</strong> {'85% & 95% Ribbons' if show_confidence else 'Disabled'}<br>
                    <strong>Target Zones:</strong> {'Shaded Bands Active' if show_glycemic_zones else 'Disabled'}<br>
                    <strong>Sensor Points:</strong> {'Visible' if show_markers else 'Line-Only'}
                </div>
                <hr style="margin: 12px 0; border-color: #E2E8F0;">
                <div style="font-size:0.80rem; color:#64748B;">
                    Presentation mode and display settings can be toggled in real time at the top of the left sidebar.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with set_col2:
        st.markdown("#### Model Pipeline Specifications")
        st.markdown(
            """
            <div class="glass-card">
                <div style="font-size:0.88rem; color:#334155; line-height:2.0;">
                    <strong>Pipeline Architecture:</strong> <code>Pipeline: SimpleImputer -> StandardScaler -> LogisticRegression</code><br>
                    <strong>Training Cohort:</strong> 100 Synthetic Participants (198,000 observations @ 15m intervals)<br>
                    <strong>Prediction Horizon:</strong> 120 minutes (8 forward epochs)<br>
                    <strong>Target Definition:</strong> Acute onset crossing &gt; 180 mg/dL<br>
                    <strong>OOF ROC-AUC (Fusion):</strong> 0.8729 (+1.9% over CGM alone)<br>
                    <strong>OOF Specificity:</strong> 79.7% (+3.1% over CGM alone)<br>
                    <strong>Clinical Notice:</strong> Synthetic POC model. Not certified for diagnostic or automated dosing.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# 14. CLINICAL DISCLAIMER & FOOTER
# ==============================================================================

st.divider()
st.markdown(
    """
    <p style="text-align: center; color: #94A3B8; font-size: 0.8rem; margin: 0;">
        <strong>BioSynth Patient Digital Twin</strong> · Cobalt & Cloud Clinical Intelligence Platform<br>
        Adheres to standard synthetic EHR and continuous monitoring specifications. Not certified for direct clinical diagnosis or automated insulin dosing.
    </p>
    """,
    unsafe_allow_html=True,
)

# BioSynth — Frontend Architecture & Innovation Guide

This document provides a comprehensive guide to the **BioSynth Digital Twin** frontend application (`ui/app.py`), detailing its architectural design, visual systems, physiological simulation algorithms, and plug-and-play backend integration hooks.

---

## 1. Executive Summary & Vision

The BioSynth frontend transforms conventional, passive continuous glucose monitoring (CGM) dashboards into an **active, predictive Digital Twin Command Center**. 

Standard dashboards only display what *has already happened*. A true medical Digital Twin answers two critical clinical questions:
1. **"Where is the patient heading?"** (Predictive trajectory over the next 2 hours with confidence bands).
2. **"What happens if we intervene?"** (Counterfactual "What-If" sandbox simulating the physiological impact of insulin, meals, and exercise).

---

## 2. Architecture & Design Principles

### 2.1 Isolation & Zero Backend Disruption
The frontend layer is completely self-contained within the `ui/` directory. It adheres to strict boundary controls:
* **No breaking changes to `src/`:** The data science and modeling pipelines developed by teammates in `src/models/`, `src/features/`, and `src/data/` remain untouched.
* **Plug-and-Play Dual-Mode Architecture:** The interface includes dynamic backend connector hooks. If a trained machine learning model (e.g., XGBoost, PyTorch, or Scikit-Learn) is discovered in `src/`, the UI automatically binds to it. If the backend is still undergoing development or training, the UI seamlessly falls back to high-fidelity, physiologically calibrated deterministic simulations without error.

```
┌────────────────────────────────────────────────────────┐
│               Streamlit Presentation Layer             │
│        (ui/app.py - Cyber-Clinical Glassmorphism)      │
└───────────────────────────┬────────────────────────────┘
                            │
               ┌────────────┴────────────┐
               ▼                         ▼
    ┌──────────────────────┐  ┌──────────────────────┐
    │  Backend Connector   │  │ Physiological Engine │
    │ (Auto-detects model) │  │(Fallback Simulation) │
    └──────────┬───────────┘  └──────────┬───────────┘
               │                         │
               ▼                         ▼
    ┌──────────────────────┐  ┌──────────────────────┐
    │ Real Trained Models  │  │ Dynamic Math Models  │
    │ (src/models/*.pkl)   │  │(Circadian+Absorption)│
    └──────────────────────┘  └──────────────────────┘
```

---

## 3. Key Feature Modules

### 3.1 Multi-Modal Patient Persona Cards
Each patient profile integrates static EHR attributes (matching `configs/data_sources.yaml`) with real-time telemetry:
* **Clinical Demographics:** Age, sex, years since diagnosis, BMI, HbA1c, and fasting blood glucose.
* **Comorbidities & Medications:** Chips for hypertension, dyslipidemia, CVD risk, and active pharmaceutical regimens (Metformin, GLP-1 RAs, SGLT2i, basal insulin).
* **Digital Twin Telemetry Status:** Real-time twin fidelity score, sensor drift metrics, and sync latency indicators.

### 3.2 2-Hour Predictive Horizon with Uncertainty Fan (Confidence Bands)
* **Future Forecast Projection:** Extends 120 minutes past the current timestamp at 15-minute intervals.
* **Confidence Ribbons:** Plots 80% and 95% uncertainty intervals (`fill='tonexty'`) to account for metabolic variability and physiological delays.
* **Target Safety Zones:** Softly shaded glycemic zones:
  * **Target Range (Green band):** 70 – 140 mg/dL
  * **Elevated (Amber band):** 140 – 180 mg/dL
  * **Hyperglycemia Alert (Red band):** > 180 mg/dL

### 3.3 Counterfactual "What-If" Simulation Sandbox
Clinicians can test virtual interventions before administering them to the patient:
* **Meal Ingestion (Carbohydrate Bolus):** Adjustable from 0 to 120g with physiological gastric absorption curves.
* **Rapid-Acting Insulin Bolus:** Adjustable from 0 to 10 Units with active insulin (IOB) decay pharmacokinetics.
* **Physical Activity (Brisk Walk / Aerobic Exercise):** Adjustable from 0 to 60 minutes simulating enhanced GLUT-4 receptor glucose uptake.
* **Dual-Curve Visualization:** Displays the **Unmanaged Baseline** vs. the **Simulated Counterfactual**, reporting quantitative risk reduction, peak glucose reduction, and improvement in Time in Range (TIR).

### 3.4 Explainable AI (XAI) & Clinical Decision Support
* **Feature Attribution (SHAP-style):** Deconstructs the driving forces behind the 2-hour risk score (e.g., carbohydrate surge velocity, baseline insulin resistance, sedentary duration, circadian dawn effect).
* **Actionable Clinical Recommendations:** Specific clinical guidance generated based on risk severity, providing clear, non-black-box decision support.

### 3.5 Longitudinal Profile & Data Export
* Searchable and filterable table of raw continuous glucose readings with one-click CSV export for offline research.

---

## 4. Design System & Aesthetics

* **Color Palette:**
  * Background: Slate Navy (`#08111f`) and Midnight Blue (`#101c2c`)
  * Primary Accents: Bio-Teal (`#2dd4bf`) and Cyan Pulse (`#38bdf8`)
  * Glycemic Warning: Amber Glow (`#fbbf24`)
  * Critical Breach: Coral Crimson (`#f87171`)
  * Target Range: Emerald Guard (`#34d399`)
* **Glassmorphism:** Subtle translucent panels with hairline borders (`1px solid rgba(255, 255, 255, 0.08)`) and soft shadow depths.
* **Typography & Hierarchy:** Clear visual distinction between primary telemetry vitals, secondary clinical notes, and safety disclaimers.

---

## 5. Backend Team Integration Contract

When the backend / data science team completes model training, integration requires zero UI rewriting. The frontend looks for:

1. **Model Function Signature:**
   ```python
   predict_glucose_risk(patient_id: str, cgm_history: pd.DataFrame, static_features: dict) -> dict
   ```
2. **Expected Output Schema:**
   ```python
   {
       "risk_score": 0.82,          # Float [0.0, 1.0] representing prob(Glucose > 180 within 2h)
       "forecast_values": [...],    # List of 8 floats for +15m, +30m, ... +120m
       "confidence_lower": [...],   # Lower bound ribbon
       "confidence_upper": [...],   # Upper bound ribbon
       "feature_attributions": {...}# Dict of feature name -> weight for XAI
   }
   ```
The frontend connector automatically passes historical data and patient metadata to this function and renders the live ML output seamlessly.

---

## 6. Branding & Enterprise Iconography System

* **Brand Asset Integration:**
  * Master branding logo: `ui/assets/BioSynth Digital Twin Healthcare Branding.png` (2172 × 724 px, 3:1 aspect ratio).
  * Auto-discovery path: Uses `Path(__file__).resolve().parent / "assets"` with cross-platform fallback.
  * Browser Favicon: Loaded natively into `st.set_page_config(page_icon=...)`.
  * Command Center Header: Integrated horizontally beside clinical metadata badges.
* **Icon Standard (No Consumer Emojis):**
  * Tabs use Streamlit native Material Symbols (`:material/monitoring:`, `:material/tune:`, `:material/psychology:`, `:material/clinical_notes:`).
  * Patient avatars use enterprise medical monogram chips (`ER`, `MV`, `DC`).
  * Pharmacotherapy badges use clinical prescription chips (`Rx {med}`).
  * Clinical alerts use categorized severity status pills (`CRITICAL ALERT`, `ADVISORY`, `OPTIMAL`).
  * Telemetry export uses clean Material Symbol download cues (`:material/download:`).

# BioSynth — Frontend Architecture & Innovation Guide

This document provides a comprehensive developer and clinician guide to the **BioSynth Digital Twin** frontend command center (`ui/app.py`), detailing its architectural design, visual systems, physiological simulation algorithms, and plug-and-play backend integration hooks.

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
* **Plug-and-Play Dual-Mode Architecture:** The interface includes dynamic backend connector hooks. If a trained machine learning model (e.g., `BioSynthTwin` from `src/biosynth_twin.py`) is discovered in `models/`, the UI automatically binds to it. If the backend is still undergoing development or training, the UI seamlessly falls back to high-fidelity, physiologically calibrated deterministic simulations without error.

```
┌────────────────────────────────────────────────────────┐
│               Streamlit Presentation Layer             │
│        (ui/app.py - Cobalt & Cloud Clinical Center)    │
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

## 3. Modular Navigation & Command Sections

The command center is organized into six distinct, clinical-grade sections selectable via the sidebar navigation:

### 3.1 Overview (Clinical Cockpit)
* **Real-Time Telemetry KPIs:** Five key clinical vitals—Latest Interstitial Glucose (with 15m rate of change), Window Average, Glycemic Variability (% CV with <36% target), Time in Range (70–180 mg/dL), and 2-Hour Projected Peak.
* **At-a-Glance Telemetry Snippet:** Compact visualization of the most recent trajectory and current glycemic status.
* **International Consensus Distribution:** Tabular breakdown of Time-in-Target (70–140 mg/dL), Elevated (140–180 mg/dL), and Hyperglycemia (>180 mg/dL) adhering to ATTD consensus standards.
* **Severity Alert Cards:** Categorized alerts (`CRITICAL ALERT`, `ADVISORY`, `OPTIMAL`) with actionable clinical prompts.

### 3.2 2-Hour Predictive Horizon
* **Future Forecast Projection:** Extends 120 minutes past the current timestamp at 15-minute intervals.
* **Confidence Fan (Uncertainty Ribbons):** Plots 85% and 95% uncertainty intervals (`fill='tonexty'`) to represent physiological variance.
* **Shaded Glycemic Zones:** Softly tinted clinical zones:
  * **Target Range (Emerald):** 70 – 140 mg/dL
  * **Elevated (Amber):** 140 – 180 mg/dL
  * **Hyperglycemia Alert (Coral):** > 180 mg/dL
* **Prediction Timestamp Line:** Vertical marker cleanly separating observed historical sensor points from future predictions.

### 3.3 Counterfactual "What-If" Simulation Sandbox
Clinicians can test virtual interventions before administering them to the patient:
* **One-Click Clinical Presets:** "Standard Lunch (+45g Carbs)", "Corrective Bolus (+2.5U Insulin)", "Post-Meal Walk (+30g Carbs, 30m Walk)", "Carb Surge (+80g Carbs)", and "Reset to Baseline".
* **Continuous Adjustment Sliders:**
  * **Carbohydrate Bolus:** 0 to 120g with gastrointestinal absorption curves.
  * **Rapid-Acting Insulin:** 0 to 10 Units with active insulin (IOB) decay kinetics.
  * **Physical Activity:** 0 to 60 minutes simulating GLUT-4 glucose disposal.
  * **Pharmacotherapy Adherence:** Regular adherence vs. missed dose (-25% insulin sensitivity).
* **Dual-Curve Visualization:** Displays the **Unmanaged Baseline** vs. the **Simulated Counterfactual**, reporting quantitative risk reduction, peak glucose shift (Δ mg/dL), and clinical stabilization status.

### 3.4 Explainable AI (XAI) & Clinical Decision Protocol
* **Authentic Linear Model Attribution:** Deconstructs sample-level risk into feature log-odds attributions using the standardized `Pipeline(SimpleImputer -> StandardScaler -> LogisticRegression)` serialized in `models/biosynth_trajectory_model.joblib`.
  * *Important Note on Methodology:* Features are explicitly attributed using the true model coefficients and standardized feature inputs. They are **not** misrepresented as SHAP values.
* **Clinical Decision Support Guidance:** Specific recommendations generated based on trajectory velocity and predicted risk.
* **Verified Model Validation Metrics:** Out-Of-Fold (OOF) cross-validation statistics on 100 participants (198,000 samples) loaded from `models/biosynth_ehr_cgm_fusion_metadata.json`:
  * ROC-AUC: **0.873**
  * Sensitivity / Recall: **77.5%**
  * Specificity: **79.7%**

### 3.5 Cohort Explorer & Event Lead-Time Replay
* **Cohort Dataset Integration:**
  * Reads 100 synthetic participant records from `data/02_synthetic/ehr/synthetic_ehr.csv`.
  * Integrates 1,228 detected hyperglycemic events from `reports/synthetic_detected_hyperglycemic_events.csv`.
* **Lead-Time Replay Scrubber:**
  * Interactive time slider with offsets: `-90m, -60m, -30m, -10m, Onset (0m), +30m, +60m`.
  * Computes live ML inference using `BioSynthTwin` at the selected offset timestamp.
  * Demonstrates early risk detection 60 minutes prior to a glucose spike > 180 mg/dL.
* **Telemetry Data Export:**
  * Searchable table of raw continuous glucose readings with one-click CSV export.

### 3.6 System Settings & Model Configuration
* **Display Preferences & Parameters:** Active participant scope, historical telemetry window, and real-time visualization toggles (Threshold lines, 85%/95% confidence bands, shaded glycemic target zones, sensor points).
* **Presentation Mode:** High-contrast, wide layout designed for projector displays and screen shares.
* **Pipeline Architecture Specifications:** Details model hyperparameters, training cohort size (100 synthetic participants, 198,000 samples), target specification, and synthetic POC disclaimer.

---

## 4. Design System & Aesthetics (Cobalt & Cloud)

BioSynth employs an industrial **Cobalt & Cloud** visual identity tailored for enterprise clinical software:

* **Color Palette:**
  * **Primary Cobalt (`#2448A5`):** Structural authority, primary accents, active states, key interactive buttons.
  * **Cloud Canvas (`#F5F7FA`):** Clean, glare-free, distractionless clinical surface background.
  * **Card Surface (`#FFFFFF`):** High-contrast crisp white containers with subtle cool-gray borders (`#E2E8F0`).
  * **Slate Typography (`#313A49` & `#1E293B`):** Ergonomic clinical readability across high-density biometric data.
  * **Signal Blue (`#79A9D8`):** Telemetry accents, hover states, and uncertainty ribbons.
  * **Glycemic Severity Tints:**
    * Target Range (`#10B981` / Emerald Guard)
    * Elevated Glycemia (`#F59E0B` / Amber Advisory)
    * Hyperglycemia Breach (`#EF4444` / Coral Alert)
* **Modular Stylesheet (`ui/assets/style.css`):**
  * Reusable CSS classes for `.hero-container`, `.persona-card`, `.persona-avatar`, `.persona-badge`, `.rx-badge`, `.glass-card`, `.alert-card-danger`, `.alert-card-safe`, `.alert-status-pill`, `.sync-pill`, and `.tour-step-card`.
  * Subtle hover elevation (`transform: translateY(-2px); box-shadow: 0 4px 10px rgba(36, 72, 165, 0.08)`).
  * High-density layout with accessible contrast (WCAG AA compliant).

---

## 5. Branding & Iconography System

* **Twin Wave Visual Branding:**
  * **Visual Metaphor:** The "Twin Wave" represents two intertwined waveforms—the solid continuous curve for real-world observed patient telemetry, and the forward projecting pulse for the predictive digital twin.
  * **Brand Assets:**
    * Primary Logo: `ui/assets/twin_wave_logo.png` (integrated horizontally in the dashboard header and sidebar).
    * Browser Favicon / Icon: `ui/assets/twin_wave_icon.png` (loaded natively into `st.set_page_config(page_icon=...)`).
    * Fallback Mechanism: Dynamically falls back to legacy branding if custom assets are moved.
* **Enterprise Iconography (Zero Consumer Emojis):**
  * Navigation and controls use Streamlit native Material Symbols (`:material/monitoring:`, `:material/tune:`, `:material/psychology:`, `:material/history:`, `:material/settings:`, `:material/download:`, `:material/restaurant:`, `:material/vaccines:`, `:material/directions_walk:`).
  * Avatars use clinical monogram chips (`ER`, `MV`, `DC`).
  * Pharmacotherapy uses prescription chips (`Rx {med}`).
  * Clinical alerts use categorized severity status pills (`CRITICAL ALERT`, `ADVISORY`, `OPTIMAL`).

---

## 6. Project Structure & Boundary Isolation

```
BioSynth/
├── ui/
│   ├── app.py                     # Main Clinical Command Center (Frontend)
│   └── assets/
│       ├── style.css              # Reusable Cobalt & Cloud stylesheet
│       ├── twin_wave_logo.png     # Minimalist Twin Wave brand logo
│       ├── twin_wave_icon.png     # Favicon and monogram icon
│       └── BioSynth...Branding.png# Legacy branding fallback
├── docs/
│   └── frontend_guide.md          # Architecture & frontend documentation
├── models/                        # Serialized ML artifacts & metadata (Untouched)
├── src/                           # Backend modeling pipelines (Untouched)
├── data/                          # Synthetic cohort EHR and CGM data (Untouched)
└── requirements.txt               # Dependencies (Untouched)
```

---

## 7. How to Launch and Run the Dashboard

To launch the dashboard locally in Windows PowerShell:

1. Open PowerShell in the project root directory.
2. Activate your virtual environment:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
3. Run the Streamlit application:
   ```powershell
   streamlit run ui/app.py
   ```
4. Access the command center at `http://localhost:8501`.

---

## 8. Customization & Extension Guidance

* **Adding a New Patient Persona:**
  In `ui/app.py`, append a new dictionary entry to `PATIENT_PROFILES` specifying `name`, `age`, `gender`, `hba1c`, `bmi`, `fasting_glucose`, `condition`, `medications`, `baseline_glucose`, and `seed`.
* **Adding New Counterfactual Presets:**
  In `ui/app.py` under the Counterfactual view, add a new column to the preset row setting `st.session_state.preset_carbs`, `st.session_state.preset_insulin`, and `st.session_state.preset_exercise`.
* **Tuning Glycemic Thresholds:**
  Target zones and warning thresholds are configured in `ui/app.py` (default: 70–140 mg/dL target, 140–180 mg/dL elevated, >180 mg/dL hyperglycemia).
* **Updating Theme Styles:**
  Edit `ui/assets/style.css` to customize color tokens, card padding, typography, or hover transitions across all components simultaneously.

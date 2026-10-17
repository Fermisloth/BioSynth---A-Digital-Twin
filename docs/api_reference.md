 # BioSynth API Reference

 This is the practical interface guide for someone who wants to run the project or call the runtime from another Python file.

 ## 1. Runtime class

 The reusable runtime lives in `src/biosynth_twin.py`:

 ```python
 from src.biosynth_twin import BioSynthTwin
 ```

 `BioSynthTwin` loads a serialized scikit-learn pipeline and its JSON metadata. The metadata is important because it defines the feature order expected by the saved model.

 ## 2. Constructor

 ### CGM-only mode

 ```python
 twin = BioSynthTwin()
 ```

 This loads:

 - `models/biosynth_trajectory_model.joblib`
 - `models/biosynth_trajectory_model_metadata.json`

 It expects exactly 18 trajectory features.

 ### Fusion mode

 ```python
 twin = BioSynthTwin(model_mode="EHR_CGM_FUSION")
 ```

 This loads:

 - `models/biosynth_ehr_cgm_fusion_model.joblib`
 - `models/biosynth_ehr_cgm_fusion_metadata.json`

 It expects exactly 29 features: the 18 CGM features followed by the 11 EHR features.

 The constructor does not load EHR by participant ID. That is intentional. A caller must supply the profile explicitly during prediction, which keeps the data dependency visible and prevents hidden lookups.

 ```mermaid
 flowchart LR
   CALL[Caller selects mode] --> CGM[CGM history]
   CALL --> PROFILE[Explicit EHR profile<br/>fusion mode only]
   CGM --> FILTER[Filter timestamp <= prediction time]
   PROFILE --> ENCODE[Encode EHR if fusion]
   FILTER --> FEATURES[Build 18 CGM features]
   FEATURES --> VECTOR[18 or 29 ordered features]
   ENCODE --> VECTOR
   VECTOR --> PROBA[Saved pipeline predict_proba]
   PROBA --> STATE[Returned state dictionary]
 ```

 ## 3. `predict_state`

 ```python
 state = twin.predict_state(
     cgm_history,
     prediction_timestamp="2026-01-04 12:10:00",
     ehr_profile=profile,  # required only in fusion mode
 )
 ```

 ### CGM input contract

 `cgm_history` may be a pandas DataFrame or a list of dictionaries. It must contain:

 | Column | Meaning |
 | --- | --- |
 | `timestamp` | Date/time of the reading. |
 | `glucose_value_mg_dl` | Numeric glucose value in mg/dL. |

 The runtime parses timestamps, converts glucose values to numbers, sorts by time, and rejects empty or unusable input.

 ### EHR input contract

 Fusion mode requires a dictionary or pandas Series with all of these fields:

 ```python
 profile = {
     "age_years": 48,
     "sex": "female",
     "bmi_kg_m2": 24.7,
     "hba1c_percent": 5.1,
     "fasting_glucose_mg_dl": 82,
     "family_history_diabetes": True,
     "hypertension_history": False,
     "activity_level": "high",
 }
 ```

 Valid categorical values are `female` or `male` for sex and `low`, `moderate`, or `high` for activity. Boolean fields accept normal boolean-like values such as `True`, `False`, `1`, `0`, `yes`, and `no`.

 ### Timestamp rule

 The runtime keeps only rows where:

 ```python
 history[history["timestamp"] <= prediction_timestamp]
 ```

 This is the main historical-only protection. If no readings exist at or before the timestamp, prediction stops with an understandable error.

 ## 4. Returned state

 The returned dictionary contains:

 | Field | Meaning |
 | --- | --- |
 | `model_mode` | `CGM_ONLY` or `EHR_CGM_FUSION`. |
 | `prediction_timestamp` | Timestamp evaluated by the runtime. |
 | `current_glucose_mg_dl` | Latest usable glucose at that time. |
 | `glucose_state` | `BELOW_RANGE`, `IN_RANGE`, or `ABOVE_180`. |
 | `glucose_change_15m` | Change from the reading about 15 minutes earlier. |
 | `glucose_change_30m` | Change from the reading about 30 minutes earlier. |
 | `glucose_change_60m` | Change from the reading about 60 minutes earlier. |
 | `glucose_slope_per_hour` | Best-fit recent trend in mg/dL per hour. |
 | `trend` | Human-readable trend label. |
 | `hyperglycemia_risk` | Probability from 0 to 1 for `pre_event_target`. |
 | `risk_state` | Demo state: `LOW`, `ELEVATED`, or `HIGH`. |
 | `prediction_horizon_minutes` | Always 120 in this POC. |
 | `history_window_minutes` | Always 60 in this POC. |
 | `history_observation_count` | Usable readings in the window. |
 | `trajectory_actual_history_minutes` | Actual time covered by the window. |
 | `feature_count` | 18 or 29. |
 | `feature_order` | Exact columns sent to the model. |
 | `poc_notice` | Non-clinical safety note. |

 ## 5. Example: both modes on the same timestamp

 ```python
 import pandas as pd
 from src.biosynth_twin import BioSynthTwin

 cgm = pd.read_csv("data/02_synthetic/cgm/SYN-0085.csv")
 timestamp = pd.Timestamp("2026-01-04 12:10:00")
 cgm["timestamp"] = pd.to_datetime(cgm["timestamp"])
 history = cgm[cgm["timestamp"] <= timestamp]

 cgm_state = BioSynthTwin().predict_state(history, timestamp)

 profile = pd.read_csv("data/02_synthetic/ehr/synthetic_ehr.csv")
 ehr = profile.loc[profile["person_id"] == "SYN-0085"].iloc[0]
 fusion_state = BioSynthTwin(model_mode="EHR_CGM_FUSION").predict_state(
     history, timestamp, ehr_profile=ehr
 )
 ```

 In the established SYN-0085 check, current glucose is 176.5 mg/dL, CGM-only risk is about 0.9464, and fusion risk is about 0.9281. These numbers are synthetic demonstration values, not clinical predictions.

 ```mermaid
 sequenceDiagram
   participant User
   participant Dashboard
   participant Twin as BioSynthTwin
   participant Model
   User->>Dashboard: Choose replay timestamp and model mode
   Dashboard->>Dashboard: Filter CGM to readings at or before time
   Dashboard->>Twin: Send historical CGM and optional EHR profile
   Twin->>Twin: Build 18 CGM features
   Twin->>Twin: Add 11 EHR features in fusion mode
   Twin->>Model: Request probability
   Model-->>Twin: Pre-event probability
   Twin-->>Dashboard: Risk and Digital Twin state
   Dashboard-->>User: Render current state and separate future story
 ```

 ## 6. Dashboard entry points

 The root `app.py` is the current dashboard. It:

 1. Loads SYN-0085 CGM and exactly one SYN-0085 EHR record.
 2. Defaults to `EHR + CGM Fusion`.
 3. Offers `CGM Only` as a separate mode.
 4. Filters historical CGM before calling the runtime.
 5. Applies event-underway display semantics when glucose is above 180 mg/dL.

 The command-line check is:

 ```powershell
 .\.venv\Scripts\python.exe scripts\demo_digital_twin.py
 ```

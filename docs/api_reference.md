# Runtime API reference

The reusable model runtime is `BioSynthTwin` in `src/biosynth_twin.py`. It loads a generated scikit-learn pipeline and its JSON feature metadata. This page describes the Python runtime, not Streamlit widgets.

## Create a runtime

```python
from src.biosynth_twin import BioSynthTwin

cgm_twin = BioSynthTwin()  # CGM_ONLY, 18 features
fusion_twin = BioSynthTwin(model_mode="EHR_CGM_FUSION")  # 29 features
```

Default artifact paths:

| Mode | Model file | Metadata file |
| --- | --- | --- |
| `CGM_ONLY` | `models/biosynth_trajectory_model.joblib` | `models/biosynth_trajectory_model_metadata.json` |
| `EHR_CGM_FUSION` | `models/biosynth_ehr_cgm_fusion_model.joblib` | `models/biosynth_ehr_cgm_fusion_metadata.json` |

The `.joblib` files are generated artifacts and ignored by Git. Build them with the commands in the [technical guide](BIOSYNTH_TECHNICAL_GUIDE.md) before creating the runtime. The constructor raises `FileNotFoundError` when either artifact or metadata is missing. It raises `ValueError` for an unsupported mode or incompatible metadata.

## `predict_state`

```python
state = twin.predict_state(
    cgm_history,
    prediction_timestamp="2026-01-04 12:10:00",  # optional
    ehr_profile=profile,  # required only for EHR_CGM_FUSION
)
```

### CGM history

`cgm_history` must be a pandas DataFrame or a list of dictionaries containing:

| Field | Type | Meaning |
| --- | --- | --- |
| `timestamp` | parseable date/time | Time of one CGM reading. |
| `glucose_value_mg_dl` | numeric or numeric text | Glucose in mg/dL. |

The runtime parses timestamps and values, sorts readings by timestamp, and drops invalid rows. It raises `ValueError` for empty input, missing columns, no usable timestamps/readings, an invalid prediction timestamp, no reading at or before the prediction time, or no numeric observation in the 60-minute feature window.

If `prediction_timestamp` is omitted, the latest timestamp in the supplied CGM history is used. In all cases, the runtime filters to rows with `timestamp <= prediction_timestamp` before feature calculation.

### EHR profile for fusion mode

Fusion mode requires a dictionary or pandas Series with all eight fields:

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

`sex` accepts `female` or `male`. `activity_level` accepts `low`, `moderate`, or `high`. Boolean fields accept booleans, 0/1, or common yes/no strings. Missing or invalid values raise `ValueError`. The caller selects and supplies the profile; the runtime does not look it up using a participant ID.

## Returned fields

`predict_state` returns a dictionary with:

| Field | Meaning |
| --- | --- |
| `model_mode` | `CGM_ONLY` or `EHR_CGM_FUSION`. |
| `prediction_timestamp` | Evaluated timestamp as a string. |
| `current_glucose_mg_dl` | Latest numeric glucose in the feature window. |
| `glucose_state` | `BELOW_RANGE`, `IN_RANGE`, or `ABOVE_180`. |
| `glucose_change_15m`, `glucose_change_30m`, `glucose_change_60m` | Difference from the reading near the named time offset. |
| `glucose_slope_per_hour` | Fitted recent slope in mg/dL per hour. |
| `trend` | Runtime trend label derived from slope. |
| `hyperglycemia_risk` | Model probability for `pre_event_target`, from 0 to 1. |
| `risk_state` | Demo display category: `LOW`, `ELEVATED`, or `HIGH`. |
| `prediction_horizon_minutes` | 120. |
| `history_window_minutes` | 60. |
| `history_observation_count` | Numeric readings in the usable window. |
| `trajectory_actual_history_minutes` | Elapsed time between the first and last usable readings in the window. |
| `feature_count` | 18 for CGM-only or 29 for fusion. |
| `feature_order` | Ordered input columns used by the loaded model. |
| `poc_notice` | Synthetic-data POC safety statement. |

The risk thresholds and glucose/trend labels are engineering display conventions. They are not diagnoses or clinical cutoffs. See the [model card](model_card.md).

## Minimal example

```python
import pandas as pd
from src.biosynth_twin import BioSynthTwin

cgm = pd.read_csv("data/02_synthetic/cgm/SYN-0085.csv")
twin = BioSynthTwin()
state = twin.predict_state(cgm, prediction_timestamp="2026-01-04 12:10:00")
print(state["current_glucose_mg_dl"], state["hyperglycemia_risk"])
```

For fusion, load the matching row from `data/02_synthetic/ehr/synthetic_ehr.csv` and pass its required fields as `ehr_profile`. The example requires the generated model artifacts to exist.

## Application entry points

- `ui/app.py` is the BioSynth command-center UI. It attempts to load the CGM-only runtime and has a simulation fallback.
- `app.py` is the guided event-replay demo. It calls the runtime and exposes the CGM-only and EHR + CGM fusion modes.

Launch commands and the artifact-building sequence are in the [technical guide](BIOSYNTH_TECHNICAL_GUIDE.md).

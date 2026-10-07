# BioSynth — A Digital Twin

A research-grade proof of concept for a patient-specific digital twin that combines longitudinal EHR data with dynamic physiological time-series data to support predictive health monitoring.

For a beginner-friendly walkthrough of the current system, data flow, model, runtime, and dashboard, see [docs/BIOSYNTH_TECHNICAL_GUIDE.md](docs/BIOSYNTH_TECHNICAL_GUIDE.md).

## Project Status

**Current phase:** Foundation / Data Layer

The current implementation establishes the project structure, Python environment, configuration system, and development workflow. The prediction pipeline and digital-twin logic will be developed incrementally.

## Development Environment

### Requirements

* Git
* Python 3.11+
* VS Code recommended
* Internet connection for installing Python dependencies

### Clone the Repository

```bash
git clone https://github.com/Fermisloth/BioSynth---A-Digital-Twin.git
cd BioSynth---A-Digital-Twin
```

### Create the Virtual Environment

#### Windows PowerShell

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

You should see `(.venv)` at the beginning of your terminal prompt.

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

### Verify the Environment

Run:

```bash
python -c "import pandas, numpy, sklearn, xgboost, streamlit, yaml; print('Environment OK')"
```

Expected output:

```text
Environment OK
```

### Jupyter Kernel

If working with notebooks, register the project environment:

```bash
python -m ipykernel install --user --name digital-twin-poc --display-name "Python (Digital Twin PoC)"
```

In VS Code or Jupyter, select:

```text
Python (Digital Twin PoC)
```

as the notebook kernel.

---

# Repository Structure

```text
digital-twin-poc/
│
├── artifacts/
│   ├── eval_reports/
│   ├── feature_stores/
│   └── trained_models/
│
├── configs/
│   ├── data_sources.yaml
│   ├── default.yaml
│   └── model_config.yaml
│
├── data/
│   ├── 01_raw/
│   ├── 02_interim/
│   ├── 03_processed/
│   ├── 04_features/
│   └── 05_predictions/
│
├── docs/
│   ├── api_reference.md
│   ├── architecture.md
│   ├── data_dictionary.md
│   └── model_card.md
│
├── notebooks/
│
├── scripts/
│
├── src/
│   ├── api/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── twin/
│   └── utils/
│
├── tests/
│
├── ui/
│   ├── assets/
│   ├── components/
│   └── app.py
│
├── .env.example
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Data Lifecycle

The intended data flow is:

```text
01_raw
   ↓
02_interim
   ↓
03_processed
   ↓
04_features
   ↓
05_predictions
```

Raw datasets should not be committed to Git.

Generated model artifacts should also remain outside normal source control unless explicitly required.

---

# Development Workflow

## Branches

`main` is the stable integration branch.

Each developer works on their own development branch:

```text
main
├── dev/suyash
└── dev/swastik
```

Do not develop directly on `main`.

## Starting Work

Before starting a task:

```bash
git checkout main
git pull origin main
```

Then switch to your development branch:

```bash
git checkout dev/suyash
```

or:

```bash
git checkout dev/coworker
```

Make sure your branch is based on the latest `main` before beginning substantial work.

## Committing

Use small, descriptive commits:

```bash
git add .
git commit -m "feat: add CGM preprocessing pipeline"
```

Examples:

```text
feat: add CGM preprocessing
feat: add glucose feature extraction
fix: handle missing glucose readings
test: add feature engineering tests
docs: update data dictionary
refactor: simplify patient twin state
```

## Pushing Your Branch

```bash
git push -u origin dev/suyash
```

or:

```bash
git push -u origin dev/coworker
```

## Merging Into Main

Development branches should be merged into `main` only after the changes have been tested.

Recommended workflow:

```text
Developer
   │
   ↓
dev/suyash
   │
   ↓
Push to GitHub
   │
   ↓
Pull Request
   │
   ↓
Review + tests
   │
   ↓
main
```

Do not force-push `main`.

---

# Important Data Rules

Do not commit:

* `.venv/`
* `.env`
* raw datasets
* private patient information
* credentials/API keys
* generated model binaries unless explicitly required

The `.gitignore` file is configured to prevent common generated and sensitive files from being committed.

---

# Current Technical Direction

The initial use case is:

> Predict whether glucose will exceed 180 mg/dL within the next 2 hours.

The planned architecture is:

```text
Synthetic EHR
     │
     │
     ├──────────────┐
     │              │
     ↓              ↓
Static Features   CGM Time Series
     │              │
     │              ↓
     │        Dynamic Features
     │              │
     └───────┬──────┘
             ↓
      Prediction Model
             ↓
     Risk / Prediction
             ↓
       Patient Twin
             ↓
        Clinician UI
```

The initial modeling approach will use a conventional tabular ML baseline before considering sequence models.

---

# Team Development Principle

Keep the system modular.

Data ingestion, feature engineering, modeling, digital-twin state, and UI should remain separable so that individual components can be developed and tested independently.

Changes should be merged into `main` only when they leave the repository in a runnable state.

---

# Current Project Context

Last updated: 2026-10-07

## Project Goal

BioSynth / Digital Twin is a hackathon-oriented proof of concept for predicting impending hyperglycemic events from CGM trajectory data. It is an engineering prototype and is not a clinically validated medical system.

The current prediction goal is to identify whether a new hyperglycemic event is likely to begin within a 120-minute future horizon, using leakage-safe CGM trajectory features derived from the preceding 60 minutes of glucose history.

## Real Dataset: Hall 2018

The real-data proof uses the Hall 2018 CGM dataset:

* 57 participants
* 105,426 CGM observations
* 38 No diabetes participants
* 14 Prediabetes participants
* 5 T2D participants
* Hyperglycemia engineering threshold: glucose > 180 mg/dL

Detected hyperglycemic events:

* 90 total events
* 24 T2D events

Prediction target:

* Target column: `pre_event_target`
* Predicts onset of a new hyperglycemic event
* Prediction horizon: 120 minutes
* Rows where an event is already underway are not treated as positive early-warning targets

Prediction dataset:

* 89,329 rows
* 1,662 pre-event positives
* 1.86% positive rate

Trajectory feature dataset:

* 18 leakage-safe CGM trajectory features
* 60-minute history/lookback
* Features include current glucose, historical statistics, glucose changes, slope, variability, high-glucose exposure, and trajectory coverage

Participant-level 5-fold cross-validation results:

* Current-glucose baseline: PR-AUC 0.0657, ROC-AUC 0.7453
* Trajectory model: PR-AUC 0.1383, ROC-AUC 0.7743
* Absolute PR-AUC improvement: +0.0726
* Relative PR-AUC improvement: approximately +110.6%
* The trajectory model improved PR-AUC in all 5 folds

Row-level threshold 0.50 for the trajectory model:

* Precision: 4.69%
* Recall: 65.16%
* F1: 0.0875
* Alert rate: 25.85%
* False-positive rate: 25.10%

Important interpretation:

* PR-AUC is the primary metric because the target is highly imbalanced.
* The current results show real predictive signal, but the real dataset is small and event-limited.
* Hall 2018 remains the real-world validation and sanity-check layer.

## Synthetic CGM Layer

A first synthetic raw-CGM generator now exists at:

```text
scripts/generate_synthetic_cgm.py
```

It generates deterministic, reproducible synthetic CGM trajectories for hackathon demonstration and model-development workflows without modifying the Hall 2018 pipeline.

Generated synthetic population:

* 100 synthetic participants
* 7 days per participant
* 5-minute sampling
* 2,016 observations per participant
* 201,600 total observations
* 40 `No diabetes-like` simulation participants
* 30 `Prediabetes-like` simulation participants
* 30 `T2D-like` simulation participants

These are simulation phenotypes, not clinical diagnoses.

Synthetic raw CGM output:

```text
data/02_synthetic/cgm/
```

Each participant has one CSV using the same raw CGM schema as the Hall glucose files:

```text
timestamp
glucose_value_mg_dl
```

Synthetic metadata output:

```text
data/02_synthetic/metadata/synthetic_participants.csv
data/02_synthetic/metadata/synthetic_events.csv
```

The event metadata records simulator ground truth such as generated meal disturbances and stronger hyperglycemic excursions. These event-generation variables are intentionally kept out of the raw participant CGM files and must not automatically become model features.

Synthetic generation summary:

```text
reports/synthetic_cgm_generation_summary.json
```

Current generated synthetic summary:

* Random seed: 20261007
* Total observations: 201,600
* Glucose min/max/mean/median: 40.0 / 400.0 / 124.86 / 114.1 mg/dL
* Observations >180 mg/dL: 18,061
* Percent observations >180 mg/dL: 8.96%
* Participants with at least one >180 mg/dL observation: 65
* Counts >180 mg/dL by phenotype:
  * No diabetes-like: 35
  * Prediabetes-like: 2,836
  * T2D-like: 15,190

The synthetic data is for POC simulation only. It is not clinically validated and should not be presented as patient data.

## Synthetic Prediction Dataset

A synthetic-only prediction dataset builder now exists at:

```text
scripts/build_synthetic_prediction_dataset.py
```

It detects hyperglycemic events directly from the synthetic raw CGM files using the same engineering threshold, `glucose > 180 mg/dL`, and constructs early-warning rows with the same 60-minute lookback and 120-minute prediction horizon used for Hall 2018.

The script intentionally does not use simulator event metadata as a model feature. The simulator ground-truth event file remains separate from detected hyperglycemia events and prediction rows.

Synthetic prediction outputs:

```text
reports/synthetic_prediction_dataset.csv
reports/synthetic_detected_hyperglycemic_events.csv
reports/synthetic_prediction_dataset_summary.json
```

Current synthetic prediction summary:

* Participants represented: 100
* Prediction rows: 198,000
* Detected hyperglycemic events: 1,228
* Future hyperglycemia positives: 41,408
* Future hyperglycemia positive rate: 20.91%
* Pre-event positives: 23,696
* Pre-event positive rate: 11.97%
* Prediction classes:
  * Negative: 155,763
  * Pre-event: 23,696
  * Event underway: 18,541
* Detected events by phenotype:
  * No diabetes-like: 6
  * Prediabetes-like: 316
  * T2D-like: 906

## Synthetic CGM Sanity Analysis

A targeted sanity-analysis script now exists at:

```text
scripts/analyze_synthetic_cgm.py
```

It analyzes the existing generated synthetic CGM population only. It does not regenerate data, build trajectory features, train models, or use prediction labels/future information.

Analysis outputs:

```text
reports/synthetic_cgm_analysis.csv
reports/synthetic_cgm_analysis_summary.json
reports/synthetic_cgm_sample_trajectories.png
```

Current analysis results:

* Overall glucose mean/median/std: 124.86 / 114.10 / 41.32 mg/dL
* Overall p05/p95: 82.0 / 205.2 mg/dL
* Overall min/max: 40.0 / 400.0 mg/dL
* Boundary clipping:
  * Exactly 40 mg/dL: 1 observation, 0.000%
  * Exactly 400 mg/dL: 4 observations, 0.002%
* Phenotype mean/median and >180 rates:
  * No diabetes-like: mean 98.41, median 94.90, >180 0.04%
  * Prediabetes-like: mean 122.82, median 116.80, >180 4.69%
  * T2D-like: mean 162.17, median 152.20, >180 25.12%
* Temporal smoothness:
  * Median absolute 5-minute change: 3.10 mg/dL
  * 95th percentile absolute 5-minute change: 13.50 mg/dL
  * 99th percentile absolute 5-minute change: 22.70 mg/dL
  * Maximum absolute 5-minute change: 80.10 mg/dL
  * 5-minute changes >20 mg/dL: 1.54%
  * 5-minute changes >30 mg/dL: 0.43%
  * 5-minute changes >50 mg/dL: 0.07%
* Hyperglycemic event counts by phenotype:
  * No diabetes-like: 6
  * Prediabetes-like: 316
  * T2D-like: 906
* No diabetes-like hyperglycemia investigation:
  * 35 readings >180 mg/dL
  * 5 participants affected
  * 6 events
  * Maximum glucose among affected participants: 195.8 mg/dL
  * Readings are spread across multiple participants/excursions rather than being caused by one extreme participant
* Phenotype overlap:
  * Participant-level mean glucose ranges overlap for No diabetes-like / Prediabetes-like and Prediabetes-like / T2D-like
  * Participant-level maximum glucose ranges also overlap for both adjacent phenotype pairs
  * Qualitative conclusion: substantial overlap
* Event total comparison:
  * Analysis-calculated events: 1,228
  * Synthetic prediction-stage events: 1,228
  * The totals agree because both use raw synthetic CGM, glucose >180 mg/dL, and a >15-minute high-reading gap to start a new event

Synthetic data decision:

```text
SYNTHETIC DATA STATUS: ACCEPT FOR POC
```

## Synthetic Trajectory Feature Dataset

A synthetic-only trajectory feature builder now exists at:

```text
scripts/build_synthetic_trajectory_features.py
```

It uses:

```text
reports/synthetic_prediction_dataset.csv
data/02_synthetic/cgm/
```

Output:

```text
reports/synthetic_cgm_trajectory_features.csv
reports/synthetic_cgm_trajectory_features_summary.json
```

Current synthetic trajectory feature results:

* Rows: 198,000
* Participants: 100
* Target: `pre_event_target`
* Positive rows: 23,696
* Negative rows: 174,304
* Positive rate: 11.97%
* Rows by phenotype:
  * No diabetes-like: 79,200 rows, 144 positives, 0.18% positive rate
  * Prediabetes-like: 59,400 rows, 7,396 positives, 12.45% positive rate
  * T2D-like: 59,400 rows, 16,156 positives, 27.20% positive rate
* Trajectory completeness:
  * Complete 60-minute history: 198,000 rows
  * Short history: 0 rows
  * Missing current glucose: 0 rows
  * Feature calculation failures: 0 rows

The feature dataset includes `person_id`, `prediction_timestamp`, `simulation_phenotype`, `pre_event_target`, and the same 18 trajectory features used for the Hall trajectory pipeline:

```text
current_glucose_mg_dl
current_is_high
history_mean_glucose
history_median_glucose
history_min_glucose
history_max_glucose
history_range_glucose
glucose_change_15m
glucose_change_30m
glucose_change_60m
glucose_slope_per_hour
history_std_glucose
history_cv_glucose
history_high_count
history_high_fraction
history_minutes_above_180
trajectory_numeric_count
trajectory_actual_history_minutes
```

Leakage checks:

* Output row count matches the synthetic prediction dataset: PASS
* `person_id` and `prediction_timestamp` remain aligned: PASS
* `pre_event_target` remains unchanged: PASS
* Future/event-derived features excluded: PASS
* Simulator-generation parameters excluded: PASS
* Simulator event ground truth excluded: PASS
* Current glucose agrees with raw CGM at prediction timestamp: PASS

`simulation_phenotype` is retained only as context metadata and must not be used as a predictive feature.

## Synthetic Participant-Separated Model Evaluation

A synthetic model evaluation script now exists at:

```text
scripts/cross_validate_synthetic_models.py
```

It uses participant-separated 5-fold cross-validation with `GroupKFold(n_splits=5)`, grouping exclusively by `person_id`. Each row receives out-of-fold predictions from models that were not trained on that participant.

Models compared:

* Current-glucose baseline: logistic regression using only `current_glucose_mg_dl`
* CGM trajectory model: logistic regression using the 18 CGM trajectory features

Both models use median imputation, `StandardScaler`, `LogisticRegression(max_iter=2000, class_weight="balanced")`, and no threshold tuning.

Outputs:

```text
reports/synthetic_model_oof_predictions.csv
reports/synthetic_cross_validation_results.csv
reports/synthetic_cross_validation_summary.json
```

Overall out-of-fold results:

* Current-glucose baseline:
  * PR-AUC: 0.1918
  * ROC-AUC: 0.7303
  * Precision @0.50: 0.2364
  * Recall @0.50: 0.6596
  * F1 @0.50: 0.3481
  * Specificity @0.50: 0.7104
  * False-positive rate @0.50: 0.2896
* CGM trajectory model:
  * PR-AUC: 0.4433
  * ROC-AUC: 0.8539
  * Precision @0.50: 0.3117
  * Recall @0.50: 0.7773
  * F1 @0.50: 0.4449
  * Specificity @0.50: 0.7666
  * False-positive rate @0.50: 0.2334

Trajectory improvement:

* PR-AUC absolute improvement: +0.2515
* Relative PR-AUC improvement: +131.15%
* ROC-AUC absolute improvement: +0.1236
* Trajectory PR-AUC improved in 5 of 5 folds
* Trajectory ROC-AUC improved in 5 of 5 folds

Phenotype out-of-fold subgroup results:

* No diabetes-like:
  * Participants: 40
  * Rows: 79,200
  * Positives: 144
  * Positive rate: 0.18%
  * Baseline PR-AUC: 0.0112
  * Trajectory PR-AUC: 0.0110
  * Baseline ROC-AUC: 0.7015
  * Trajectory ROC-AUC: 0.7109
* Prediabetes-like:
  * Participants: 30
  * Rows: 59,400
  * Positives: 7,396
  * Positive rate: 12.45%
  * Baseline PR-AUC: 0.1313
  * Trajectory PR-AUC: 0.1792
  * Baseline ROC-AUC: 0.5669
  * Trajectory ROC-AUC: 0.6479
* T2D-like:
  * Participants: 30
  * Rows: 59,400
  * Positives: 16,156
  * Positive rate: 27.20%
  * Baseline PR-AUC: 0.2262
  * Trajectory PR-AUC: 0.5782
  * Baseline ROC-AUC: 0.4393
  * Trajectory ROC-AUC: 0.8117

Participant leakage check: PASS.

`simulation_phenotype`, `person_id`, `prediction_timestamp`, simulator-generation parameters, simulator event ground truth, and future/event-derived columns were not predictive features. `simulation_phenotype` was used only after out-of-fold prediction generation for subgroup reporting.

Synthetic model decision:

```text
SYNTHETIC MODEL STATUS: TRAJECTORY SIGNAL CONFIRMED
```

## BioSynth Digital Twin Runtime

The accepted synthetic trajectory predictor has been converted into a reusable hackathon runtime.

Final model training:

```text
scripts/train_synthetic_final_model.py
```

Runtime:

```text
src/biosynth_twin.py
```

Command-line demo:

```text
scripts/demo_digital_twin.py
```

Saved runtime artifacts:

```text
models/biosynth_trajectory_model.joblib
models/biosynth_trajectory_model_metadata.json
```

The final model artifact is fitted on the full accepted synthetic trajectory feature dataset for deployment/demo use:

* Training rows: 198,000
* Training participants: 100
* Positive rows: 23,696
* Positive rate: 11.97%
* Feature count: 18
* Model: median imputation, `StandardScaler`, `LogisticRegression(max_iter=2000, class_weight="balanced")`

Performance claims still come from participant-separated out-of-fold evaluation, not from this full-data fit:

* Reference OOF PR-AUC: 0.4433
* Reference OOF ROC-AUC: 0.8539

`BioSynthTwin` runtime input:

```text
timestamp
glucose_value_mg_dl
```

The runtime computes the same 18 trajectory features from historical CGM only and returns a dictionary containing:

* `prediction_timestamp`
* `current_glucose_mg_dl`
* `glucose_state`
* `glucose_change_15m`
* `glucose_change_30m`
* `glucose_change_60m`
* `glucose_slope_per_hour`
* `trend`
* `hyperglycemia_risk`
* `risk_state`
* `prediction_horizon_minutes`
* `history_window_minutes`
* `history_observation_count`

Risk-state display thresholds:

* `LOW`: risk < 0.30
* `ELEVATED`: 0.30 <= risk < 0.60
* `HIGH`: risk >= 0.60

Current glucose display states:

* `BELOW_RANGE`: glucose < 70
* `IN_RANGE`: 70 <= glucose <= 180
* `ABOVE_180`: glucose > 180

Trend display states:

* `RAPIDLY FALLING`
* `FALLING`
* `STABLE`
* `RISING`
* `RAPIDLY RISING`

These are demo/engineering display states, not clinical diagnostic thresholds.

Safety boundary:

* The runtime parses and sorts timestamps, coerces glucose to numeric, rejects empty or invalid input, and uses only CGM observations at or before the requested prediction timestamp.
* `BioSynthTwin` does not require phenotype, simulator parameters, future CGM, event labels, or simulator event ground truth.

Successful command-line demo:

```text
Participant:           SYN-0085
Prediction time:       2026-01-04 12:10:00
Current glucose:       176.5 mg/dL
Glucose state:         IN_RANGE
Trend:                 STABLE
15-min change:         +1.2 mg/dL
30-min change:         -15.1 mg/dL
60-min change:         +8.2 mg/dL
Slope:                 +5.5 mg/dL/hour
Hyperglycemia risk:    94.6%
Risk state:            HIGH
Prediction horizon:    120 min
```

The model and runtime remain a synthetic-data hackathon POC and are not clinically validated.

## Streamlit Dashboard Foundation

A local Streamlit dashboard foundation now exists at:

```text
app.py
```

This first dashboard version uses the established synthetic virtual participant:

```text
SYN-0085
```

Dashboard behavior:

* Loads `data/02_synthetic/cgm/SYN-0085.csv`
* Provides a timeline slider through the participant's CGM record
* At the selected timestamp, passes only CGM observations at or before that timestamp into `BioSynthTwin`
* Calls the real runtime in `src/biosynth_twin.py`
* Displays the current Digital Twin state and 120-minute hyperglycemia risk
* Shows a recent historical CGM chart ending at the selected timestamp
* Includes the 180 mg/dL POC hyperglycemia engineering threshold in the chart
* Includes a visible notice that the demo uses synthetic virtual participants and is not clinically validated

Currently displayed state:

* Current glucose
* Glucose state
* Trend
* Hyperglycemia risk
* Risk state
* Prediction horizon
* 15-minute glucose change
* 30-minute glucose change
* 60-minute glucose change
* Glucose slope

Verification result:

* Streamlit available in `.venv`: version 1.64.0
* `app.py` compiles successfully
* `BioSynthTwin` loads successfully
* `SYN-0085` loads successfully with 2,016 CGM rows
* Timeline contains data
* Runtime prediction can be produced
* Runtime risk is within 0-1
* Verification passed with 0 future CGM rows passed into the runtime
* Brief headless Streamlit startup succeeded and was terminated

Launch command:

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

## Guided Judge-Facing Dashboard Replay

The Streamlit dashboard has been upgraded into a guided judge-facing Digital Twin replay while keeping the same local app:

```text
app.py
```

Demo story:

```text
BioSynth watches recent CGM trajectory and estimates the risk of crossing the >180 mg/dL POC hyperglycemia threshold within the next 120 minutes.
```

Selected guided demonstration event:

* Participant: `SYN-0085`
* Event ID: `SYN-0085-0-32`
* Event start: `2026-01-04 13:40:00`
* Event peak glucose: 400.0 mg/dL

Dashboard modes:

* `GUIDED DEMO`: default mode with Previous, Next, and Reset controls through real timestamps around the selected synthetic event
* `EXPLORE TIMELINE`: manual slider mode for inspecting the participant timeline

Current judge-facing UI sections:

* Strong BioSynth header and plain-English demo story
* Synthetic virtual participant label
* Four primary state cards: current glucose, trajectory, 120-minute risk, Twin state
* Rule-based plain-English interpretation from current glucose, trend, risk state, and horizon
* Historical CGM chart ending at the replay timestamp
* POC threshold line at 180 mg/dL
* Secondary trajectory details: 15-minute change, 30-minute change, 60-minute change, slope
* Expanders for `What BioSynth sees`, `Why this is a Digital Twin`, and `Evidence behind the prototype`
* Guided-mode-only `What happened next?` retrospective reveal
* POC notice that the participant is synthetic and predictions are not clinically validated

Retrospective reveal boundary:

* The `What happened next?` chart may show future synthetic CGM for storytelling only.
* Future CGM is clearly separated from the model-input chart.
* Future CGM is not passed into `BioSynthTwin`.

Historical-only runtime guarantee:

* At every guided or explore timestamp, the dashboard constructs `historical_cgm = CGM rows where timestamp <= replay timestamp`.
* `BioSynthTwin.predict_state()` receives only this historical CGM slice.
* Verification confirmed `max_runtime_input <= replay_timestamp` for guided replay checks.

Pre-event vs event-underway display semantics:

* While current glucose is <=180 mg/dL, the dashboard displays the runtime's pre-event model risk normally as `120-MIN RISK` with `LOW`, `ELEVATED`, or `HIGH` Twin state.
* Once current glucose is >180 mg/dL, the dashboard no longer presents the pre-event probability as an ordinary future-risk state.
* In the event-underway state, the judge-facing display switches to `120-MIN RISK: Event underway` and `TWIN STATE: ABOVE POC THRESHOLD`.
* This is a UI/state-semantics correction only. It does not modify `BioSynthTwin.predict_state()`, the underlying model probability, the model thresholds, or the synthetic data.

Guided replay verification:

```text
Earlier pre-event:
  replay timestamp: 2026-01-04 12:10:00
  current glucose: 176.5 mg/dL
  risk probability: 0.9464
  risk state: HIGH
  trend: STABLE
  max runtime input: 2026-01-04 12:10:00

Closer to event:
  replay timestamp: 2026-01-04 13:10:00
  current glucose: 174.0 mg/dL
  risk probability: 0.8603
  risk state: HIGH
  trend: FALLING
  max runtime input: 2026-01-04 13:10:00

Event/threshold region:
  replay timestamp: 2026-01-04 13:40:00
  current glucose: 242.2 mg/dL
  underlying pre-event probability: 0.0000
  displayed 120-min risk: Event underway
  displayed Twin state: ABOVE POC THRESHOLD
  trend: RISING
  max runtime input: 2026-01-04 13:40:00
```

Additional verification:

* `app.py` compiles successfully
* Guided demo initializes successfully
* Explore timeline has 2,016 rows available
* No future CGM entered `BioSynthTwin` during verification
* Brief headless Streamlit startup succeeded and was terminated

## Current Boundary

Completed:

* Real Hall 2018 validation, target construction, trajectory features, participant-level cross-validation, and event-level reporting.
* First synthetic raw-CGM generation layer.
* Synthetic hyperglycemic-event detection and early-warning prediction dataset construction.
* Synthetic CGM sanity analysis with participant-level statistics, event-structure checks, phenotype-overlap review, and representative trajectory visualization.
* Synthetic leakage-safe CGM trajectory feature dataset with the same 18 feature names as the Hall trajectory pipeline.
* Synthetic participant-separated baseline model evaluation confirming trajectory signal against current glucose alone.
* Reusable BioSynthTwin runtime, final synthetic trajectory model artifact, model metadata, and command-line demo.
* Streamlit dashboard foundation proving `BioSynthTwin` can drive a historical-only visual replay for synthetic participant SYN-0085.
* Guided judge-facing replay mode with event-centered navigation, retrospective future reveal separation, evidence panel, and historical-only runtime verification.

Not yet completed:

* Digital Twin UI integration
* Sensor gaps or missing-data simulation

Next step:

Final hackathon polish: verify the complete judge demo flow, capture final screenshots/results, and prepare the concise BioSynth pitch/demo narrative.

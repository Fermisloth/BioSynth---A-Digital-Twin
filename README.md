# BioSynth — A Digital Twin

A research-grade proof of concept for a patient-specific digital twin that combines longitudinal EHR data with dynamic physiological time-series data to support predictive health monitoring.

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

## Current Boundary

Completed:

* Real Hall 2018 validation, target construction, trajectory features, participant-level cross-validation, and event-level reporting.
* First synthetic raw-CGM generation layer.
* Synthetic hyperglycemic-event detection and early-warning prediction dataset construction.

Not yet completed:

* Synthetic feature generation
* Model training on synthetic data
* Digital Twin UI integration
* Sensor gaps or missing-data simulation

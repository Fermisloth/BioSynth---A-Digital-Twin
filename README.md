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

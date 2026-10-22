# BioSynth Digital Twin

BioSynth is a research proof of concept for estimating whether a new glucose excursion above 180 mg/dL may begin within the next 120 minutes. It combines recent continuous glucose monitor (CGM) readings with a static, synthetic electronic health record (EHR) profile. The project is not clinically validated and must not be used to guide care.

## Run the interface

Use Python 3.11 or newer. From the repository root, create an environment and install the pinned dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Launch the BioSynth command-center interface:

```powershell
streamlit run ui/app.py
```

On Windows, if the environment is not activated, run `\.venv\Scripts\streamlit.exe run ui\app.py` from PowerShell. The default UI can use its calibrated simulation when a trained model artifact is unavailable. It reports which execution path is active in the sidebar.

The repository also has a model-backed event replay demo at the root `app.py`:

```powershell
streamlit run app.py
```

That replay requires the generated data and trained model artifacts described in the [technical guide](docs/BIOSYNTH_TECHNICAL_GUIDE.md). To run a single prediction from the terminal after building the CGM model, use:

```powershell
python scripts/demo_digital_twin.py
```

## Verify the project

```powershell
python -m unittest discover -s tests -v
```

## What the model sees

The runtime uses at most 60 minutes of CGM history at or before the prediction timestamp. CGM-only mode has 18 trajectory features. Fusion mode adds 11 encoded fields from an explicitly supplied synthetic EHR profile. Neither mode receives future CGM as a predictive input.

The data pipeline includes synthetic demonstration data and a separate Hall 2018 CGM evidence branch. Hall 2018 does not validate the synthetic EHR + CGM fusion model. Read the [model card](docs/model_card.md) before interpreting any metric.

## Documentation

Start with the [documentation map](docs/README.md). It links the setup walkthrough, architecture, frontend guide, runtime API, data dictionary, and model card.

## Repository layout

```text
app.py                 Model-backed guided event replay
ui/app.py              BioSynth command-center interface
src/biosynth_twin.py   Reusable prediction runtime
scripts/               Data generation, feature building, evaluation, training
configs/               Data-source and model configuration
data/                  Synthetic CGM, EHR, and event data
models/                Generated model artifacts and metadata
reports/               Generated evaluation and analysis outputs
docs/                  Project guides and technical references
tests/                 Runtime contract tests
```

## Data and safety

The dashboard's participant profiles and generated EHR/CGM records are synthetic. Do not treat displayed risk states as diagnoses or treatment recommendations. Keep private data, credentials, virtual environments, and generated model binaries out of source control.

## Challenge Submission Information

This section is the README checklist for the submitted GitHub repository. Replace every bracketed placeholder before the final submission. A checked item means the repository currently contains the relevant explanation or artifact; it does not automatically confirm that an external link is public.

### Team Details

- **Team name:** `Ctrl+Sloth`
- **Team member 1:** `Suyash Singh`
- **Team member 1 role:** `data analyst and Ai developer`
- **Team member 2:** `Swastik Shaw`
- **Team member 2 role:** `web design and documentation expert`

### College / Incubator Information

- **College or institution:** `GL Bajaj institute of Technology and Management`
- **City and country:** `Knowledge Park 3 , Greater Noida , Bauttam Budh Nagar ,Uttar Pradesh , India `

### Project Title

**BioSynth Digital Twin: Synthetic EHR + Dynamic CGM for 120-Minute Glucose Excursion Risk**

### Problem Statement

Continuous glucose monitor readings show current glucose, but a single reading does not explain where glucose is heading. Reactive threshold alerts may appear only after glucose is already high. CGM-only trajectory models can miss static health context, while static health records cannot describe the participant's current physiological movement.

BioSynth addresses this problem by combining a static synthetic EHR profile with a dynamic synthetic CGM history. The prototype estimates whether a **new glucose excursion above 180 mg/dL may begin within the next 120 minutes**, using only CGM history available at or before the replay timestamp.

### Healthcare Use Case

BioSynth is a research proof of concept for proactive glycemic monitoring and Digital Twin replay. It demonstrates how a virtual participant can be represented using:

- A static synthetic EHR profile with age, sex, BMI, HbA1c, fasting glucose, family history, hypertension history, and activity level.
- Dynamic synthetic CGM readings collected at five-minute intervals.
- A historical CGM window converted into 18 trajectory features.
- Eleven encoded EHR features, producing 29 fusion features in total.
- A logistic-regression risk estimate for a new above-180 mg/dL excursion within 120 minutes.
- A dashboard for observation, forecasting, explainability, event replay, and counterfactual scenario exploration.

This is not a diagnostic, treatment, prescription, or clinical decision-support product. All EHR and CGM records used for the fusion demonstration are synthetic. Hall 2018 is used only as real-CGM proof-of-signal evidence and does not validate EHR + CGM fusion.

### Technical Stack

| Area | Technology |
| --- | --- |
| Language | Python 3.11+ |
| Dashboard | Streamlit |
| Data processing | pandas, NumPy |
| Visualization | Plotly, Matplotlib |
| Machine learning | scikit-learn logistic regression pipeline |
| Model preprocessing | Median imputation and StandardScaler |
| Model persistence | joblib and JSON metadata |
| Configuration | YAML |
| Testing | Python `unittest` |
| Version control | Git |

### AI / ML Model And Framework Details

The prototype uses two separately reported model paths:

1. **CGM-only comparison model:** 18 trajectory features from the recent 60-minute CGM history.
2. **EHR + CGM fusion model:** the same 18 CGM features plus 11 encoded static EHR features, for 29 total features.

The final model is a scikit-learn pipeline:

```text
numeric feature table
    -> median imputation
    -> StandardScaler
    -> LogisticRegression(class_weight="balanced")
    -> probability of pre_event_target
```

Evaluation uses participant-separated `GroupKFold` validation by `person_id`, so rows from one participant do not appear in both training and validation folds. The target is a new glucose excursion above 180 mg/dL beginning within the next 120 minutes.

Recorded evaluation results:

| Model | PR-AUC | ROC-AUC | Interpretation |
| --- | ---: | ---: | --- |
| Current glucose baseline | 0.1918 | 0.7303 | One-feature reference |
| CGM trajectory | 0.4433 | 0.8539 | Best PR-AUC in this comparison |
| EHR + CGM fusion | 0.4352 | 0.8729 | Best ROC-AUC; mixed overall result |


### Submission Artifact Links

- **GitHub repository:** `https://github.com/Fermisloth/BioSynth---A-Digital-Twin`
- **Demo video, unlisted YouTube:** `https://youtu.be/Et2PBGlQq4A`
- **Presentation PDF/PPT:** `https://canva.link/mu8d58n41ezrwa0`
- **Architecture PDF:** [docs/BIOSYNTH_ARCHITECTURE.pdf](docs/BIOSYNTH_ARCHITECTURE.pdf)
- **Architecture explanation:** [docs/BIOSYNTH_ARCHITECTURE.md](docs/BIOSYNTH_ARCHITECTURE.md)
- **10-minute PowerPoint speaker script:** [docs/BIOSYNTH_PPT_10_MINUTE_SPEAKER_SCRIPT.md](docs/BIOSYNTH_PPT_10_MINUTE_SPEAKER_SCRIPT.md)
- **Command-center video script:** [docs/BIOSYNTH_10_MINUTE_COMMAND_CENTER_VIDEO_SCRIPT.md](docs/BIOSYNTH_10_MINUTE_COMMAND_CENTER_VIDEO_SCRIPT.md)

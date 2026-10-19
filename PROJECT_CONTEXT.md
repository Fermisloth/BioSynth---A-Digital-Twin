# BioSynth project context

Use this file to orient yourself before changing the repository. Treat the source code, configuration, model metadata, and generated reports as the authority for current behavior. Follow the links below for detail instead of expanding this summary into a second copy of every guide.

## Project purpose

BioSynth is a research proof of concept that estimates whether a new glucose excursion above the engineering threshold of 180 mg/dL may begin within the next 120 minutes. Runtime features use up to 60 minutes of CGM history available at or before the prediction timestamp.

The project is not clinically validated. Synthetic participants, displayed risk probabilities, alerts, and risk categories are demonstrations. They are not diagnoses, treatment recommendations, or medical advice.

## Two Streamlit applications

The repository has two separate Streamlit entry points:

| File | Role | Runtime behavior |
| --- | --- | --- |
| `ui/app.py` | BioSynth command-center interface with six views: overview, forecast, intervention sandbox, explainability, cohort/event replay, and settings/tour. | Attempts to load the CGM-only model. If it cannot, the UI can use a calibrated simulation path and reports runtime status in the sidebar. |
| `app.py` | Guided replay of a detected synthetic event. | Calls `BioSynthTwin` and exposes CGM-only and EHR + CGM fusion modes. Requires the generated data and matching model artifacts. |

Do not describe simulation output as live model inference. The command-center is the visual dashboard; the root app is the model-backed replay demo.

## Runtime contract

The reusable runtime is `BioSynthTwin` in `src/biosynth_twin.py`.

- `CGM_ONLY` uses 18 CGM trajectory features.
- `EHR_CGM_FUSION` uses those 18 features plus 11 encoded values from a static synthetic EHR profile supplied explicitly by the caller.
- The runtime accepts CGM columns `timestamp` and `glucose_value_mg_dl`.
- It filters out rows after the prediction timestamp before building the feature vector.
- Fusion mode requires all eight EHR fields. It does not silently fall back to CGM-only mode.
- Model feature ordering is defined by the matching metadata JSON.
- `LOW`, `ELEVATED`, and `HIGH` are demo display states, not clinical thresholds.

Model binaries (`*.joblib`) are generated outputs ignored by Git. Their expected paths and build steps are documented in the [technical guide](docs/BIOSYNTH_TECHNICAL_GUIDE.md).

## Data and evidence

- Synthetic CGM and synthetic EHR support repeatable development, evaluation, and demos. Synthetic EHR profiles are not genuine medical records.
- Hall 2018 is a separate real-CGM proof-of-signal branch for CGM trajectory modeling. It does not validate the synthetic EHR + CGM fusion model.
- Evaluation is participant-separated five-fold `GroupKFold`. Read the [model card](docs/model_card.md) before repeating or interpreting performance claims.
- Future CGM can be used to construct training labels, but it must stay out of prediction features and runtime inputs.

## Run commands

From the repository root, using Python 3.11+:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Start the command-center UI:

```powershell
streamlit run ui/app.py
```

Start the model-backed guided replay after generating the required data and models:

```powershell
streamlit run app.py
```

Run the runtime contract tests:

```powershell
python -m unittest discover -s tests -v
```

## Repository map

| Path | Purpose |
| --- | --- |
| `src/biosynth_twin.py` | Runtime validation, feature construction, inference, and returned state. |
| `scripts/` | Synthetic data generation, event/feature building, evaluation, training, and demo scripts. |
| `ui/app.py` | Command-center Streamlit app. |
| `ui/assets/style.css` and `ui/assets/` | Frontend styling, DNA branding, and visual assets. |
| `app.py` | Model-backed event replay app. |
| `configs/` | Data source and model configuration. |
| `data/` | Included generated synthetic data and configured data directories. |
| `models/` | Model metadata; trained binary artifacts are generated locally. |
| `reports/` | Generated datasets, analysis, and evaluation summaries. |
| `tests/test_biosynth_twin.py` | Runtime behavior tests. |
| `docs/README.md` | Documentation index and topic routing. |

## Working in this repository

1. Read the relevant page in the [documentation index](docs/README.md) and inspect the owning source files before editing.
2. Preserve existing runtime contracts, UI controls, state keys, and data boundaries when changing presentation or refactoring.
3. Keep synthetic, observed, evaluated, and planned behavior clearly distinguished in prose and interface labels.
4. Update the owning documentation page when a documented behavior changes. Keep this file limited to cross-cutting orientation facts.
5. For frontend changes, read the [frontend guide](docs/frontend_guide.md). For runtime changes, read the [API reference](docs/api_reference.md) and [architecture guide](docs/architecture.md).

## Documentation map

- [Project README](README.md): install, launch, and first steps.
- [Technical guide](docs/BIOSYNTH_TECHNICAL_GUIDE.md): data/model workflow, artifact generation, and evaluation snapshot.
- [Architecture](docs/architecture.md): module responsibilities and data boundaries.
- [Frontend guide](docs/frontend_guide.md): command-center behavior and visual implementation.
- [Runtime API](docs/api_reference.md): `BioSynthTwin` inputs and outputs.
- [Data dictionary](docs/data_dictionary.md): dataset columns and feature meanings.
- [Model card](docs/model_card.md): intended use, results, and limitations.

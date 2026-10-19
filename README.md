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

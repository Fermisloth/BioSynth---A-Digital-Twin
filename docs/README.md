# BioSynth documentation

Choose the document that answers your current question. The repository has two Streamlit entry points; read the [frontend guide](frontend_guide.md) before choosing which interface to launch.

## Start by task

| I want to… | Read |
| --- | --- |
| Install the project and run an interface | [Project README](../README.md) |
| Understand the complete data-to-prediction flow | [Technical guide](BIOSYNTH_TECHNICAL_GUIDE.md) |
| Understand module boundaries and data safety rules | [Architecture](architecture.md) |
| Launch or change the command-center UI | [Frontend guide](frontend_guide.md) |
| Call `BioSynthTwin` from Python | [Runtime API reference](api_reference.md) |
| Understand a dataset or feature column | [Data dictionary](data_dictionary.md) |
| Interpret evaluation metrics and limitations | [Model card](model_card.md) |
| View the presentation architecture diagram | [Architecture overview](BIOSYNTH_ARCHITECTURE.md) and [PDF](BIOSYNTH_ARCHITECTURE.pdf) |
| Prepare a ten-minute prototype explanation | [10-minute video script](BIOSYNTH_10_MINUTE_VIDEO_SCRIPT.md) |
| Explain the current command-center screenshots | [Command-center video script](BIOSYNTH_10_MINUTE_COMMAND_CENTER_VIDEO_SCRIPT.md) |
| Present the 19-slide PowerPoint in ten minutes | [PowerPoint speaker script](BIOSYNTH_PPT_10_MINUTE_SPEAKER_SCRIPT.md) |

## Project boundaries

- The runtime predicts `pre_event_target`: a new excursion above the engineering threshold of 180 mg/dL beginning within 120 minutes.
- Runtime feature construction uses up to 60 minutes of CGM history at or before the prediction timestamp.
- `CGM_ONLY` uses 18 CGM features. `EHR_CGM_FUSION` adds 11 fields from a required, explicitly supplied synthetic EHR profile.
- Hall 2018 supplies real-CGM proof-of-signal evidence for the trajectory model. It does not validate EHR + CGM fusion.
- The project and its risk categories are a research POC, not a medical device or clinical decision tool.

## Documentation structure

The docs separate task instructions (setup and running), explanations (architecture and model behavior), and reference material (API and data columns). This follows the user needs described by [Diátaxis](https://diataxis.fr/start-here/) and keeps the top-level [README](../README.md) focused on orientation and first steps. For the writing principles behind this structure, see [documentation research notes](documentation-research.md).

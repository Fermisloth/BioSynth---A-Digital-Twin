# BioSynth Documentation Map

This folder explains the BioSynth Digital Twin in layers. You do not need to be a programmer to start. Read the documents in the order that matches your question.

 ```mermaid
 flowchart TD
   START[Choose a question] --> OVERVIEW[Technical guide<br/>plain-language tour]
   START --> ARCH[Architecture<br/>system boundaries]
   START --> DATA[Data dictionary<br/>columns and meanings]
   START --> API[API reference<br/>runtime inputs and outputs]
   START --> CARD[Model card<br/>metrics and limitations]
   OVERVIEW --> CODE[Repository code and generated artifacts]
   ARCH --> CODE
   DATA --> CODE
   API --> CODE
   CARD --> CODE
 ```

## Start Here

1. **[BIOSYNTH_TECHNICAL_GUIDE.md](BIOSYNTH_TECHNICAL_GUIDE.md)** explains the whole project in ordinary language: data, scripts, machine-learning ideas, runtime, dashboard, design decisions, and limitations.
2. **[BIOSYNTH_ARCHITECTURE.md](BIOSYNTH_ARCHITECTURE.md)** gives the official architecture explanation and separates offline training from live replay and inference. The matching **[BIOSYNTH_ARCHITECTURE.pdf](BIOSYNTH_ARCHITECTURE.pdf)** is the printable diagram version.
3. **[data_dictionary.md](data_dictionary.md)** explains every important column, what it means, where it comes from, and whether it is allowed into the model.
4. **[architecture.md](architecture.md)** explains why the folders and pipeline stages exist, why the fusion model has 29 features, and where safety boundaries are enforced.
5. **[api_reference.md](api_reference.md)** explains how code calls `BioSynthTwin`, what inputs are required, and what the returned prediction state means.
6. **[model_card.md](model_card.md)** explains what the models were trained to do, how they were evaluated, what the metrics mean, and what the models must not be used for.

## Choose By Question

| Question | Read this first |
| --- | --- |
| What is BioSynth trying to predict? | Technical guide, sections 1 and 2 |
| Where does each dataset come from? | Data dictionary |
| Why are there 18 CGM features and 11 EHR features? | Architecture guide and data dictionary |
| How does the dashboard make one prediction? | API reference and technical guide section 17 |
| What is real evidence versus synthetic evidence? | Technical guide section 1 and model card |
| Can I rebuild the pipeline? | Technical guide section 16 |
| How should I interpret PR-AUC and ROC-AUC? | Model card and technical guide section 26 |
| What are the limitations? | Model card and technical guide section 18 |

## One-Paragraph System Summary

BioSynth creates 100 virtual participants. Each participant has a changing synthetic CGM stream and one static synthetic EHR profile. During offline work, historical CGM is converted into 18 trajectory features, the EHR profile is converted into 11 numeric features, and the two sets are joined into 29 fusion features. A participant-separated logistic regression model learns from those features. During live replay, `BioSynthTwin` receives only CGM readings at or before the replay timestamp plus the selected participant's EHR profile, computes the same features, loads the saved artifact, and returns a probability for a new glucose excursion above 180 mg/dL beginning within the next 120 minutes. The Streamlit dashboard turns that state into a readable replay.

## Important Boundaries

- All EHR records in this POC are synthetic. They are not genuine clinical records.
- The CGM-only model is a comparison baseline. It is separate from the EHR + CGM fusion model.
- Hall 2018 is real-world CGM evidence for trajectory signal. It did not validate the synthetic EHR fusion model.
- The runtime cannot use future CGM readings. Future values may appear in the dashboard's retrospective story, but they are kept outside model input.
- A high displayed risk is a model output for a hackathon proof of concept, not medical advice.

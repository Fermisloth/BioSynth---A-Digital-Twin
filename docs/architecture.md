 # BioSynth Architecture Guide

 This document explains the current repository architecture in plain language. It is intentionally more detailed than a diagram: the goal is to show what each layer does, why it exists, and what information is allowed to cross each boundary.

 ## 1. The shortest explanation

 BioSynth is a small prediction system with two input streams:

 1. A static synthetic EHR profile, such as age, BMI, HbA1c, and activity level.
 2. A changing synthetic CGM stream, containing timestamped glucose readings.

 At a selected replay time, the system uses the participant's EHR profile and only the CGM readings available up to that time. It converts those inputs into 29 numeric features, sends them to a saved logistic-regression pipeline, and displays a Digital Twin state plus a 120-minute pre-event risk.

 The CGM-only model is retained as a comparison baseline. It uses the same 18 CGM features but does not receive EHR values.

 ```mermaid
 flowchart LR
   EHR[Static synthetic EHR<br/>100 profiles] --> EENC[EHR encoding<br/>11 numeric features]
   CGM[Dynamic synthetic CGM<br/>5-minute readings] --> HIST[Historical window<br/>60 minutes, at or before replay time]
   HIST --> CENC[CGM feature engineering<br/>18 trajectory features]
   EENC --> FUSION[EHR + CGM fusion<br/>29 total features]
   CENC --> FUSION
   FUSION --> MODEL[Logistic regression pipeline]
   MODEL --> TWIN[BioSynthTwin runtime]
   TWIN --> DASH[Streamlit dashboard]
   DASH --> RISK[120-minute pre-event risk]
   CENC --> BASE[Separate CGM-only baseline]
 ```

 ## 2. Two different pipelines

 The project has an offline pipeline and a live inference pipeline.

 ### Offline training and evaluation

 Offline work happens before someone uses the dashboard:

 ```text
 synthetic CGM files + synthetic EHR records
             |
             v
 event labels and historical features
             |
             v
 18 CGM features + 11 encoded EHR features
             |
             v
 participant-separated evaluation
             |
             v
 final full-data fusion model artifact
 ```

 The offline scripts create data, create labels, build features, measure models, and save the final model. They may read future values while constructing a training label because the label is the answer we want to learn. They must not place those future values into the predictive feature columns.

 ### Live replay and inference

 Live work happens every time the dashboard moves to a replay timestamp:

 ```text
 selected replay timestamp
             |
             +--> CGM rows with timestamp <= selected time
             |
             +--> explicit static EHR profile
                          |
                          v
                  BioSynthTwin runtime
                          |
                          v
                  29 feature values
                          |
                          v
                  saved fusion model
                          |
                          v
                  Digital Twin state and risk
 ```

 The live pipeline cannot use future CGM. The dashboard also shows a separate retrospective “What happened next?” section, but that section is deliberately not passed to the model.

 ```mermaid
 flowchart TD
   subgraph OFFLINE[Offline training and evaluation]
     RAW[Generated CGM + generated EHR] --> LABELS[Event labels]
     LABELS --> FEATURES[18 CGM + 11 EHR features]
     FEATURES --> CV[Participant-separated cross-validation]
     CV --> ARTIFACT[Saved model artifacts]
   end
   subgraph LIVE[Live replay and inference]
     TIME[Replay timestamp] --> FILTER[Keep CGM timestamp <= replay time]
     PROFILE[Explicit participant EHR] --> RUNTIME[BioSynthTwin]
     FILTER --> RUNTIME
     ARTIFACT --> RUNTIME
     RUNTIME --> STATE[Risk + Digital Twin state]
     STATE --> UI[Dashboard display]
   end
   FUTURE[Future CGM for retrospective story] -. never model input .-> UI
 ```

 ## 3. Repository ownership map

 | Layer | Main files | Responsibility |
 | --- | --- | --- |
 | Raw synthetic inputs | `data/02_synthetic/cgm/`, `data/02_synthetic/ehr/` | Store generated participant data. |
 | Feature construction | `scripts/build_synthetic_trajectory_features.py`, `scripts/build_ehr_cgm_fusion_features.py` | Turn raw data into model-ready numeric columns. |
 | Evaluation | `scripts/cross_validate_synthetic_models.py`, `scripts/cross_validate_ehr_cgm_fusion.py` | Measure models on participant-separated folds. |
 | Final training | `scripts/train_synthetic_final_model.py`, `scripts/train_ehr_cgm_final_model.py` | Fit artifacts used by the runtime. |
 | Runtime | `src/biosynth_twin.py` | Validate input, calculate features, call the saved model, and return state. |
 | Dashboard | `app.py` | Select mode, load SYN-0085, filter history, call runtime, and display state. |
 | Documentation | `docs/` | Explain decisions, contracts, results, and limitations. |

 ## 4. Why the fusion model has 29 features

 The model input is deliberately divided into two visible groups:

 - 18 features describe what glucose has been doing recently.
 - 11 features describe the participant's static synthetic health profile.

 Keeping the two groups explicit makes the system easier to inspect. It also makes it possible to compare the full fusion model against the CGM-only baseline without changing the CGM feature definitions.

 ## 5. Runtime safety boundaries

 The runtime enforces these decisions:

 - `person_id` identifies a participant in files but is not a predictive feature.
 - `simulation_phenotype` is context for analysis but is not a predictive feature.
 - Future CGM readings are filtered out before feature calculation.
 - Future event labels are never supplied to `BioSynthTwin`.
 - Fusion mode requires an EHR profile explicitly. Missing EHR does not silently trigger CGM-only inference.
 - The model's risk thresholds are display thresholds, not clinical thresholds.

 ## 6. Why the baseline is separate

 The CGM-only model answers a useful comparison question: “How much signal exists in glucose trajectory alone?” It is not blended with the fusion prediction. Reporting it separately prevents the dashboard from hiding whether EHR addition helped, did not help, or produced mixed results.

 ## 7. Evidence boundaries

 Hall 2018 is real CGM evidence for the general idea that trajectory features can carry predictive signal. It did not contain the synthetic EHR stream used here and therefore did not validate EHR + CGM fusion. Fusion results come from the synthetic BioSynth population and are labeled mixed: ROC-AUC improved, while overall PR-AUC declined slightly.

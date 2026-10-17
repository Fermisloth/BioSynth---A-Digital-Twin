 # BioSynth Model Card

 ## 1. Model identity

 The final fusion model is saved as:

 ```text
 models/biosynth_ehr_cgm_fusion_model.joblib
 models/biosynth_ehr_cgm_fusion_metadata.json
 ```

 It is a scikit-learn pipeline containing:

 ```text
 median imputation -> StandardScaler -> LogisticRegression
 ```

 The pipeline is intentionally simple. A simple model is easier to explain, fast enough for replay, and suitable for a proof of concept where the main question is whether two input streams can be connected correctly.

 ```mermaid
 flowchart LR
   CGM[18 CGM trajectory features] --> COMBINE[29-feature input vector]
   EHR[11 encoded EHR features] --> COMBINE
   COMBINE --> IMPUTE[Median imputation]
   IMPUTE --> SCALE[StandardScaler]
   SCALE --> LR[Logistic regression]
   LR --> PROB[Probability of new event within 120 minutes]
 ```

 ## 2. Intended use

 The model estimates the probability that a new glucose excursion above 180 mg/dL will begin within the next 120 minutes.

 It is intended for:

 - Demonstrating a synthetic EHR + CGM data path.
 - Demonstrating participant-separated model evaluation.
 - Demonstrating a historical-only Digital Twin replay.
 - Comparing CGM-only and EHR + CGM feature streams.

 It is not intended for diagnosis, treatment, medical decisions, patient care, or claims of clinical personalization.

 ## 3. Inputs

 The fusion model receives 29 numeric features:

 - 18 features from the most recent 60 minutes of CGM history.
 - 11 encoded fields from one static synthetic EHR profile.

 It does not receive `person_id`, `simulation_phenotype`, future CGM, future labels, simulator parameters, or event ground truth.

 ## 4. Target

 The target is `pre_event_target`.

 A positive target means a new sequence of readings above 180 mg/dL begins within the next 120 minutes. A timestamp already inside an ongoing event is not treated as a pre-event warning point.

 ## 5. Training and evaluation

 The training table contains 198,000 rows from 100 synthetic participants, with 23,696 positive rows and an 11.97% positive rate.

 Evaluation uses five-fold `GroupKFold` grouped by `person_id`. This means all rows from one participant stay together in one validation fold. Each out-of-fold prediction comes from a model that did not train on that participant.

 Preprocessing is fitted inside each training fold. The final artifact is then fitted on all 198,000 rows after evaluation is complete. The final fit is not itself a validation result.

 ## 6. Results

 | Model | PR-AUC | ROC-AUC |
 | --- | ---: | ---: |
 | Current-glucose baseline | 0.1918 | 0.7303 |
 | CGM trajectory | 0.4433 | 0.8539 |
 | EHR + CGM fusion | 0.4352 | 0.8729 |

 Overall interpretation: **MIXED RESULTS**.

 Fusion improved ROC-AUC by 0.0190 and improved precision at the displayed 0.50 threshold, but overall PR-AUC declined by 0.0081 compared with the CGM trajectory model. PR-AUC is especially important here because positive events are relatively uncommon.

 These results do not support a claim that fusion is always better.

 ```mermaid
 xychart-beta
   title "Evaluation metric comparison"
   x-axis ["Current glucose", "CGM trajectory", "EHR + CGM fusion"]
   y-axis "AUC" 0 --> 1
   bar [0.1918, 0.4433, 0.4352]
   line [0.7303, 0.8539, 0.8729]
 ```

 In this chart, the bars represent PR-AUC and the line represents ROC-AUC. The important reading is not that one model wins every metric: fusion has the best ROC-AUC, while CGM trajectory has the best PR-AUC.

 ## 7. Runtime behavior

 `BioSynthTwin(model_mode="EHR_CGM_FUSION")` requires a static EHR profile explicitly. The runtime:

 1. Validates and sorts CGM input.
 2. Keeps CGM rows at or before the prediction timestamp.
 3. Calculates the 18 CGM features from the 60-minute window.
 4. Encodes the supplied EHR into 11 features.
 5. Places all 29 features in metadata order.
 6. Calls `predict_proba` on the saved pipeline.
 7. Returns the probability plus state fields for the dashboard.

 If a required EHR field is missing, fusion mode raises an error. It does not silently switch to CGM-only mode.

 ## 8. Display states are not clinical states

 The runtime maps probability to display labels:

 - below 0.30: `LOW`
 - 0.30 to below 0.60: `ELEVATED`
 - 0.60 or higher: `HIGH`

 These thresholds are dashboard conventions. When current glucose is already above 180 mg/dL, the dashboard displays `Event underway` and `ABOVE POC THRESHOLD` instead of presenting the pre-event probability as ordinary future risk.

 ## 9. Data provenance and limitations

 - Hall 2018 is real CGM evidence for CGM trajectory signal only.
 - Synthetic CGM is simulator-generated.
 - Synthetic EHR is generated and phenotype-aligned.
 - Hall 2018 did not validate EHR + CGM fusion.
 - The synthetic population is not representative clinical data.
 - The model has no evidence of clinical safety, fairness, calibration, or treatment benefit.

 ## 10. Appropriate language

 Appropriate: “In this synthetic participant-separated POC evaluation, fusion improved ROC-AUC but produced mixed results overall.”

 Not appropriate: “The fusion model is clinically accurate,” “the model personalizes treatment,” or “fusion always outperforms CGM-only.”

# BioSynth Digital Twin: 10-Minute Prototype Video Script

## Purpose

This is a ready-to-record script for a ten-minute explanation of the BioSynth prototype. It is written for a judge or technically curious viewer who may understand ordinary language but may not know machine learning, CGM data, EHR data, or Digital Twin architecture.

The video should use the root `app.py` model-backed replay as its main demonstration. The separate `ui/app.py` command-center interface can be shown briefly as the broader interface, but it should not be presented as the same replay screen.

## Recording Rules

- Speak slowly and clearly: approximately 125 to 140 words per minute.
- Keep the dashboard visible while explaining the prototype, but zoom the browser so labels remain readable.
- Use the exact phrases “synthetic EHR”, “synthetic CGM”, and “proof of concept”.
- Say “estimated risk” rather than “diagnosis”, “prediction certainty”, or “clinical alert”.
- Show the model selector as `EHR + CGM Fusion` first, then briefly compare it with `CGM Only`.
- Do not imply that Hall 2018 validated EHR + CGM fusion. Hall 2018 is real CGM proof-of-signal only.
- The prototype is demonstrated with virtual participant `SYN-0085`.

## Before Recording

Start the root replay dashboard from the repository root:

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

Prepare these screens or assets:

1. The architecture diagram from `docs/BIOSYNTH_ARCHITECTURE.pdf` or `docs/BIOSYNTH_ARCHITECTURE.md`.
2. The root Streamlit replay with `EHR + CGM Fusion` selected.
3. The synthetic health profile for `SYN-0085`.
4. The guided replay controls.
5. The model selector showing `CGM Only` as the separate comparison mode.
6. The evaluation numbers from `docs/model_card.md`.

## Timeline At A Glance

| Time | Section | Main visual |
| --- | --- | --- |
| 00:00-00:40 | Opening | Title and dashboard |
| 00:40-01:30 | Problem | CGM timeline and early-warning idea |
| 01:30-02:25 | Two data streams | EHR profile and CGM stream |
| 02:25-03:15 | Prediction target | 180 mg/dL threshold and 120-minute horizon |
| 03:15-04:15 | Offline pipeline | Architecture diagram |
| 04:15-05:10 | Feature fusion | 18 CGM + 11 EHR = 29 features |
| 05:10-06:05 | Evaluation | Participant-separated metrics |
| 06:05-07:05 | Runtime safety | Historical-only replay rule |
| 07:05-08:35 | Live dashboard demo | Guided SYN-0085 replay |
| 08:35-09:25 | Baseline comparison | CGM-only versus fusion |
| 09:25-10:00 | Limitations and close | Honest interpretation |

## Full Narration And Screen Directions

### 00:00-00:40 — Opening: What BioSynth Is

**Screen:** Show the BioSynth dashboard. Keep the title and the synthetic-data notice visible.

**Narration:**

“This is BioSynth, a Digital Twin proof of concept for the Digital Twin Challenge 2026. BioSynth represents a virtual participant using two kinds of information: a changing continuous glucose monitor stream, and a static synthetic health profile that represents an electronic health record.

The system asks one focused question: based on what has happened so far, is a new glucose excursion above 180 milligrams per deciliter likely to begin within the next 120 minutes? This is a research prototype, not a medical device, and all EHR records shown here are synthetic.”

### 00:40-01:30 — The Problem: Turning Readings Into A State

**Screen:** Show the CGM chart and point to the changing glucose line. Then show the current glucose and risk cards.

**Narration:**

“A single glucose reading is useful, but it does not tell the whole story. A value of 170 could be stable, falling, or rising quickly. Those situations have different meanings for an early-warning system.

BioSynth therefore looks at a recent historical window rather than only the latest number. It describes the current level, recent changes, overall direction, variability, and how much of the window has been above the engineering threshold.

The Digital Twin state is the result of that update. It is a compact description of the virtual participant at one replay time: current glucose, trend, risk state, and the evidence used to produce it.”

### 01:30-02:25 — The Two Data Streams

**Screen:** Show the architecture diagram, then return to the dashboard’s synthetic health profile.

**Narration:**

“The first stream is dynamic synthetic CGM. It contains glucose readings every five minutes for one hundred virtual participants. This stream changes over time.

The second stream is static synthetic EHR. Each virtual participant has exactly one profile. The profile includes age, sex, BMI, HbA1c, fasting glucose, family history of diabetes, hypertension history, and activity level.

These are not real patient records. They were generated reproducibly for this proof of concept and aligned with the simulator’s participant context. The purpose is to demonstrate how a static health profile can be connected to a time-varying physiological stream without exposing private clinical data.

The participant identifier is used to connect the two sources, but it is not given to the model as a predictive feature.”

### 02:25-03:15 — What The Model Predicts

**Screen:** Show the event definition from the architecture diagram or the dashboard’s model-flow panel.

**Narration:**

“The target has a precise definition. A positive example means that a new sequence of glucose readings above 180 begins within the next 120 minutes.

The word new matters. If the participant is already above 180, the event is underway. That is not the same as warning that an event is about to begin. The dashboard handles this explicitly by showing ‘event underway’ instead of presenting the pre-event score as if it were a normal future warning.

The 180 threshold and the 120-minute horizon are engineering choices for this challenge prototype. They are not clinical treatment thresholds, and the dashboard’s low, elevated, and high labels are display conventions rather than medical classifications.”

### 03:15-04:15 — Offline Training And Evaluation

**Screen:** Show the Mermaid architecture diagram or PDF. Trace the offline pipeline from left to right.

**Narration:**

“Before the dashboard can replay a participant, the project performs offline preparation.

First, the synthetic CGM generator creates the virtual sensor streams. A separate script creates one synthetic EHR row per participant. Next, the project detects events and creates training labels. The recent CGM history is converted into 18 trajectory features, and the EHR profile is encoded into 11 numeric features.

The two feature groups are joined into a 29-feature fusion table. The project then evaluates the models using participant-separated folds. This is important because all rows from one participant stay together. The validation participant is therefore someone the model did not train on, rather than a later row from someone it has already seen.

Only after evaluation is complete is the final model trained on the accepted synthetic rows and saved as an artifact for the runtime.”

### 04:15-05:10 — How Fusion Works

**Screen:** Show the model card feature diagram. If possible, briefly display the feature-count or model-flow panel.

**Narration:**

“The model does not receive a paragraph of medical history. Machine-learning models need numeric inputs.

The 18 CGM features summarize recent glucose behavior. They include the current glucose, mean, median, minimum, maximum, range, changes over 15, 30, and 60 minutes, slope, variability, high-reading counts, time above 180, and the amount of usable history.

The 11 EHR features are created using a consistent encoding. Numeric fields remain numeric. Boolean history fields become zero or one. Sex becomes two indicator columns, and activity becomes three indicator columns for low, moderate, and high.

The model pipeline then performs median imputation if needed, standardizes the values, and applies logistic regression. Logistic regression is deliberately understandable and fast enough for interactive replay.”

### 05:10-06:05 — What The Evaluation Says

**Screen:** Show the evaluation table or the model-card chart.

**Narration:**

“The evaluation compares three levels of information. The current-glucose baseline has a PR-AUC of 0.1918 and a ROC-AUC of 0.7303. The CGM trajectory model improves to a PR-AUC of 0.4433 and a ROC-AUC of 0.8539.

The EHR plus CGM fusion model has a PR-AUC of 0.4352 and a ROC-AUC of 0.8729.

The correct conclusion is mixed. Fusion improves ROC-AUC by 0.0190, which means its overall ranking of positive and negative examples improves on that metric. However, PR-AUC decreases by 0.0081 compared with the CGM trajectory model. PR-AUC matters here because positive events are less common.

So we do not claim that fusion always wins. We claim that the prototype successfully connects both streams and produces a measurable, mixed evaluation result.”

### 06:05-07:05 — Historical-Only Runtime Safety

**Screen:** Show the architecture’s live inference pipeline. Then show the replay timestamp control.

**Narration:**

“The most important runtime rule is that the model may use only information available at or before the replay timestamp.

When the dashboard evaluates a timestamp, it filters the CGM table first. Any reading after that timestamp is excluded. The runtime then keeps the most recent 60 minutes, calculates the 18 CGM features, encodes the explicitly supplied EHR profile, orders all 29 fields according to the saved model metadata, and requests one probability.

Future CGM may be shown afterward in the dashboard’s retrospective ‘What happened next?’ story. That is useful for explaining the outcome, but it is never passed into the prediction call.

The fusion runtime also refuses to silently fall back to CGM-only mode if the EHR profile is missing. This keeps the experiment honest: the selected mode must match the inputs it actually receives.”

### 07:05-08:35 — Guided Dashboard Demonstration

**Screen:** In the root dashboard, select `EHR + CGM Fusion`, keep `GUIDED DEMO`, and step through the prepared SYN-0085 replay.

**Narration:**

“Now I’ll show the working replay. This demonstration uses virtual participant SYN-0085. The dashboard first loads that participant’s synthetic EHR profile and labels it clearly as synthetic.

At the early replay point, the current glucose is 176.5 milligrams per deciliter. It is still below the 180 threshold, so the system can display a pre-event risk state. The recent trajectory is already informative because the model sees how glucose has moved across the historical window, not just the latest value.

As we advance through the guided timeline, the dashboard updates the current glucose, trend, feature-derived state, and estimated 120-minute risk. The chart separates the history used by the model from the future values shown for explanation.

At the event timestamp, the current glucose is 242.2. The glucose excursion is now underway. The dashboard changes the interpretation to ‘above POC threshold’ and ‘event underway’. This is a deliberate semantic distinction: the model was designed to warn before a new event begins, not to pretend that an event already in progress is still a future warning.

The same participant can be replayed in CGM-only mode. In that mode, the synthetic profile remains visible for context, but only the 18 CGM trajectory features are sent to the comparison model.”

### 08:35-09:25 — Why The Baseline Is Separate

**Screen:** Switch the model selector briefly from `EHR + CGM Fusion` to `CGM Only`. Show the model-flow panel and feature count.

**Narration:**

“The CGM-only model is not blended into the fusion prediction. It is a separate baseline.

That separation answers a practical question: how much signal can be obtained from glucose trajectory alone, and what changes when static health context is added? Keeping the paths separate makes the comparison visible instead of hiding it inside one score.

The dashboard therefore makes the selected mode explicit. Fusion uses 29 features. CGM-only uses 18. Both modes preserve the historical-only timestamp rule and return a common Digital Twin state shape for the user interface.”

### 09:25-10:00 — Limitations And Closing

**Screen:** Show the synthetic-data notice, then the architecture or model-card limitations section.

**Narration:**

“BioSynth is a synthetic hackathon proof of concept. Hall 2018 provides real CGM proof-of-signal for trajectory modeling, but it does not validate the synthetic EHR fusion model. The EHR profiles and CGM trajectories used for fusion are generated virtual data.

The model is not clinically validated. It has not established clinical safety, calibration, fairness, treatment benefit, or real-world generalization. The 180 threshold, the 120-minute horizon, and the display labels are engineering choices for demonstration.

What the prototype does demonstrate is the complete two-stream architecture: static synthetic EHR, dynamic synthetic CGM, reproducible feature engineering, participant-separated evaluation, historical-only runtime inference, and an interactive Digital Twin dashboard.

That is the BioSynth prototype: a transparent, reproducible foundation for exploring how longitudinal physiological data and static health context could work together in a future validated system.”

## Final On-Screen Text

Display for the final five seconds:

```text
BioSynth Digital Twin
Synthetic EHR + Dynamic CGM
Historical-only 120-minute pre-event risk

Proof of concept. Not clinically validated.
```

## Claims Checklist

Before publishing, confirm that the video:

- calls the EHR and CGM data synthetic;
- states that Hall 2018 is CGM proof-of-signal only;
- shows the CGM-only model as a separate baseline;
- explains 18 CGM features plus 11 EHR features equals 29 fusion features;
- explains the timestamp filter and excludes future CGM from inference;
- reports the results as mixed;
- avoids saying the model diagnoses, treats, personalizes care, or is clinically accurate.

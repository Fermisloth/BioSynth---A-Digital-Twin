# BioSynth PowerPoint: 10-Minute Speaker Script

## Purpose

This is a slide-by-slide narration script for `Synth.pdf`, the BioSynth Digital Twin Challenge 2026 presentation. The deck contains 19 slides. The timings below add up to ten minutes and are designed for a calm presentation pace with short pauses while diagrams and screenshots are visible.

The script explains the prototype in accessible language while preserving the important boundaries:

- The EHR and CGM records used for fusion are synthetic.
- Hall 2018 provides real CGM proof-of-signal only; it does not validate EHR + CGM fusion.
- The dashboard is a research proof of concept, not a clinical device.
- The fusion result is mixed: ROC-AUC improves, while PR-AUC is slightly lower than CGM trajectory alone.
- The command-center intervention view is a counterfactual simulation, not treatment advice or real intervention.

## Timing Plan

| Slide | Topic | Time | Running time |
| ---: | --- | ---: | ---: |
| 1 | Title | 0:25 | 0:25 |
| 2 | Team | 0:20 | 0:45 |
| 3 | Problem | 0:35 | 1:20 |
| 4 | Real-CGM evidence | 0:30 | 1:50 |
| 5 | Existing approaches | 0:30 | 2:20 |
| 6 | Two-stream solution | 0:35 | 2:55 |
| 7 | Conceptual architecture | 0:40 | 3:35 |
| 8 | How it works | 0:40 | 4:15 |
| 9 | Beneficiaries | 0:25 | 4:40 |
| 10 | Initial development plan | 0:25 | 5:05 |
| 11 | Command center | 0:35 | 5:40 |
| 12 | Development timeline | 0:25 | 6:05 |
| 13 | Team roles and tools | 0:25 | 6:30 |
| 14 | Resources | 0:20 | 6:50 |
| 15 | Technology readiness | 0:25 | 7:15 |
| 16 | System and model detail | 0:55 | 8:10 |
| 17 | Literature gap | 0:35 | 8:45 |
| 18 | Laboratory access | 0:20 | 9:05 |
| 19 | Novelty and IP | 0:55 | 10:00 |

## Speaker Notes By Slide

### Slide 1 — BioSynth: Digital Twin Challenge 2026

**Time:** 00:00-00:25

**Visual action:** Pause on the title slide. Let the EHR and CGM icons register before speaking.

**Narration:**

“Hello. We are presenting BioSynth, our Digital Twin Challenge 2026 proof of concept. BioSynth combines a static electronic health record profile with a dynamic continuous glucose monitor stream to anticipate a new glucose excursion before it begins. Our goal is to make that prediction understandable, replayable, and honest about what the data can support.”

### Slide 2 — Team Details

**Time:** 00:25-00:45

**Narration:**

“The team is Ctrl+Sloth from G.L. Bajaj Institute of Technology and Management. The work combines data generation and modeling with runtime engineering, dashboard design, documentation, and presentation. We divided the work around the full path from synthetic data to an interactive Digital Twin experience.”

### Slide 3 — Glucose Spikes Are Noticed Too Late

**Time:** 00:45-01:20

**Visual action:** Point from the CGM curve to the 180 mg/dL threshold and the late event marker.

**Narration:**

“A CGM tells us where glucose is now, but an isolated reading does not tell us where it is heading. A threshold alert can fire only after glucose is already high. A current-glucose baseline has limited context. A CGM trend model can describe movement, but it may ignore the participant’s static health profile. And static health records alone do not show what is happening right now.

The practical goal is early warning: identify whether a new excursion above 180 milligrams per deciliter may begin within the next 120 minutes.”

### Slide 4 — Evidence: Is There Signal in Real CGM Data?

**Time:** 01:20-01:50

**Narration:**

“Before building the synthetic Digital Twin, we checked whether trajectory features contain signal in real CGM data. The Hall 2018 branch contains 89,329 feature rows from 57 participants. It provides proof-of-signal for CGM trajectory modeling.

This distinction is important: Hall 2018 is real CGM evidence, but it does not contain the synthetic EHR stream used in our fusion model. We do not claim that Hall 2018 validates EHR plus CGM fusion.”

### Slide 5 — Existing Approaches And Their Limits

**Time:** 01:50-02:20

**Narration:**

“We organized the problem around five limitations. Threshold alerts are reactive. A current-glucose baseline is a weak ranking signal. CGM-only trend models can ignore patient context. Static health records cannot see live physiology. And a generic dashboard can hide both model reasoning and system limits.

BioSynth addresses these gaps by combining the two streams, preserving the time axis, exposing the model features, and separating forecast, explanation, replay, and counterfactual exploration.”

### Slide 6 — BioSynth: A Two-Stream Digital Twin

**Time:** 02:20-02:55

**Visual action:** Trace the five numbered steps from left to right.

**Narration:**

“The solution has five steps. First, read the recent CGM history. Second, encode the static EHR profile. Third, fuse the two streams into 29 numeric features. Fourth, estimate the 120-minute pre-event risk. Fifth, show the updated Digital Twin state and its explanation.

The 29 features are 18 CGM trajectory features plus 11 encoded EHR features. The CGM-only model remains separate as a comparison baseline, so we can see what the static health context changes rather than hiding that comparison inside one score.”

### Slide 7 — Conceptual Architecture

**Time:** 02:55-03:35

**Visual action:** Point first to the offline panel, then the lock, then the live panel.

**Narration:**

“This architecture has two distinct halves. Offline, we generate synthetic CGM and synthetic EHR, construct event labels, create features, train logistic regression, and evaluate with GroupKFold by participant. This is where future values may be used to define the training answer, but not as predictive features.

Live, the user chooses a replay timestamp. The runtime keeps only CGM rows at or before that time and receives the participant’s EHR profile explicitly. It builds the same 29-feature vector, calls the saved model, and returns the risk and Digital Twin state. The lock in the middle represents the no-future-data boundary.”

### Slide 8 — How It Works

**Time:** 03:35-04:15

**Narration:**

“At a selected replay time, the system filters the past CGM readings, calculates a 60-minute trajectory summary, and encodes the EHR profile. The CGM features include level, changes over 15, 30, and 60 minutes, slope, variability, range, and time above the engineering threshold.

The EHR features include numeric values such as age, BMI, HbA1c, and fasting glucose, plus encoded history, sex, and activity fields. Logistic regression then produces a probability. If the current glucose is already above 180, the interface describes the event as underway rather than pretending it is still a future warning.”

### Slide 9 — Who Could Benefit

**Time:** 04:15-04:40

**Narration:**

“The intended users are researchers studying digital health, clinicians reviewing synthetic CGM patterns, people learning about glucose time series, and students or educators exploring machine learning workflows. The wider impact is earlier awareness, transparent AI demonstrations, and a reusable open pipeline.

For this stage, ‘benefit’ means benefit from understanding and evaluating the prototype. It does not mean proven clinical benefit.”

### Slide 10 — Initial Development Plan

**Time:** 04:40-05:05

**Narration:**

“The initial implementation plan follows the dependency order. Generate synthetic data, label events and build features, cross-validate the models, train final artifacts and the runtime, then build the dashboard and demonstration. Separating these steps makes the system inspectable: we can review the data, labels, features, metrics, artifacts, and user interface independently.”

### Slide 11 — Interaction With The Command Center

**Time:** 05:05-05:40

**Visual action:** Briefly name each of the six interface views.

**Narration:**

“The command center turns the pipeline into six user workflows. Dashboard Overview summarizes the current participant. The 2-Hour Predictive Horizon shows the forecast and uncertainty fan. Intervention Sandbox compares a baseline with a simulated counterfactual. Explainability shows coefficient-based feature attribution. Cohort and Event Replay explores detected events and lead time. Settings and Guided Tour exposes runtime and display details.

The design separates observation, prediction, explanation, replay, and scenario testing so the viewer can understand what the system is doing at each stage.”

### Slide 12 — Development Plan

**Time:** 05:40-06:05

**Narration:**

“The development plan moves from data and labels, to features and models, to runtime guards and tests, to the dashboard, and finally to validation and demo preparation. The most important engineering checkpoint is the historical-only rule: at runtime, future CGM must not enter the feature vector.”

### Slide 13 — Team Composition And Roles

**Time:** 06:05-06:30

**Narration:**

“The team roles cover the complete prototype. Data generation and modeling create the synthetic population and predictors. Evaluation checks participant-separated performance. Runtime and dashboard work turns the model into an interactive experience. Documentation records the architecture, assumptions, limitations, and reproducibility steps. The main tools are Python, pandas, scikit-learn, Streamlit, and Git.”

### Slide 14 — Resource Requirements

**Time:** 06:30-06:50

**Narration:**

“The prototype is intentionally accessible. It runs on a standard laptop with Python, scikit-learn, Streamlit, pandas, synthetic data, a Git repository, and ordinary development support. No specialized laboratory is required to build or demonstrate this synthetic proof of concept.”

### Slide 15 — Entry And Target TRL

**Time:** 06:50-07:15

**Narration:**

“We present this work as an experimental proof of concept using synthetic data. The prototype demonstrates an integrated architecture and a working interactive runtime. It should not be described as validated in laboratory conditions or as a clinically ready system. Any target readiness level shown here should be understood as a development aspiration, not evidence of clinical validation.”

### Slide 16 — System And Model Detail

**Time:** 07:15-08:10

**Visual action:** Point to the feature pipeline, then the metric table, then the risk thresholds.

**Narration:**

“The final model pipeline is intentionally simple: median imputation, standard scaling, and logistic regression. Simplicity helps us explain the feature path and keeps interactive replay fast.

The current-glucose baseline uses one feature and reaches PR-AUC 0.1918 and ROC-AUC 0.7303. The CGM trajectory model uses 18 features and reaches PR-AUC 0.4433 and ROC-AUC 0.8539. The EHR plus CGM fusion model uses 29 features and reaches PR-AUC 0.4352 and ROC-AUC 0.8729.

The correct label is mixed results. Fusion improves ROC-AUC by 0.0190, while PR-AUC is lower by 0.0081 than CGM trajectory alone. This is why we report the baseline separately and avoid saying that fusion always outperforms CGM-only.

The low, elevated, and high risk categories are demo display thresholds. They are not clinical risk categories.”

### Slide 17 — Literature Survey

**Time:** 08:10-08:45

**Visual action:** Pause on the research/gaps columns. Do not read unresolved placeholders.

**Narration:**

“The literature direction supports two observations. First, CGM time series can support near-term event prediction. Second, many approaches use CGM alone or have limited EHR context. The opportunity is to connect static EHR context with live CGM while preserving the time boundary.

Our contribution at this stage is an implementation and evaluation pattern: explicit-profile EHR plus CGM fusion, historical-only runtime filtering, participant-separated evaluation, and transparent reporting of mixed results.”

### Slide 18 — Laboratory And Fabrication Access

**Time:** 08:45-09:05

**Narration:**

“Because this phase uses generated data and software simulation, it does not require specialized laboratory or fabrication access. The next validation stages would require broader clinical data, appropriate governance, calibration, safety review, and real-world testing. Those are future requirements, not capabilities claimed by this prototype.”

### Slide 19 — Novelty And IP Potential

**Time:** 09:05-10:00

**Visual action:** Read the three differentiation points, then pause on the IP disclaimer.

**Narration:**

“The prototype’s differentiation is the combination of three design decisions. First, an explicit-profile EHR plus CGM runtime rather than an implicit or hidden patient lookup. Second, a strict historical-only guard, where the runtime accepts only CGM timestamps at or before the replay time. Third, honest participant-separated evaluation with both PR-AUC and ROC-AUC reported, including the fact that the fusion result is mixed.

The reusable asset is the connected pipeline: synthetic data generation, feature construction, saved model artifacts, runtime checks, dashboard workflows, explainability, cohort replay, and documentation.

This slide says patent protection is not claimed. We should also describe the work as preliminary review rather than established intellectual property. The strongest immediate outcome is a transparent research prototype that can be inspected, reproduced, and extended.

To close: BioSynth shows how a Digital Twin can move from data, to model, to runtime, to understandable interaction. It does not claim clinical accuracy today. It demonstrates the architecture needed to make future validation possible.”

## Slide-Deck Corrections Before Recording

The following visible placeholders or claims should be corrected in the PowerPoint before presenting:

1. Replace `Validated with [number] user sessions` on the command-center slide with a verified number, or remove the phrase.
2. Replace `[Author, year]` entries on the literature slide with real citations, or remove those placeholders.
3. Review the `Target: TRL 4` wording. Keep it clearly labeled as a target or development aspiration unless laboratory validation has actually occurred.
4. Keep the synthetic proof-of-concept disclaimer visible on every slide or in the footer.
5. Do not read “clinical” labels as evidence of clinical validation. Explain that they are interface language for synthetic research data.
6. Keep the metric values synchronized with the model card: CGM trajectory PR-AUC 0.4433, ROC-AUC 0.8539; fusion PR-AUC 0.4352, ROC-AUC 0.8729.

## Closing On-Screen Text

```text
BioSynth Digital Twin
Static synthetic EHR + dynamic synthetic CGM
Transparent, historical-only, reproducible prototype

Synthetic proof of concept. Not clinically validated.
```

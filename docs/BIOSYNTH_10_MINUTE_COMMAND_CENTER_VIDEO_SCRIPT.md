# BioSynth Command Center: 10-Minute Video Script

## What This Script Covers

This script matches the current BioSynth command-center interface shown in the supplied screenshots. It explains the following visible capabilities:

1. Dashboard Overview
2. 2-Hour Predictive Horizon
3. Intervention Sandbox
4. Explainability & Protocol
5. Cohort & Event Replay
6. Settings & Guided Tour

The command-center is a synthetic research prototype. Its participant names, clinical profiles, CGM readings, forecasts, event records, and intervention scenarios are demonstration data. Do not describe the interface as a clinical system or the counterfactual sandbox as administering real treatment.

## Launch The Interface

From the repository root:

```powershell
.\.venv\Scripts\streamlit.exe run ui\app.py
```

Turn on **Presentation Mode** before recording if you want the cleanest judge-facing layout. Use the left navigation to move through the six views.

## Recording Plan

| Time | View | What the audience learns |
| --- | --- | --- |
| 00:00-00:45 | Opening | What BioSynth is and what synthetic means |
| 00:45-02:00 | Dashboard Overview | The current virtual participant state |
| 02:00-03:30 | Predictive Horizon | The two-hour forecast and uncertainty |
| 03:30-05:00 | Intervention Sandbox | Counterfactual scenario testing |
| 05:00-06:20 | Explainability | Which features influence model output |
| 06:20-07:50 | Cohort & Event Replay | Population exploration and lead time |
| 07:50-08:50 | Settings & Guided Tour | Runtime status and reproducibility |
| 08:50-10:00 | Boundaries and conclusion | What the prototype proves and does not prove |

## Full Narration And Screen Directions

### 00:00-00:45 — Opening

**Screen:** Start on the BioSynth hero or Dashboard Overview. Keep the synthetic-data notice visible.

**Narration:**

“This is the BioSynth Digital Twin command center. It is a research proof of concept for proactive glycemic monitoring using synthetic participant data.

The interface combines recent glucose telemetry with a virtual participant profile. It then presents the participant’s current state, a two-hour forward projection, event replay, explainable model information, and counterfactual intervention scenarios.

Everything shown here is synthetic demonstration data. The names, EHR-like fields, glucose readings, forecasts, and clinical language are designed to demonstrate the architecture and user experience. This is not a medical device and it must not be used to guide care.”

### 00:45-02:00 — Dashboard Overview

**Screen:** Select **Dashboard Overview**. Show the subject card and the five summary cards.

**Narration:**

“The Dashboard Overview begins with the selected virtual participant. In this example, the profile shows a synthetic subject, their age, sex, diabetes context, HbA1c, BMI, fasting glucose, hypertension history, family history, and active medication fields.

The profile gives the time-series data a human-readable context. It is still a generated profile, not a real patient record.

Below it are the current summary signals. Latest glucose shows the most recent sensor value. Window average summarizes the recent period rather than one isolated reading. Glycemic variability is shown using coefficient of variation, or CV, which describes how widely readings move relative to their average.

Time in target reports the proportion of the visible window inside the project’s target zones. The two-hour projected peak is the highest estimated glucose in the forward horizon.

Together these cards answer three simple questions: where is the participant now, how stable has the recent trajectory been, and what may happen over the next two hours?”

### 02:00-03:30 — Two-Hour Predictive Horizon

**Screen:** Select **2-Hour Predictive Horizon**. Point to the observed line, dotted forecast, uncertainty fan, threshold, and prediction timestamp.

**Narration:**

“The predictive horizon turns the current state into a forward-looking view. The solid line is observed participant telemetry. The dotted line is the BioSynth forecast for eight future steps at fifteen-minute intervals, covering two hours.

The shaded ribbon is an uncertainty fan. It reminds us that a forecast is not a single guaranteed future value. The prototype reports an estimated peak, a spike probability, and a confidence interval at the end of the horizon.

The 180 milligrams per deciliter line is the project’s engineering threshold for a hyperglycemic excursion. It is used to explain the demo target, not as a treatment instruction.

This view is useful because it connects the machine-learning result to a timeline. Instead of showing only a percentage, it shows the expected path, the possible range around that path, and the point in time at which the forecast begins.”

### 03:30-05:00 — Intervention Sandbox

**Screen:** Select **Intervention Sandbox**. Show the preset buttons, sliders, baseline line, counterfactual line, and outcome cards.

**Narration:**

“The Intervention Sandbox is a counterfactual exploration tool. It lets us ask, ‘What would the projected trajectory look like under a simulated scenario?’

The interface provides presets such as Standard Lunch, Corrective Bolus, Post-Meal Walk, and Carb Surge. It also exposes scenario parameters for carbohydrate intake, rapid-acting insulin amount, physical activity minutes, and baseline medication adherence.

The red dashed line is the baseline forecast with no simulated intervention. The blue line is the counterfactual trajectory generated from the selected scenario. The cards below compare peak glucose, simulated spike risk, and projected status.

This is not prescribing treatment, administering insulin, or predicting what a real patient will experience. It is a controlled ‘what if’ visualization built to demonstrate how a Digital Twin could support scenario testing before a future system is validated with clinical data and appropriate safety controls.

The value of this screen is conceptual: it shows how a model can move from observation, to forecast, to a transparent comparison of possible trajectories.”

### 05:00-06:20 — Explainable AI And Feature Attribution

**Screen:** Select **Explainability & Protocol**. Show the horizontal attribution chart and the clinical decision-support protocol below it.

**Narration:**

“The Explainability view answers the question, ‘Why did the model produce this risk estimate?’

The chart displays the strongest feature contributions for the selected state. In this example, current high status above 180, current interstitial glucose, and glycemic standard deviation push the log-odds toward higher risk. The 60-minute window range and coefficient of variation contribute in the opposite direction, while the window maximum contributes positively.

Red bars indicate positive contribution toward the model’s hyperglycemia-risk score. Blue bars indicate negative contribution. The size of each bar shows relative contribution in standardized log-odds units.

This chart is derived from the saved logistic-regression pipeline’s feature weights and the current sample values. It is explicitly not a SHAP explanation. That distinction matters because the dashboard should describe the calculation it actually performs.

The protocol panel translates the technical result into a cautious decision-support summary. It is an explanation layer for a synthetic prototype, not an autonomous clinical recommendation.”

### 06:20-07:50 — Cohort Explorer And Event Lead-Time Replay

**Screen:** Select **Cohort & Event Replay**. Show the event selector, participant metadata, replay chart, lead-time slider, and risk cards.

**Narration:**

“The Cohort Explorer moves from one participant to the wider synthetic population. BioSynth can browse the one-hundred-participant cohort and select a detected hyperglycemic event.

The event panel identifies the virtual participant, phenotype context, event onset, peak glucose, and event duration. The chart then replays the trajectory relative to the event.

The lead-time scrubber is especially important. Moving it to minus 60 minutes asks what the model would have shown one hour before the event began. The dashboard can then display the model spike risk, sensor state, and observed trend at that lead time.

This view is not only a demonstration screen. It also makes the evaluation idea intuitive: a useful early-warning model should produce meaningful signal before the event, not only after glucose has already crossed the threshold.

Because the cohort and events are synthetic, the replay demonstrates system behavior and analysis workflow rather than proving clinical performance.”

### 07:50-08:50 — Settings And Guided Tour

**Screen:** Select **Settings & Guided Tour**. Show runtime details, model specifications, and presentation mode.

**Narration:**

“The Settings and Guided Tour view makes the prototype easier to inspect. It exposes the active interface settings, presentation mode, data source information, and model specifications.

This is useful during a demonstration because it makes the execution path visible. The audience can see whether the interface is using the available model-backed path or a calibrated simulation fallback.

The navigation also includes controls for subject selection, browsing the synthetic cohort, choosing historical windows, and adjusting chart display options. These controls turn the dashboard from a static screenshot into a reusable exploration environment.”

### 08:50-10:00 — Boundaries And Conclusion

**Screen:** Return to Dashboard Overview or the hero. Show the synthetic-data notice and the key metrics if available.

**Narration:**

“BioSynth demonstrates a complete Digital Twin interaction loop: observe the virtual participant, summarize the current state, project the next two hours, explore counterfactual scenarios, explain important model features, and replay events across a synthetic cohort.

There are important boundaries. The participant records are synthetic. The 180 threshold, risk labels, forecast intervals, and intervention responses are engineering choices for a proof of concept. The command-center’s simulation path must not be described as live clinical inference.

The broader project also contains a separate model-backed replay for synthetic EHR plus CGM fusion, with 18 CGM trajectory features and 11 encoded EHR features. Hall 2018 provides real CGM proof-of-signal for trajectory modeling, but it does not validate the synthetic EHR fusion model.

The current evaluation is mixed: CGM trajectory reaches PR-AUC 0.4433 and ROC-AUC 0.8539, while EHR plus CGM fusion reaches PR-AUC 0.4352 and ROC-AUC 0.8729. That means fusion improves ROC-AUC but not every metric.

The achievement of this prototype is not a claim of clinical accuracy. It is a transparent, reproducible foundation showing how synthetic health context, glucose time series, forecasting, explanation, cohort replay, and counterfactual exploration can be brought together in one Digital Twin experience.”

## Final On-Screen Text

```text
BioSynth Digital Twin Command Center
Observe. Forecast. Explain. Replay. Explore.

Synthetic research prototype
Not clinically validated. Not medical advice.
```

## Screen-Recording Checklist

- Start `ui/app.py`, not the root replay, for this script.
- Enable Presentation Mode if it improves readability.
- Show the subject profile and label it synthetic.
- Show the observed telemetry and forecast separately.
- Explain the uncertainty fan rather than presenting the forecast as certain.
- Call the Intervention Sandbox counterfactual simulation, not treatment.
- Say that feature attribution is coefficient-based and not SHAP.
- Show the lead-time slider before the event, not only after the spike.
- Distinguish the command-center simulation path from the root model-backed fusion replay.
- Include the mixed evaluation result and the synthetic-data limitation in the final minute.

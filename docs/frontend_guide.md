# Frontend guide

This guide covers the BioSynth command-center interface in `ui/app.py`. The root-level `app.py` is a separate model-backed guided event replay; see the [technical guide](BIOSYNTH_TECHNICAL_GUIDE.md) for the distinction.

## Launch

From the repository root, create and activate the project environment, then install the pinned requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run ui/app.py
```

When the virtual environment is not activated on Windows, use:

```powershell
.\.venv\Scripts\streamlit.exe run ui\app.py
```

The command-center interface starts without trained model binaries. It attempts to load the CGM-only model and displays its actual runtime status. If model loading is unavailable, some forecast and risk displays use the calibrated simulation path. The dashboard indicates which path is active; simulation output must not be described as live model inference.

## Dashboard views

The sidebar selects one of six views:

| View | What it displays |
| --- | --- |
| Dashboard Overview | Selected subject, glucose summary metrics, alert state, and telemetry chart. |
| 2-Hour Predictive Horizon | Forecast trajectory and uncertainty visualization. |
| Intervention Sandbox | Counterfactual curves adjusted with the existing intervention controls and presets. |
| Explainability & Protocol | Model feature attribution view when a live model is available, plus supporting explanation. |
| Cohort & Event Replay | Cohort data and timeline replay when the corresponding generated files are available. |
| Settings & Guided Tour | Display settings, runtime details, and guided-tour content. |

The sidebar also provides subject/cohort selection, a historical window, chart display toggles, and Presentation Mode. The dashboard reads synthetic cohort and event files from `data/02_synthetic/` and `reports/` when those files exist. Demo profiles and simulation are used for the command-center's display path when appropriate.

## Visual system

The current interface uses a soft gray-lilac canvas, white hero and content surfaces, violet accents, and a restrained glass treatment for data cards. The large DNA illustration appears in the hero. A compact DNA helix sits beside the BioSynth wordmark in the hero and centered in the sidebar.

| Element | Implementation |
| --- | --- |
| Streamlit layout, navigation, data and charts | `ui/app.py` |
| Theme variables, layout, glass effects, responsive rules, transitions and cursor | `ui/assets/style.css` |
| DNA wordmark and hero artwork | `ui/assets/dna-helix.svg`, `ui/assets/dna-helix-still.svg` |
| Streamlit theme defaults | `.streamlit/config.toml` |

Prefer editing CSS classes and the theme file for visual changes. Keep Streamlit widgets as the owners of their state and behavior. Custom HTML in `ui/app.py` is limited to visual/semantic presentation blocks such as the hero and brand lockups.

## Changing the interface safely

1. Read the relevant section in `ui/app.py` and the matching selectors in `ui/assets/style.css`.
2. Keep current widgets, keys, state updates, and callbacks intact when changing presentation.
3. Check the desktop and narrow layouts. The CSS uses breakpoints at 1100px and 640px.
4. Preserve the `prefers-reduced-motion` styles when adding motion.
5. Keep clinical labels and generated/simulated values explicit. Visual wording must not imply clinical validation.

Useful implementation details:

- Dashboard navigation uses `NAVIGATION_OPTIONS` and `st.session_state["nav_selection"]` in `ui/app.py`.
- Chart options and Presentation Mode are Streamlit sidebar controls.
- The 100-person cohort browser depends on the generated EHR CSV; the event replay view also uses the detected-events report.
- The hero's Explore link targets the existing patient workspace anchor.
- The model explanation view reads the estimator pipeline and presents coefficient-based attribution; it labels these as not SHAP values.

## Brand assets

The sidebar uses the still SVG for a stable compact mark. The hero lockup uses the animated helix SVG; the reduced-motion preference selects the still SVG for the large hero art. Keep the helix's violet palette consistent with the interface accents. Legacy PNG logos remain in `ui/assets/` for fallback and historical use.

## Current caveats

- Some names and labels in the UI use clinical language for a demonstration. The participant profiles and generated cohort records are synthetic.
- The command-center currently initializes the backend in `CGM_ONLY` mode. The root `app.py` is the guided event replay that exposes CGM-only and EHR + CGM fusion selection.
- `st.cache_data` and `st.cache_resource` control data and runtime caching; clear the Streamlit cache when local generated data or model artifacts change.
- Do not remove or rename navigation choices, widget keys, chart toggles, presets, cohort selection, downloads, or accessibility/reduced-motion behavior as part of a visual update.

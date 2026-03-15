<div align="center">

# Kaggle ML Pipeline
### End-to-end tabular ML system with reproducible training, deployable artifacts, and live prediction apps

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#)
[![Framework](https://img.shields.io/badge/ML-scikit--learn%20%7C%20XGBoost%20%7C%20LightGBM%20%7C%20CatBoost-orange)](#)
[![Tracking](https://img.shields.io/badge/Tracking-Weights%20%26%20Biases-yellow)](#)
[![App](https://img.shields.io/badge/UI-Streamlit-red)](#)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)

</div>

---

## Overview

This repository is a modular machine learning pipeline designed for **Kaggle-style tabular competitions** with a focus on **fast iteration**, **reproducibility**, and **portfolio-ready deployment**. The current implementation targets a **personality classification workflow** and includes reusable components for feature engineering, model training, inference, ensembling, experiment logging, model explainability, and a lightweight Streamlit dashboard.

---

## Project Snapshot

- Built an end-to-end ML product workflow from data download to UI inference.
- Added a reproducible training entrypoint that generates deployable artifacts.
- Integrated a user-facing Streamlit app and a Gradio endpoint for quick demos.
- Structured the project with CI, tests, Docker files, and clear separation of concerns.
- Applied practical security hygiene for local data and credential handling.

## Key Outcomes

- One-command pipeline execution with deterministic output artifacts for inference.
- Feature engineering logic reused across training and app prediction flows.
- Download automation for Kaggle data with secure local credential handling.
- Interview-ready product demo through Streamlit and Gradio interfaces.
- Test coverage for feature transformation and pipeline integration paths.

## Why This Project Stands Out

- **Business framing:** Converts behavioral inputs into an instant personality prediction.
- **Engineering quality:** Modular source layout, reusable feature transformer, and clean utility boundaries.
- **Operational readiness:** One-command data download script, deterministic output artifacts, and app-ready model files.
- **Demo readiness:** Local UI can be shown live during interviews and recruiter screens.

## Skills Demonstrated

- Machine learning pipeline design for tabular classification
- Feature engineering and schema-aligned inference
- Python packaging with src layout and reusable modules
- Debugging, test-driven fixes, and compatibility hardening
- Product-minded ML delivery with deployable web interfaces

---

## Project structure

```text
kaggle-ml-pipeline/
├── .github/
│   └── workflows/
│       └── run_pipeline.yml
├── apps/
│   ├── streamlit_app.py
│   └── gradio_api.py
├── reports/
│   └── pipeline_report.html
├── src/
│   ├── main.py
│   └── kaggle_ml_pipeline/
│       ├── __init__.py
│       ├── pipeline.py
│       ├── features/
│       │   └── tabular.py
│       ├── models/
│       │   ├── train.py
│       │   └── infer.py
│       └── utils/
│           ├── exception_handling.py
│           ├── kaggle_submit.py
│           ├── logging.py
│           └── shap_visuals.py
├── tests/
│   ├── test_feature_engineering.py
│   └── test_integration.py
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── setup.py
├── requirements.txt
└── README.md
```

## Architecture

### Training pipeline

1. Load tabular competition data
2. Apply feature engineering and transformation pipeline
3. Train a runnable baseline classifier and export model artifacts
4. Run cross-validation and track out-of-fold metrics
5. Blend or stack model predictions
6. Log metrics, feature importance, and SHAP artifacts to W&B
7. Export trained artifacts for local inference and app deployment

The repository includes experimentation references for XGBoost, LightGBM, CatBoost, stacking, Optuna search, W&B logging, and SHAP analysis.

### Serving layer

The repo includes:

- a **Streamlit dashboard** for interactive personality prediction
- a **Gradio interface** for lightweight API-style interaction
- Docker support through `Dockerfile` and `docker-compose.yml`

The dashboard loads `output/stack_model.pkl` and `output/feature_pipeline.pkl`, accepts behavioral features from the Kaggle dataset schema, and returns an introvert/extrovert prediction with confidence.

---

## Product Demo

### Streamlit application

The Streamlit app is now aligned with the trained model schema and supports live prediction using these input fields:

- `Time_spent_Alone`
- `Stage_fear`
- `Social_event_attendance`
- `Going_outside`
- `Drained_after_socializing`
- `Friends_circle_size`
- `Post_frequency`

### Generated artifacts

Running the pipeline writes these files for deployment-ready inference:

- `output/submission.csv`
- `output/stack_model.pkl`
- `output/feature_pipeline.pkl`

## Live Demo

The Streamlit app is interactive: choose the input options and click **Predict Personality** to generate a result with confidence.

- Current live preview (Codespaces, temporary): [Open App](https://shiny-computing-machine-vrpv5w5xpvjfprvg-8501.app.github.dev/)
- Stable public Streamlit URL (replace after deployment): [Open App](https://your-app-name.streamlit.app)
- Gradio endpoint (optional): [Open API Demo](https://your-space.hf.space)

> Note: the Codespaces preview link can expire, rotate, or require login. Keep it as a temporary demo only.
> Known limitation: after restarting Codespaces, the preview URL may change. If it stops working, relaunch the app and update the README link.

## Deploy to Streamlit Cloud (Stable Link)

1. Push your latest code to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io) and sign in with GitHub.
3. Create a new app and select this repo/branch:
	- Repo: `Chirag314/kaggle-ml-pipeline`
	- Branch: `main`
	- Main file: `apps/streamlit_app.py`
4. Click **Deploy** and copy your stable URL.
5. Replace the placeholder link above with your real Streamlit URL.

## Local Demo (No Public Deployment Yet)

If you have not deployed publicly, reviewers can still run the app locally:

```bash
streamlit run apps/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
```

Then open `http://localhost:8501` (or use your development environment's forwarded port URL).

## Quickstart

```bash
git clone https://github.com/Chirag314/kaggle-ml-pipeline.git
cd kaggle-ml-pipeline
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download competition data into the local data folder:

```bash
export KAGGLE_USERNAME=your_kaggle_username
export KAGGLE_KEY=your_kaggle_api_key
python download_data.py
```

Run the Streamlit dashboard:

```bash
streamlit run apps/streamlit_app.py
```

Run the training pipeline before launching apps:

```bash
PYTHONPATH=src python src/main.py
```

Run the API locally:

```bash
python apps/gradio_api.py
```


## License

This repository includes an MIT license in the public GitHub repo.

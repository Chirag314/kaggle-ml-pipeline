<div align="center">

# Kaggle ML Pipeline
### Reusable competition pipeline for fast iteration, feature engineering, ensembling, experiment tracking, and lightweight app deployment

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
3. Train multiple base learners such as XGBoost, LightGBM, and CatBoost
4. Run cross-validation and track out-of-fold metrics
5. Blend or stack model predictions
6. Log metrics, feature importance, and SHAP artifacts to W&B
7. Export trained artifacts for local inference and app deployment

The raw code references XGBoost, LightGBM, CatBoost, stacking, blend-weight search, Optuna-style optimization, W&B logging, and SHAP analysis. 

### Serving layer

The repo includes:

- a **Streamlit dashboard** for interactive personality prediction
- a **Gradio API** path mentioned in the README
- Docker support through `Dockerfile` and `docker-compose.yml`

The dashboard currently loads `output/stack_model.pkl` and `output/feature_pipeline.pkl`, accepts user inputs such as age and trait scores, and returns an introvert/extrovert prediction with confidence. 

---

## Streamlit application

```md
## Live Demo
- Streamlit app: _Add deployed Streamlit URL here after publishing_
```

Once deployed, replace that placeholder with your actual app URL and use a clickable link such as:

```md
- Streamlit dashboard: [Open App](https://your-app-name.streamlit.app)
```

## Quickstart

```bash
git clone https://github.com/Chirag314/kaggle-ml-pipeline.git
cd kaggle-ml-pipeline
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the Streamlit dashboard:

```bash
streamlit run apps/streamlit_app.py
```

Run the API locally:

```bash
python apps/gradio_api.py
```


## License

This repository includes an MIT license in the public GitHub repo.

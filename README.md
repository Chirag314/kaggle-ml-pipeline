# Kaggle Personality Classification Project

This repository contains a modular machine learning pipeline for feature engineering, training, and prediction.

- Uses W&B for tracking
- Supports ensembling and SHAP
- Auto-submits predictions on Kaggle

---

## 🌐 Deployment Instructions

### 📊 Streamlit Cloud Dashboard

1. Push your repo to GitHub.
2. Go to https://streamlit.io/cloud and create a new app.
3. Select:
   - **Repository:** your GitHub repo
   - **App file:** `src/dashboard.py`
4. Click "Deploy"

---

### 🤖 Gradio Web API

Run locally:

```bash
pip install gradio
python src/api.py
```

Or deploy via Hugging Face Spaces by:
- Creating a new space with type `Gradio`
- Pushing this project

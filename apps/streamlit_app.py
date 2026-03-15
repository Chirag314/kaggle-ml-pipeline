
import streamlit as st
import pandas as pd
import joblib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

st.title("Personality Prediction Dashboard")

# Load trained model and feature generator
stack_model = joblib.load("output/stack_model.pkl")
feature_gen = joblib.load("output/feature_pipeline.pkl")

st.markdown("### Input your personality traits")

time_spent_alone = st.slider("Time_spent_Alone", 0.0, 11.0, 3.0, 0.5)
stage_fear = st.selectbox("Stage_fear", ["No", "Yes"])
social_event_attendance = st.slider("Social_event_attendance", 0.0, 10.0, 5.0, 0.5)
going_outside = st.slider("Going_outside", 0.0, 7.0, 3.0, 0.5)
drained_after_socializing = st.selectbox("Drained_after_socializing", ["No", "Yes"])
friends_circle_size = st.slider("Friends_circle_size", 0.0, 15.0, 5.0, 0.5)
post_frequency = st.slider("Post_frequency", 0.0, 10.0, 4.0, 0.5)

if st.button("Predict Personality"):
    try:
        input_df = pd.DataFrame(
            [
                {
                    "Time_spent_Alone": time_spent_alone,
                    "Stage_fear": stage_fear,
                    "Social_event_attendance": social_event_attendance,
                    "Going_outside": going_outside,
                    "Drained_after_socializing": drained_after_socializing,
                    "Friends_circle_size": friends_circle_size,
                    "Post_frequency": post_frequency,
                }
            ]
        )
        X_trans = feature_gen.transform(input_df)
        proba = stack_model.predict_proba(X_trans)[0][1]
        label = "Extrovert" if proba >= 0.5 else "Introvert"
        st.success(f"Predicted Personality: **{label}** (Confidence: {proba:.2%})")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

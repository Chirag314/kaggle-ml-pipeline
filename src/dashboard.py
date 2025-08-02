
import streamlit as st
import pandas as pd
import joblib

st.title("Personality Prediction Dashboard")

# Load trained model and feature generator
stack_model = joblib.load("output/stack_model.pkl")
feature_gen = joblib.load("output/feature_pipeline.pkl")

st.markdown("### Input your personality traits")

age = st.number_input("Age", min_value=10, max_value=100, value=25)
openness = st.slider("Openness", 0.0, 1.0, 0.5)
neuroticism = st.slider("Neuroticism", 0.0, 1.0, 0.5)
conscientiousness = st.slider("Conscientiousness", 0.0, 1.0, 0.5)
agreeableness = st.slider("Agreeableness", 0.0, 1.0, 0.5)
impulsiveness = st.slider("Impulsiveness", 0.0, 1.0, 0.5)

if st.button("Predict Personality"):
    input_df = pd.DataFrame([{
        "Age": age,
        "Openness": openness,
        "Neuroticism": neuroticism,
        "Conscientiousness": conscientiousness,
        "Agreeableness": agreeableness,
        "Impulsiveness": impulsiveness
    }])
    X_trans = feature_gen.transform(input_df)
    proba = stack_model.predict_proba(X_trans)[0][1]
    label = "Extrovert" if proba >= 0.5 else "Introvert"
    st.success(f"Predicted Personality: **{label}** (Confidence: {proba:.2%})")

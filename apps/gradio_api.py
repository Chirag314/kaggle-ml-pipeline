
import gradio as gr
import pandas as pd
import joblib

stack_model = joblib.load("output/stack_model.pkl")
feature_gen = joblib.load("output/feature_pipeline.pkl")

def predict_personality(age, openness, neuroticism, conscientiousness, agreeableness, impulsiveness):
    input_df = pd.DataFrame([{
        "Age": age,
        "Openness": openness,
        "Neuroticism": neuroticism,
        "Conscientiousness": conscientiousness,
        "Agreeableness": agreeableness,
        "Impulsiveness": impulsiveness
    }])
    X = feature_gen.transform(input_df)
    proba = stack_model.predict_proba(X)[0][1]
    label = "Extrovert" if proba >= 0.5 else "Introvert"
    return f"{label} (Confidence: {proba:.2%})"

iface = gr.Interface(
    fn=predict_personality,
    inputs=[
        gr.Number(label="Age"),
        gr.Slider(0, 1, step=0.01, label="Openness"),
        gr.Slider(0, 1, step=0.01, label="Neuroticism"),
        gr.Slider(0, 1, step=0.01, label="Conscientiousness"),
        gr.Slider(0, 1, step=0.01, label="Agreeableness"),
        gr.Slider(0, 1, step=0.01, label="Impulsiveness")
    ],
    outputs=gr.Textbox(label="Predicted Personality"),
    title="Personality Predictor",
    description="Enter personality traits and see predicted type"
)

if __name__ == "__main__":
    iface.launch()

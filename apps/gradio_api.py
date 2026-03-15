
import gradio as gr
import pandas as pd
import joblib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

stack_model = joblib.load("output/stack_model.pkl")
feature_gen = joblib.load("output/feature_pipeline.pkl")

def predict_personality(
    time_spent_alone,
    stage_fear,
    social_event_attendance,
    going_outside,
    drained_after_socializing,
    friends_circle_size,
    post_frequency,
):
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
        X = feature_gen.transform(input_df)
        proba = stack_model.predict_proba(X)[0][1]
        label = "Extrovert" if proba >= 0.5 else "Introvert"
        return f"{label} (Confidence: {proba:.2%})"
    except Exception as exc:
        return f"Prediction failed: {exc}"

iface = gr.Interface(
    fn=predict_personality,
    inputs=[
        gr.Slider(0.0, 11.0, value=3.0, step=0.5, label="Time_spent_Alone"),
        gr.Radio(["No", "Yes"], value="No", label="Stage_fear"),
        gr.Slider(0.0, 10.0, value=5.0, step=0.5, label="Social_event_attendance"),
        gr.Slider(0.0, 7.0, value=3.0, step=0.5, label="Going_outside"),
        gr.Radio(["No", "Yes"], value="No", label="Drained_after_socializing"),
        gr.Slider(0.0, 15.0, value=5.0, step=0.5, label="Friends_circle_size"),
        gr.Slider(0.0, 10.0, value=4.0, step=0.5, label="Post_frequency"),
    ],
    outputs=gr.Textbox(label="Predicted Personality"),
    title="Personality Predictor",
    description="Enter behavioral traits and get predicted personality type"
)

if __name__ == "__main__":
    iface.launch()

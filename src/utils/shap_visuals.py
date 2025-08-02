import shap
import matplotlib.pyplot as plt

def plot_shap_summary(model, X, model_name="Model"):
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    plt.title(f"SHAP Summary for {model_name}")
    shap.plots.beeswarm(shap_values, max_display=20)

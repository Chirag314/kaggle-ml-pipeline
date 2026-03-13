"""Load and return train, test and sample submission dataframes"""
train = pd.read_csv(f"{base_path}train.csv").set_index("id")
test = pd.read_csv(f"{base_path}test.csv").set_index("id")
submission = pd.read_csv(f"{base_path}sample_submission.csv").set_index("id")

# Columns for submission data
TARGET = 'Personality'
ID_COL = 'id'

train["Personality"] = train["Personality"].map({"Extrovert": 1, "Introvert": 0})
train.head()

X = train.drop(columns=[ "Personality"])
y = train[TARGET]

# Identify feature columns
numerical = train.drop(columns=[ "Personality"]).select_dtypes(exclude='object').columns.tolist()
categorical = train.drop(columns=[ "Personality"]).select_dtypes(include='object')\
                      .columns.tolist()

print(f"Train shape: {train.shape}, Test shape: {test.shape}")
print(f"Numeric cols ({len(numerical)}): {numerical}")
print(f"Categorical cols ({len(categorical)}): {categorical}")



# Feature Generator
# Final Prediction and Submission
final_sub = best_weights[0] * sub_preds['xgb'] + best_weights[1] * sub_preds['lgbm'] + best_weights[2] * sub_preds['cat']
submission["Personality"] = (final_sub >= 0.5).astype(int)
submission["Personality"] = submission["Personality"].map({1:"Extrovert", 0:"Introvert"})
submission.to_csv("submission.csv")

# SHAP Importance for final model
import shap
explainer = shap.Explainer(models['xgb'], X_enhanced)
shap_values = explainer(X_enhanced[:500])
shap.summary_plot(shap_values, X_enhanced[:500])


# Sweep configuration (Random Search example)
sweep_config = {
    "method": "random",
    "metric": {"name": "oof_auc", "goal": "maximize"},
    "parameters": {
        "n_estimators": {"values": [100, 200, 300]},
        "max_depth": {"values": [3, 5, 7]},
        "learning_rate": {"values": [0.01, 0.05, 0.1]},
        "subsample": {"values": [0.8, 1.0]}
    }
}
sweep_id = wandb.sweep(sweep_config, project="phd_pipeline")
print("Sweep created with ID:", sweep_id)


# Visualize weight search results
weights_df = pd.DataFrame(weights_auc, columns=["w_xgb", "w_lgbm", "w_cat", "AUC"])
fig, ax = plt.subplots(figsize=(10, 6))
weights_df.plot.scatter(x="w_xgb", y="AUC", alpha=0.7, c="w_cat", cmap="viridis", ax=ax)
plt.title("Blend Weight Grid Search Performance")
plt.xlabel("XGB Weight")
plt.ylabel("Ensemble AUC")
plt.grid(True)
plt.tight_layout()
plt.savefig("blend_auc_plot.png")
wandb.init(project="phd_pipeline", name="ensemble_grid")
wandb.log({"blend_weights_auc_plot": wandb.Image("blend_auc_plot.png")})
wandb.finish()



# Log final submission as artifact
artifact = wandb.Artifact('final_submission', type='submission')
artifact.add_file("submission.csv")
wandb.init(project="phd_pipeline", name="submission_artifact_log")
wandb.log_artifact(artifact)
wandb.finish()



# Parallel Sweep Runner (e.g. from CLI or multiple cells)
# Run this on multiple notebooks or threads to parallelize
def run():
    with wandb.init() as run:
        config = wandb.config
submission1=submission
submission1["Personality"] = (stack_preds >= 0.5).astype(int)
submission1["Personality"] = submission1["Personality"].map({1:"Extrovert", 0:"Introvert"})
submission1.to_csv("submission.csv")
#stack_auc = roc_auc_score(y, stack_preds)
#wandb.log({"stacking_auc": stack_auc})
wandb.finish()


submission.head()

submission1=submission
submission1["Personality"] = (stack_preds >= 0.5).astype(int)
submission1["Personality"] = submission1["Personality"].map({1:"Extrovert", 0:"Introvert"})
#submission1.to_csv("submission1.csv")
#submission1.head()

# Save submission1 to Kaggle working directory
submission1_path = "/kaggle/working/submission1.csv"
submission1.to_csv(submission1_path)

# Confirm it's saved
print("Saved files in working directory:")
print(os.listdir("/kaggle/working"))

submission1.head()


import optuna

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "use_label_encoder": False,
        "eval_metric": "logloss"
    }
submission2=submission
submission2["Personality"] = (best_preds >= 0.5).astype(int)
submission2["Personality"] = submission2["Personality"].map({1:"Extrovert", 0:"Introvert"})
submission2.to_csv("submission2.csv")





# Log artifact to W&B
artifact = wandb.Artifact('xgb_best_model', type='model')
artifact.add_file("best_model.pkl")
wandb.init(project="phd_pipeline", name="artifact_upload")
wandb.log_artifact(artifact)
wandb.finish()
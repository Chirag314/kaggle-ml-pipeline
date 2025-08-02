from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, log_loss
from sklearn.feature_selection import mutual_info_classif
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders import TargetEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
sns.set_palette('pastel')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# Secrets and W&B login
import wandb
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
wandb_key = user_secrets.get_secret("wandb_key")
wandb.login(key=wandb_key)


# Function to load data
base_path="/kaggle/input/playground-series-s5e7/"

fs_model = XGBClassifier(n_estimators=100, random_state=42)
fs_model.fit(X_enhanced, y)
importances = fs_model.feature_importances_
importance_df = pd.DataFrame({'feature': X_enhanced.columns, 'importance': importances})
importance_df = importance_df.sort_values(by='importance', ascending=False)
top_features = importance_df.head(200)['feature'].tolist()
X_enhanced = X_enhanced[top_features]
test_enhanced = test_enhanced[top_features]



wandb.init(project="phd_pipeline", name="feature_importance")
wandb.log({"feature_importances": wandb.Table(dataframe=importance_df)})
wandb.finish()

# Define models
models = {
    'xgb': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'lgbm': LGBMClassifier(),
    'cat': CatBoostClassifier(verbose=0)
}


# Manual 5-Fold CV
from sklearn.metrics import roc_auc_score
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = {}
sub_preds = {}
oof_scores = {}

for model_name, model in models.items():
    wandb.init(project="phd_pipeline", name=f"CV_{model_name}")
    oof = np.zeros(X_enhanced.shape[0])
    test_pred = np.zeros(test_enhanced.shape[0])

    for fold, (trn_idx, val_idx) in enumerate(folds.split(X_enhanced, y)):
        X_trn, y_trn = X_enhanced.iloc[trn_idx], y.iloc[trn_idx]
        X_val, y_val = X_enhanced.iloc[val_idx], y.iloc[val_idx]

        model.fit(X_trn, y_trn)
        oof[val_idx] = model.predict_proba(X_val)[:, 1]
        test_pred += model.predict_proba(test_enhanced)[:, 1] / folds.n_splits

        score = roc_auc_score(y_val, oof[val_idx])
        wandb.log({f"fold_{fold+1}_auc": score})

    final_auc = roc_auc_score(y, oof)
    wandb.log({"oof_auc": final_auc})
    wandb.log({"cv_scores": oof.tolist()})
    wandb.finish()

    oof_preds[model_name] = oof
    sub_preds[model_name] = test_pred
    oof_scores[model_name] = final_auc


# Weighted Model Ensemble (Grid Search)
best_auc = 0
best_weights = (0, 0, 1)
weights_auc = []

for w1 in np.arange(0, 1.05, 0.05):
    for w2 in np.arange(0, 1.05 - w1, 0.05):
        w3 = 1 - w1 - w2
        oof_blend = w1 * oof_preds['xgb'] + w2 * oof_preds['lgbm'] + w3 * oof_preds['cat']
        auc = roc_auc_score(y, oof_blend)
        weights_auc.append((w1, w2, w3, auc))
        if auc > best_auc:
            best_auc = auc
            best_weights = (w1, w2, w3)

print("Best Weights:", best_weights, "| Best Ensemble AUC:", best_auc)


        model = XGBClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            subsample=config.subsample,
            use_label_encoder=False,
            eval_metric="logloss"
        )
        # Train using full training data
        model.fit(X_enhanced, y)
        oof_pred = model.predict_proba(X_enhanced)[:, 1]
        auc = roc_auc_score(y, oof_pred)
        wandb.log({"oof_auc": auc})

# Uncomment below to run sweep agent
wandb.agent(sweep_id, function=run, count=5)



import shap

for model_name, model in models.items():
    explainer = shap.Explainer(model, X_enhanced)
    shap_values = explainer(X_enhanced[:100])  # Only use subset for speed

    plt.figure(figsize=(12, 6))
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.title(f"SHAP Summary for {model_name}")
    plt.savefig(f"shap_{model_name}.png")

    wandb.init(project="phd_pipeline", name=f"shap_{model_name}")
    wandb.log({f"shap_summary_{model_name}": wandb.Image(f"shap_{model_name}.png")})
    wandb.finish()



from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_enhanced), columns=X_enhanced.columns)
test_scaled = pd.DataFrame(scaler.transform(test_enhanced), columns=test_enhanced.columns)


# Define base models
base_learners = [
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
    ('lgbm', LGBMClassifier(max_iter=1000)),
    ('cat', CatBoostClassifier(verbose=0))
]

# Meta model
meta_model = LogisticRegression()

# Full Stacking Ensemble
stack_model = StackingClassifier(estimators=base_learners, final_estimator=meta_model, cv=5, passthrough=True)

wandb.init(project="phd_pipeline", name="stacking_ensemble")
stack_model.fit(X_scaled, y)
stack_preds = stack_model.predict_proba(test_scaled)[:, 1]
    model = XGBClassifier(**params)
    model.fit(X_enhanced, y)
    preds = model.predict_proba(X_enhanced)[:, 1]
    auc = roc_auc_score(y, preds)
    return auc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

print("Best Optuna AUC:", study.best_value)
print("Best Params:", study.best_params)


# Save best model
import joblib

best_model = XGBClassifier(**study.best_params)
best_model.fit(X_enhanced, y)
joblib.dump(best_model, "best_model.pkl")
best_preds = best_model.predict_proba(test_scaled)[:, 1]
from src.data.load_data import load_data
from src.features.feature_engineering import FeatureGenerator
from src.training.train_models import train_base_models
from src.inference.predict_test import run_inference
from src.meta_model.train_meta_model import train_meta_learner
from src.utils.wandb_logger import log_to_wandb
import pandas as pd


def main():
    # Load Data
    train, test, y = load_data()
    X = train.drop(columns=["y"])

    # Feature Engineering
    fg = FeatureGenerator()
    X_trans = fg.fit_transform(X)
    test_trans = fg.transform(test)

    # Train base models with Stratified Nested CV and get OOF predictions
    oof_preds, test_preds, base_models = train_base_models(X_trans, y, test_trans)

    # Train PyTorch Meta Model
    final_preds = train_meta_learner(oof_preds, y, test_preds)

    # Save submission
    submission = pd.read_csv("data/sample_submission.csv")
    submission["y"] = final_preds
    submission.to_csv("submission.csv", index=False)

    # Log to Weights & Biases
    log_to_wandb(oof_preds, y, final_preds)


if __name__ == "__main__":
    main()

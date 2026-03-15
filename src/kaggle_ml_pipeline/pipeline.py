from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from kaggle_ml_pipeline.features.tabular import FeatureGenerator


def _load_data():
    train_path = Path("data/train.csv")
    test_path = Path("data/test.csv")
    sample_submission_path = Path("data/sample_submission.csv")
    target_col = "Personality"

    if train_path.exists() and test_path.exists() and sample_submission_path.exists():
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        submission = pd.read_csv(sample_submission_path)
        if target_col in train.columns:
            y = train[target_col]
            X_train = train.drop(columns=[target_col])
            if "id" in X_train.columns:
                X_train = X_train.drop(columns=["id"])
            X_test = test.drop(columns=["id"]) if "id" in test.columns else test.copy()
        else:
            raise ValueError(f"Target column '{target_col}' not found in data/train.csv")
        return X_train, y, X_test, submission

    # Fallback data keeps the entrypoint/test runnable when dataset files are absent.
    X_train = pd.DataFrame(
        {
            "Age": [25, 30, 45, 22, 34, 41],
            "Openness": [0.8, 0.5, 0.3, 0.9, 0.6, 0.4],
            "Neuroticism": [0.2, 0.7, 0.8, 0.3, 0.5, 0.6],
            "Conscientiousness": [0.7, 0.6, 0.4, 0.8, 0.5, 0.3],
            "Agreeableness": [0.6, 0.5, 0.4, 0.8, 0.7, 0.3],
            "Impulsiveness": [0.2, 0.6, 0.7, 0.3, 0.4, 0.8],
        }
    )
    y = pd.Series([1, 0, 0, 1, 1, 0], name=target_col)
    test = X_train.copy()
    submission = pd.DataFrame({"id": np.arange(len(test)), target_col: ["Introvert"] * len(test)})
    return X_train, y, test, submission


def run_pipeline(output_path="output/submission.csv"):
    X_train, y_raw, X_test, submission = _load_data()
    label_map = {"Extrovert": 1, "Introvert": 0, "1": 1, "0": 0}
    y = y_raw.astype(str).str.strip().map(label_map)
    if y.isna().any():
        y_num = pd.to_numeric(y_raw, errors="coerce")
        y = y.fillna(y_num)
    if y.isna().any():
        raise ValueError("Unable to normalize target labels in data/train.csv")
    y = y.astype(int)

    numerical = X_train.select_dtypes(exclude="object").columns.tolist()
    categorical = X_train.select_dtypes(include="object").columns.tolist()

    fg = FeatureGenerator(numerical=numerical, categorical=categorical)
    X_train_trans = fg.fit_transform(X_train, y)
    X_test_trans = fg.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_trans, y)
    pred = model.predict_proba(X_test_trans)[:, 1]

    submission_col = "Personality" if "Personality" in submission.columns else submission.columns[-1]
    submission[submission_col] = np.where(pred >= 0.5, "Extrovert", "Introvert")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Persist artifacts for app inference.
    joblib.dump(model, output_file.parent / "stack_model.pkl")
    joblib.dump(fg, output_file.parent / "feature_pipeline.pkl")

    submission.to_csv(output_file, index=False)
    return output_file

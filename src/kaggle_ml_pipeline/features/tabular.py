from itertools import combinations

import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer, PolynomialFeatures, StandardScaler


class FeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, numerical, categorical):
        self.numerical = numerical
        self.categorical = categorical

    def fit(self, X, y=None):
        self.target_encoders = {
            col: TargetEncoder().fit(X[[col]], y) for col in self.categorical
        }
        self.scalers = {
            col: StandardScaler().fit(X[[col]].fillna(0)) for col in self.numerical
        }
        return self

    def transform(self, X):
        X_ = X.copy()
        for col in self.numerical:
            X_[col] = self.scalers[col].transform(X_[[col]].fillna(0)).ravel()
            X_[f"{col}_squared"] = X_[col] ** 2
            X_[f"{col}_sqrt"] = np.sqrt(X_[col].clip(lower=0))
            X_[f"{col}_log"] = np.log1p(X_[col].clip(lower=0))
            X_[f"{col}_inv"] = 1 / (X_[col].clip(lower=1e-5))
            X_[f"{col}_exp"] = np.exp(X_[col].clip(upper=20))
            std = X_[col].std()
            X_[f"{col}_z"] = (X_[col] - X_[col].mean()) / (std if std else 1)
            try:
                X_[f"{col}_bin"] = (
                    KBinsDiscretizer(
                        n_bins=5, encode="ordinal", strategy="quantile"
                    )
                    .fit_transform(X_[[col]].fillna(0))
                    .astype(int)
                )
            except ValueError:
                X_[f"{col}_bin"] = 0

        for f1, f2 in combinations(self.numerical[:10], 2):
            X_[f"{f1}_x_{f2}"] = X_[f1] * X_[f2]
            X_[f"{f1}_add_{f2}"] = X_[f1] + X_[f2]
            X_[f"{f1}_div_{f2}"] = X_[f1] / (X_[f2] + 1e-5)

        for f1, f2, f3 in combinations(self.numerical[:5], 3):
            X_[f"{f1}_x_{f2}_x_{f3}"] = X_[f1] * X_[f2] * X_[f3]

        if self.numerical:
            poly = PolynomialFeatures(degree=2, include_bias=False)
            poly_feats = poly.fit_transform(X_[self.numerical[:5]].fillna(0))
            poly_cols = [f"poly_{i}" for i in range(poly_feats.shape[1])]
            X_poly = pd.DataFrame(poly_feats, columns=poly_cols, index=X_.index)
            X_ = pd.concat([X_, X_poly], axis=1)

        for col in self.categorical:
            try:
                X_[f"{col}_te"] = self.target_encoders[col].transform(X[[col]]).values.ravel()
            except Exception:
                X_[f"{col}_te"] = 0

        return X_.select_dtypes(include=[np.number])
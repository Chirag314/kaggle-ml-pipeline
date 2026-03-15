from itertools import combinations
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer, PolynomialFeatures, StandardScaler


class FeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, numerical, categorical):
        self.numerical = numerical
        self.categorical = categorical

    def fit(self, X, y=None):
        self.scalers = {
            col: StandardScaler().fit(X[[col]].fillna(0)) for col in self.numerical
        }

        target = pd.Series(y, index=X.index) if y is not None else pd.Series(dtype=float)
        if not target.empty and target.dtype == object:
            target = target.map({"Extrovert": 1, "Introvert": 0})
        target = pd.to_numeric(target, errors="coerce")
        self.target_global_mean = float(target.mean()) if not target.empty and target.notna().any() else 0.0
        self.target_maps = {}

        if not target.empty:
            for col in self.categorical:
                mapping = (
                    pd.DataFrame({"cat": X[col], "target": target})
                    .groupby("cat", dropna=False)["target"]
                    .mean()
                )
                self.target_maps[col] = mapping

        return self

    def transform(self, X):
        scaled_numeric = {}
        engineered = {}

        for col in self.numerical:
            scaled = self.scalers[col].transform(X[[col]].fillna(0)).ravel()
            scaled_numeric[col] = scaled
            engineered[f"{col}_squared"] = scaled ** 2
            engineered[f"{col}_sqrt"] = np.sqrt(np.clip(scaled, a_min=0, a_max=None))
            engineered[f"{col}_log"] = np.log1p(np.clip(scaled, a_min=0, a_max=None))
            engineered[f"{col}_inv"] = 1 / np.clip(scaled, a_min=1e-5, a_max=None)
            engineered[f"{col}_exp"] = np.exp(np.clip(scaled, a_min=None, a_max=20))
            std = np.std(scaled)
            engineered[f"{col}_z"] = (scaled - np.mean(scaled)) / (std if std else 1)
            try:
                unique_count = max(int(np.unique(scaled).size), 1)
                if unique_count < 2:
                    engineered[f"{col}_bin"] = np.zeros(len(X), dtype=int)
                    continue

                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="Bins whose width are too small",
                        category=UserWarning,
                    )
                    bins = (
                        KBinsDiscretizer(
                            n_bins=min(5, unique_count),
                            encode="ordinal",
                            strategy="quantile",
                            quantile_method="averaged_inverted_cdf",
                        )
                        .fit_transform(np.asarray(scaled).reshape(-1, 1))
                        .astype(int)
                        .ravel()
                    )
                engineered[f"{col}_bin"] = bins
            except ValueError:
                engineered[f"{col}_bin"] = np.zeros(len(X), dtype=int)

        X_numeric = pd.DataFrame(scaled_numeric, index=X.index)

        for f1, f2 in combinations(self.numerical[:10], 2):
            engineered[f"{f1}_x_{f2}"] = X_numeric[f1] * X_numeric[f2]
            engineered[f"{f1}_add_{f2}"] = X_numeric[f1] + X_numeric[f2]
            engineered[f"{f1}_div_{f2}"] = X_numeric[f1] / (X_numeric[f2] + 1e-5)

        for f1, f2, f3 in combinations(self.numerical[:5], 3):
            engineered[f"{f1}_x_{f2}_x_{f3}"] = X_numeric[f1] * X_numeric[f2] * X_numeric[f3]

        X_engineered = pd.DataFrame(engineered, index=X.index)

        features_to_concat = [X_numeric, X_engineered]
        if self.numerical:
            poly = PolynomialFeatures(degree=2, include_bias=False)
            poly_feats = poly.fit_transform(X_numeric[self.numerical[:5]].fillna(0))
            poly_cols = [f"poly_{i}" for i in range(poly_feats.shape[1])]
            X_poly = pd.DataFrame(poly_feats, columns=poly_cols, index=X.index)
            features_to_concat.append(X_poly)

        target_encoded = {}
        for col in self.categorical:
            mapping = self.target_maps.get(col)
            if mapping is None:
                target_encoded[f"{col}_te"] = np.full(len(X), self.target_global_mean)
            else:
                target_encoded[f"{col}_te"] = X[col].map(mapping).fillna(self.target_global_mean)

        if target_encoded:
            X_target_encoded = pd.DataFrame(target_encoded, index=X.index)
            features_to_concat.append(X_target_encoded)

        X_out = pd.concat(features_to_concat, axis=1)
        return X_out.select_dtypes(include=[np.number])
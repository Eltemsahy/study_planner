"""Neural network mark predictor + baselines (all share fit/predict on a DataFrame)."""
from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .. import config as C


def _prep(df: pd.DataFrame) -> np.ndarray:
    X = df[C.FEATURES].astype(float).copy()
    for c in ["clicks_content", "clicks_interaction", "clicks_practice", "clicks_lag_total"]:
        X[c] = np.log1p(X[c])            # clicks are heavy tailed; log keeps NN stable
    return X.values


class MarkNN:
    name = "NN (MLP 16-8)"
    uses_clicks = True

    def __init__(self, seed=C.SEED, hidden=(16, 8)):
        self.seed, self.hidden = seed, hidden
        self.net = make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=hidden, alpha=1e-3, max_iter=800, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=20, random_state=seed))

    def fit(self, df):
        self.net.fit(_prep(df), df[C.TARGET].values)
        return self

    def predict(self, df):
        return np.clip(self.net.predict(_prep(df)), 0, 100)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)


class GlobalMean:
    name = "Baseline: global mean"
    uses_clicks = False

    def fit(self, df):
        self.m = float(df[C.TARGET].mean()); return self

    def predict(self, df):
        return np.full(len(df), self.m)


class PrevAverage:
    """Simple average-mark predictor: mean of the student's earlier marks (global mean if none).
    Ignores clicks, so it cannot respond to extra study effort."""
    name = "Baseline: average of previous marks"
    uses_clicks = False

    def fit(self, df):
        self.m = float(df[C.TARGET].mean()); return self

    def predict(self, df):
        return np.where(df.n_prev.values > 0, df.prev_mean.values, self.m)


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = p - y
    ss = ((y - y.mean()) ** 2).sum()
    return {"MAE": float(np.abs(err).mean()), "RMSE": float(np.sqrt((err ** 2).mean())),
            "R2": float(1 - (err ** 2).sum() / ss) if ss > 0 else float("nan")}


def evaluate_all(train, test, seed=C.SEED):
    models = [MarkNN(seed), PrevAverage(), GlobalMean()]
    out = {}
    for m in models:
        m.fit(train)
        out[m.name] = metrics(test[C.TARGET], m.predict(test))
    return out, models[0]

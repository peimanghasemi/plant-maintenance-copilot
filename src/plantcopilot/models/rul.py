"""Remaining-useful-life model.

Gradient boosting on rolling features, trained on a capped RUL target (far from
failure the exact horizon is neither predictable nor useful). Evaluated on units
cut at a random point, the usual setup for run-to-failure benchmarks.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import add_features, feature_columns

RUL_CAP = 400


@dataclass
class RULModel:
    model: HistGradientBoostingRegressor
    features: list[str]

    @classmethod
    def fit(cls, train: pd.DataFrame, seed: int = 0) -> "RULModel":
        f = add_features(train)
        cols = feature_columns(f)
        y = f["rul"].clip(upper=RUL_CAP)
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=seed)
        m.fit(f[cols], y)
        return cls(m, cols)

    def predict_last(self, history: pd.DataFrame) -> pd.DataFrame:
        """Predicted RUL at the last observed hour of every unit."""
        f = add_features(history)
        last = f.groupby("unit_id", sort=False).tail(1)
        pred = np.clip(self.model.predict(last[self.features]), 0, RUL_CAP)
        return pd.DataFrame({"unit_id": last["unit_id"].values, "hour": last["hour"].values,
                             "rul_pred": pred.round(1)})

    def evaluate(self, test: pd.DataFrame) -> dict:
        pred = self.predict_last(test)
        truth = test.groupby("unit_id", sort=False).tail(1)[["unit_id", "rul"]]
        m = pred.merge(truth, on="unit_id")
        y = m["rul"].clip(upper=RUL_CAP)
        err = m["rul_pred"] - y
        near = y <= 250                       # the range where the maintenance policy acts
        return {
            "units": int(len(m)),
            "rmse": round(float(np.sqrt((err ** 2).mean())), 2),
            "mae": round(float(err.abs().mean()), 2),
            "rmse_rul_below_250": round(float(np.sqrt((err[near] ** 2).mean())), 2) if near.any() else None,
            "late_share": round(float((err > 0).mean()), 3),   # share of predictions that are too optimistic
        }

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "RULModel":
        return joblib.load(path)


def log_run(params: dict, metrics: dict, out_dir: str | Path = "runs") -> None:
    """Log to MLflow if it is installed, otherwise to a JSON file."""
    try:
        import mlflow
        with mlflow.start_run(run_name="rul"):
            mlflow.log_params(params)
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
    except ImportError:
        out = Path(out_dir); out.mkdir(exist_ok=True)
        (out / "rul_run.json").write_text(json.dumps({"params": params, "metrics": metrics}, indent=2))

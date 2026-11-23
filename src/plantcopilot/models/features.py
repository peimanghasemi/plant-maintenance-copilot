"""Rolling-window features per unit."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..simulate import SENSORS

WINDOWS = (12, 48)


def _slope(x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    t = np.arange(len(x))
    t = t - t.mean()
    return float((t * (x - x.mean())).sum() / (t ** 2).sum())


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, g in df.groupby("unit_id", sort=False):
        g = g.sort_values("hour").copy()
        for s in SENSORS:
            for w in WINDOWS:
                roll = g[s].rolling(w, min_periods=1)
                g[f"{s}_mean_{w}"] = roll.mean()
                g[f"{s}_std_{w}"] = roll.std().fillna(0.0)
            g[f"{s}_slope_48"] = g[s].rolling(48, min_periods=2).apply(_slope, raw=True).fillna(0.0)
            # drift against the unit's own first day, which removes unit-to-unit offsets
            g[f"{s}_drift"] = g[f"{s}_mean_48"] - g[s].iloc[:24].mean()
        out.append(g)
    return pd.concat(out, ignore_index=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    keep = ("_mean_", "_std_", "_slope_", "_drift")
    return [c for c in df.columns if any(k in c for k in keep)]

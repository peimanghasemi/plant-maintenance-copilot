"""Anomaly detection against each unit's own healthy baseline.

For every unit the first `baseline_hours` define normal behaviour. The score is
the largest robust z-score of the smoothed sensors against that baseline. An
alarm needs `persist` consecutive hours above the threshold, which filters out
single spikes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..simulate import SENSORS


def anomaly_scores(df: pd.DataFrame, baseline_hours: int = 72, smooth: int = 24) -> pd.DataFrame:
    out = []
    for _, g in df.groupby("unit_id", sort=False):
        g = g.sort_values("hour").copy()
        z = []
        for s in SENSORS:
            x = g[s].rolling(smooth, min_periods=1).median()
            base = x.iloc[:baseline_hours]
            med = base.median()
            # floor the spread at 2% of the level so a very quiet baseline does not alarm on tiny drifts
            mad = max((base - med).abs().median() * 1.4826, 0.02 * abs(med), 1e-9)
            z.append(((x - med) / mad).clip(lower=0))
        g["score"] = np.max(np.column_stack(z), axis=1)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def first_alarm(scored: pd.DataFrame, threshold: float, persist: int = 6) -> pd.Series:
    """Hour of the first alarm per unit (NaN if none)."""
    res = {}
    for uid, g in scored.groupby("unit_id", sort=False):
        above = (g["score"].values > threshold).astype(int)
        run = np.convolve(above, np.ones(persist, dtype=int), mode="full")[: len(above)]
        hit = np.flatnonzero(run >= persist)
        res[uid] = float(g["hour"].values[hit[0]]) if hit.size else np.nan
    return pd.Series(res)


def evaluate_alarms(failing: pd.DataFrame, healthy: pd.DataFrame, threshold: float, persist: int = 6) -> dict:
    """Detection and lead time on units that fail, false alarms on units that stay healthy."""
    alarm = first_alarm(anomaly_scores(failing), threshold, persist)
    fail = failing.groupby("unit_id", sort=False)["hour"].max()
    lead = (fail - alarm).dropna()
    false = first_alarm(anomaly_scores(healthy), threshold, persist).notna()
    return {
        "threshold": threshold,
        "detection_rate": round(float(alarm.notna().mean()), 3),
        "median_lead_hours": float(lead.median()) if len(lead) else None,
        "p10_lead_hours": float(lead.quantile(0.1)) if len(lead) else None,
        "false_alarm_rate": round(float(false.mean()), 3),
    }

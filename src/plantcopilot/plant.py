"""Plant state shared by the tools and the API: sensor history, models, documents, CMMS."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .cmms import CMMS
from .models.anomaly import anomaly_scores
from .models.rul import RULModel
from .rag.retriever import Retriever
from .simulate import simulate_fleet, truncate_for_test

ALARM_THRESHOLD = 6.0          # chosen on the alarm trade-off curve (see README)


def priority_from_rul(rul: float) -> str | None:
    if rul < 48:
        return "urgent"
    if rul < 120:
        return "high"
    if rul < 250:
        return "medium"
    return None


@dataclass
class Plant:
    history: pd.DataFrame
    model: RULModel
    retriever: Retriever
    cmms: CMMS

    @classmethod
    def demo(cls, docs: str | Path = "docs/manuals", model_path: str | Path | None = None) -> "Plant":
        history = truncate_for_test(simulate_fleet(24, seed=11), seed=12)
        if model_path and Path(model_path).exists():
            model = RULModel.load(model_path)
        else:
            model = RULModel.fit(simulate_fleet(60, seed=0))
        return cls(history, model, Retriever.from_folder(docs), CMMS())

    def units(self) -> list[str]:
        return list(self.history["unit_id"].unique())

    def unit_history(self, unit_id: str) -> pd.DataFrame:
        h = self.history[self.history["unit_id"] == unit_id]
        if h.empty:
            raise KeyError(f"unknown unit {unit_id}")
        return h

    def status(self, unit_id: str) -> dict:
        h = self.unit_history(unit_id)
        scored = anomaly_scores(h)
        last = scored.iloc[-1]
        recent = scored["score"].tail(6)
        return {
            "unit_id": unit_id,
            "equipment": last["equipment"],
            "hour": int(last["hour"]),
            "vibration": float(last["vibration"]),
            "bearing_temp": float(last["bearing_temp"]),
            "motor_current": float(last["motor_current"]),
            "pressure_drop": float(last["pressure_drop"]),
            "anomaly_score": round(float(last["score"]), 2),
            "alarm": bool((recent > ALARM_THRESHOLD).all()),
        }

    def rul(self, unit_id: str) -> dict:
        pred = self.model.predict_last(self.unit_history(unit_id)).iloc[0]
        rul = float(pred["rul_pred"])
        return {"unit_id": unit_id, "rul_hours": rul, "priority": priority_from_rul(rul)}

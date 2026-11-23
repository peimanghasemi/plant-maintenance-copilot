"""Run-to-failure sensor data for rotating equipment in a cement plant.

Each unit starts healthy and degrades at its own random rate, with a slow phase
followed by accelerating wear. Sensors respond to the hidden health state plus
operating noise and occasional spikes. A unit fails when health reaches zero.

This is a simulator, not plant data. It is meant to make the pipeline runnable
end to end and to give a known ground truth (remaining useful life, RUL).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EQUIPMENT = {
    # name: (baseline vibration mm/s, bearing temp C, motor current A, typical life in hours)
    "cement_mill": (2.8, 62.0, 410.0, 900),
    "raw_mill": (3.1, 58.0, 330.0, 800),
    "kiln_id_fan": (2.2, 55.0, 250.0, 1100),
    "clinker_cooler_fan": (2.5, 60.0, 180.0, 1000),
}
SENSORS = ["vibration", "bearing_temp", "motor_current", "pressure_drop"]


@dataclass
class Unit:
    unit_id: str
    equipment: str
    life: int


def simulate_unit(unit: Unit, rng: np.random.Generator, degrade: bool = True) -> pd.DataFrame:
    vib0, temp0, cur0, _ = EQUIPMENT[unit.equipment]
    T = unit.life
    t = np.arange(T)
    # health goes from 1 to 0; wear accelerates after a random onset
    onset = rng.uniform(0.35, 0.65)
    s = t / (T - 1)
    late = np.clip((s - onset) / (1 - onset), 0, 1)
    wear = np.where(s < onset, 0.15 * s / onset, 0.15 + 0.85 * late ** rng.uniform(1.6, 2.4))
    health = 1.0 - wear if degrade else np.ones(T)
    d = 1.0 - health

    # units differ: sensor offsets, and how strongly each sensor reacts to wear
    off = rng.normal(1.0, 0.08, 4)
    react = rng.uniform(0.5, 1.5, 4)

    load = 1.0 + 0.05 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.02, T)   # daily load cycle
    vib = off[0] * vib0 * load * (1 + 2.2 * react[0] * d ** 1.5) + rng.normal(0, 0.25, T)
    temp = off[1] * temp0 + 8 * (load - 1) + 18 * react[1] * d ** 2 + rng.normal(0, 1.2, T)
    cur = off[2] * cur0 * load * (1 + 0.12 * react[2] * d) + rng.normal(0, 6.0, T)
    dp = off[3] * 12.0 * (1 + 0.6 * react[3] * d) + rng.normal(0, 0.5, T)

    # rare short spikes (sensor glitches, process upsets) that are not failures
    for arr, scale in ((vib, 2.0), (temp, 6.0)):
        k = rng.random(T) < 0.004
        arr[k] += rng.uniform(0.5, 1.0, k.sum()) * scale

    return pd.DataFrame({
        "unit_id": unit.unit_id, "equipment": unit.equipment, "hour": t,
        "vibration": vib.round(3), "bearing_temp": temp.round(2),
        "motor_current": cur.round(1), "pressure_drop": dp.round(3),
        "rul": T - 1 - t if degrade else np.full(T, -1),
    })


def simulate_fleet(n_units: int = 60, seed: int = 0, degrade: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = list(EQUIPMENT)
    frames = []
    for i in range(n_units):
        eq = names[i % len(names)]
        life = int(EQUIPMENT[eq][3] * rng.uniform(0.6, 1.4))
        frames.append(simulate_unit(Unit(f"{eq.upper()[:3]}-{i:03d}", eq, life), rng, degrade))
    return pd.concat(frames, ignore_index=True)


def truncate_for_test(df: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    """Cut every unit at a random point, as if observed before failure."""
    rng = np.random.default_rng(seed)
    parts = []
    for _, g in df.groupby("unit_id", sort=False):
        cut = int(len(g) * rng.uniform(0.3, 0.95))
        parts.append(g.iloc[:cut])
    return pd.concat(parts, ignore_index=True)

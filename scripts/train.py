"""Train the RUL model on simulated run-to-failure data, evaluate it and pick the alarm threshold.

Writes models/rul.joblib, results/rul.json and results/alarms.md.
"""
import json
from pathlib import Path

from plantcopilot.models.anomaly import evaluate_alarms
from plantcopilot.models.rul import RULModel, log_run
from plantcopilot.simulate import simulate_fleet, truncate_for_test

Path("models").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

train = simulate_fleet(60, seed=0)
test_full = simulate_fleet(30, seed=1)
test = truncate_for_test(test_full, seed=1)

model = RULModel.fit(train)
metrics = model.evaluate(test)
model.save("models/rul.joblib")
log_run({"model": "HistGradientBoosting", "train_units": 60, "rul_cap": 400}, metrics)
Path("results/rul.json").write_text(json.dumps(metrics, indent=2))
print("RUL", metrics)

healthy = simulate_fleet(40, seed=2, degrade=False)
failing = simulate_fleet(40, seed=3)
lines = ["| threshold | detection rate | median lead (h) | 10% lead (h) | false alarm rate |", "|---|---|---|---|---|"]
for th in (3, 4, 5, 6, 8, 10):
    m = evaluate_alarms(failing, healthy, th)
    lines.append(f"| {th} | {m['detection_rate']:.2f} | {m['median_lead_hours']:.0f} | "
                 f"{m['p10_lead_hours']:.0f} | {m['false_alarm_rate']:.3f} |")
    print(lines[-1])
Path("results/alarms.md").write_text("\n".join(lines) + "\n")

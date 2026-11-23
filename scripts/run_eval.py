"""Evaluate the assistant on eval/cases.yaml.

    python scripts/run_eval.py                    # offline rule-based policy (CI)
    python scripts/run_eval.py --llm openai       # any OpenAI-compatible endpoint, needs LLM_API_KEY
    python scripts/run_eval.py --update-baseline  # accept the current results as the new baseline

Exits with status 1 if a case that passed in eval/baseline.json now fails,
or if a work order was written without approval.
"""
import argparse
import json
import sys
from pathlib import Path

from plantcopilot.agent.agent import Agent, Monitor
from plantcopilot.agent.llm import OpenAIChatLLM, RuleBasedLLM
from plantcopilot.agent.tools import Tools
from plantcopilot.evaluation import compare_to_baseline, evaluate
from plantcopilot.plant import Plant

p = argparse.ArgumentParser()
p.add_argument("--llm", choices=["rules", "openai"], default="rules")
p.add_argument("--cases", default="eval/cases.yaml")
p.add_argument("--baseline", default="eval/baseline.json")
p.add_argument("--update-baseline", action="store_true")
a = p.parse_args()

plant = Plant.demo(model_path="models/rul.joblib")
llm = OpenAIChatLLM() if a.llm == "openai" else RuleBasedLLM()
agent = Agent(llm, Tools(plant), monitor=Monitor(f"runs/eval_{a.llm}.jsonl"))
report = evaluate(agent, plant, a.cases)

Path("results").mkdir(exist_ok=True)
Path(f"results/eval_{a.llm}.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report["summary"], indent=2))
for c in report["cases"]:
    if not c["passed"]:
        print("FAILED", c["id"], {k: v for k, v in c["checks"].items() if not v})

if a.update_baseline:
    Path(a.baseline).write_text(json.dumps(report, indent=2))
    print("baseline updated")
    sys.exit(0)

problems = compare_to_baseline(report, a.baseline)
for msg in problems:
    print("REGRESSION:", msg)
sys.exit(1 if problems else 0)

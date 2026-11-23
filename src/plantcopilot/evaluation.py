"""Evaluation of the assistant on fixed cases.

Per case: did it call the right tools, cite the right document, say what it must
say, and propose a work order exactly when the policy calls for one with the
priority the policy gives. Across all cases: task success, a safety check that
no work order was ever written without approval, tokens, cost and latency.
A baseline file turns this into a regression test for prompt or model changes.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from .agent.agent import Agent, Monitor
from .plant import Plant, priority_from_rul


@dataclass
class CaseResult:
    id: str
    passed: bool
    checks: dict = field(default_factory=dict)
    tools: list = field(default_factory=list)
    answer: str = ""
    tokens: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0


def expected_priority(status: dict, rul: dict) -> str | None:
    """Priority the maintenance policy gives (RUL rule first, alarm means inspection)."""
    p = priority_from_rul(rul["rul_hours"])
    if p is None and status.get("alarm"):
        p = "medium"
    return p


def check_case(case: dict, run, plant: Plant) -> CaseResult:
    answer = run.answer.lower()
    checks = {}
    used = set(run.tools_used)
    checks["tools"] = set(case.get("expect_tools", [])) <= used
    if "expect_cite" in case:
        checks["citation"] = f"[{case['expect_cite']}".lower() in answer
    for t in case.get("expect_text", []):
        checks[f"text:{t}"] = t.lower() in answer
    proposals = [c for c in run.tool_calls if c["name"] == "propose_work_order"]
    checks["proposal"] = bool(proposals) == case.get("expect_proposal", False)
    if proposals and case.get("expect_proposal"):
        unit = proposals[0]["arguments"]["unit_id"]
        want = expected_priority(plant.status(unit), plant.rul(unit))
        checks["priority"] = proposals[0]["arguments"]["priority"] == want
    checks["no_error"] = run.error is None
    return CaseResult(case["id"], all(checks.values()), checks, run.tools_used, run.answer,
                      run.prompt_tokens + run.completion_tokens, round(run.seconds, 3))


def evaluate(agent: Agent, plant: Plant, cases_path: str | Path = "eval/cases.yaml") -> dict:
    cases = yaml.safe_load(Path(cases_path).read_text())
    monitor = agent.monitor or Monitor()
    results = []
    for c in cases:
        run = agent.run(c["question"])
        r = check_case(c, run, plant)
        r.cost_usd = round(monitor.cost(run), 6)
        results.append(r)
    n = len(results)
    tool_ok = [r.checks["tools"] for r in results]
    cite = [r.checks["citation"] for r in results if "citation" in r.checks]
    prop = [r.checks["proposal"] for r in results]
    prio = [r.checks["priority"] for r in results if "priority" in r.checks]
    summary = {
        "cases": n,
        "task_success": round(sum(r.passed for r in results) / n, 3),
        "tool_selection": round(sum(tool_ok) / n, 3),
        "citation_accuracy": round(sum(cite) / len(cite), 3) if cite else None,
        "proposal_decision_accuracy": round(sum(prop) / n, 3),
        "priority_accuracy": round(sum(prio) / len(prio), 3) if prio else None,
        # safety: the agent may only propose; anything in work_orders came from an approval
        "work_orders_written_without_approval": len(plant.cmms.work_orders()),
        "mean_tokens": round(sum(r.tokens for r in results) / n, 1),
        "total_cost_usd": round(sum(r.cost_usd for r in results), 5),
        "p95_seconds": sorted(r.seconds for r in results)[int(0.95 * (n - 1))],
    }
    return {"summary": summary, "cases": [asdict(r) for r in results]}


def compare_to_baseline(report: dict, baseline_path: str | Path) -> list[str]:
    """Regressions: cases that passed in the baseline and fail now, or any safety violation."""
    problems = []
    if report["summary"]["work_orders_written_without_approval"]:
        problems.append("work orders were written without approval")
    path = Path(baseline_path)
    if path.exists():
        base = {c["id"]: c["passed"] for c in json.loads(path.read_text())["cases"]}
        for c in report["cases"]:
            if base.get(c["id"]) and not c["passed"]:
                problems.append(f"regression in case {c['id']}: {c['checks']}")
    return problems

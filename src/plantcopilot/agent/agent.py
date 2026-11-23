"""Tool-calling loop with a step limit, a trace of every call and per-run monitoring."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .tools import SCHEMAS, Tools

SYSTEM = """You are a maintenance assistant in a cement plant.
Use the tools to look up unit status, remaining useful life and documentation before answering.
Cite documentation by its id in square brackets, for example [cement_mill#recommended-actions].
You may propose a work order only when the maintenance policy calls for one, and you must say
that it needs approval by the maintenance engineer. You cannot create or approve work orders.
If a unit id is unknown, say so instead of guessing."""


@dataclass
class RunResult:
    question: str
    answer: str
    tool_calls: list = field(default_factory=list)       # [{"name", "arguments", "result"}]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    error: str | None = None

    @property
    def tools_used(self) -> list[str]:
        return [c["name"] for c in self.tool_calls]


class Agent:
    def __init__(self, llm, tools: Tools, max_steps: int = 8, monitor: "Monitor | None" = None):
        self.llm, self.tools, self.max_steps, self.monitor = llm, tools, max_steps, monitor

    def run(self, question: str) -> RunResult:
        t0 = time.perf_counter()
        res = RunResult(question, "")
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
        try:
            for _ in range(self.max_steps):
                reply = self.llm.chat(messages, SCHEMAS)
                res.prompt_tokens += reply.usage.get("prompt_tokens", 0)
                res.completion_tokens += reply.usage.get("completion_tokens", 0)
                if not reply.tool_calls:
                    res.answer = reply.content or ""
                    break
                messages.append({"role": "assistant", "content": reply.content, "tool_calls": [
                    {"id": c["id"], "type": "function",
                     "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                    for c in reply.tool_calls]})
                for c in reply.tool_calls:
                    out = self.tools.call(c["name"], c["arguments"])
                    res.tool_calls.append({"name": c["name"], "arguments": c["arguments"], "result": json.loads(out)})
                    messages.append({"role": "tool", "tool_call_id": c["id"], "name": c["name"], "content": out})
            else:
                res.error = "step limit reached"
        except Exception as e:                            # keep the service up, record the failure
            res.error = f"{type(e).__name__}: {e}"
        res.seconds = time.perf_counter() - t0
        if self.monitor:
            self.monitor.log(res)
        return res


class Monitor:
    """Appends one JSON line per run: tools, tokens, estimated cost, latency, errors."""

    def __init__(self, path: str | Path = "runs/agent_runs.jsonl", usd_per_1k_prompt: float = 0.00015,
                 usd_per_1k_completion: float = 0.0006):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.p_in, self.p_out = usd_per_1k_prompt, usd_per_1k_completion

    def cost(self, r: RunResult) -> float:
        return r.prompt_tokens / 1000 * self.p_in + r.completion_tokens / 1000 * self.p_out

    def log(self, r: RunResult) -> None:
        rec = {"ts": time.time(), "question": r.question, "tools": r.tools_used,
               "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
               "cost_usd": round(self.cost(r), 6), "seconds": round(r.seconds, 3), "error": r.error,
               "proposed": [c["result"].get("proposal_id") for c in r.tool_calls if c["name"] == "propose_work_order"]}
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    def summary(self) -> dict:
        if not self.path.exists():
            return {}
        rows = [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]
        if not rows:
            return {}
        n = len(rows)
        return {"runs": n,
                "error_rate": round(sum(r["error"] is not None for r in rows) / n, 3),
                "mean_tool_calls": round(sum(len(r["tools"]) for r in rows) / n, 2),
                "mean_tokens": round(sum(r["prompt_tokens"] + r["completion_tokens"] for r in rows) / n, 1),
                "total_cost_usd": round(sum(r["cost_usd"] for r in rows), 4),
                "p95_seconds": sorted(r["seconds"] for r in rows)[int(0.95 * (n - 1))],
                "proposals": sum(len(r["proposed"]) for r in rows)}

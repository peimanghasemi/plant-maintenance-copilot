"""LLM backends with one interface: chat(messages, tools) -> LLMReply.

OpenAIChatLLM talks to any OpenAI-compatible endpoint (OpenAI, Azure OpenAI,
vLLM, ...). RuleBasedLLM is a deterministic stand-in with no language model
behind it. It follows a fixed tool plan, so the agent loop, the tools, the
approval gate and the evaluation run in CI without an API key or network.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

UNIT_RE = re.compile(r"\b[A-Z]{3}-\d{3}\b")


@dataclass
class LLMReply:
    content: str | None
    tool_calls: list = field(default_factory=list)      # [{"id", "name", "arguments": dict}]
    usage: dict = field(default_factory=dict)            # prompt_tokens, completion_tokens


class OpenAIChatLLM:
    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None,
                 temperature: float = 0.0):
        from openai import OpenAI                         # optional dependency
        self.client = OpenAI(base_url=base_url or os.environ.get("LLM_BASE_URL"),
                             api_key=api_key or os.environ["LLM_API_KEY"])
        self.model = model or os.environ.get("LLM_MODEL", "gpt-4o-mini")
        self.temperature = temperature

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMReply:
        r = self.client.chat.completions.create(model=self.model, messages=messages, tools=tools,
                                                temperature=self.temperature)
        m = r.choices[0].message
        calls = [{"id": c.id, "name": c.function.name, "arguments": json.loads(c.function.arguments or "{}")}
                 for c in (m.tool_calls or [])]
        return LLMReply(m.content, calls, {"prompt_tokens": r.usage.prompt_tokens,
                                           "completion_tokens": r.usage.completion_tokens})


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class RuleBasedLLM:
    ORDER = ("work order", "schedule", "plan maintenance", "plan the maintenance", "create")
    DOCS = ("how do", "how to", "how should", "what should", "procedure", "check", "policy", "why", "recommend", "spare", "what to do")
    STATUS = ("status", "health", "how is", "vibration", "temperature", "alarm", "condition")
    RUL = ("rul", "remaining", "how long", "fail", "life")

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMReply:
        question = next(m["content"] for m in messages if m["role"] == "user")
        q = question.lower()
        done = {}
        for m in messages:
            if m["role"] == "tool":
                done[m["name"]] = json.loads(m["content"])
        units = UNIT_RE.findall(question)
        unit = units[0] if units else None
        wants_order = any(w in q for w in self.ORDER)
        wants_docs = any(w in q for w in self.DOCS) or wants_order
        wants_status = any(w in q for w in self.STATUS) or wants_order
        wants_rul = any(w in q for w in self.RUL) or wants_order

        plan = []
        if unit and wants_status:
            plan.append(("get_unit_status", {"unit_id": unit}))
        if unit and wants_rul:
            plan.append(("predict_rul", {"unit_id": unit}))
        if wants_docs or not unit:
            equipment = done.get("get_unit_status", {}).get("equipment", "")
            plan.append(("search_docs", {"query": f"{equipment.replace('_', ' ')} {question}".strip()}))
        unit_error = any("error" in done.get(n, {}) for n in ("get_unit_status", "predict_rul"))
        for name, args in plan:
            if unit_error:
                break
            if name not in done:
                if name == "search_docs" and unit and wants_status and "get_unit_status" not in done:
                    continue
                return self._call(name, args, messages)

        if wants_order and unit and "propose_work_order" not in done:
            status, rul = done.get("get_unit_status", {}), done.get("predict_rul", {})
            if "error" not in status and "error" not in rul:
                if rul.get("priority"):
                    return self._call("propose_work_order", {
                        "unit_id": unit, "action": f"Inspect and repair {status['equipment'].replace('_', ' ')}",
                        "priority": rul["priority"],
                        "reason": f"Predicted RUL {rul['rul_hours']:.0f} h"}, messages)
                if status.get("alarm"):
                    return self._call("propose_work_order", {
                        "unit_id": unit, "action": f"Inspect {status['equipment'].replace('_', ' ')}",
                        "priority": "medium", "reason": f"Anomaly alarm, score {status['anomaly_score']}"}, messages)
        return self._answer(question, unit, done, messages)

    def _call(self, name, args, messages):
        prompt = sum(_approx_tokens(str(m.get("content") or "")) for m in messages)
        return LLMReply(None, [{"id": f"call_{name}_{len(messages)}", "name": name, "arguments": args}],
                        {"prompt_tokens": prompt, "completion_tokens": 20})

    def _answer(self, question, unit, done, messages):
        parts = []
        for name in ("get_unit_status", "predict_rul", "propose_work_order"):
            if "error" in done.get(name, {}):
                parts.append(f"I could not find unit {unit}. Please check the unit id.")
                break
        else:
            if "get_unit_status" in done:
                s = done["get_unit_status"]
                parts.append(f"{s['unit_id']} ({s['equipment'].replace('_', ' ')}): vibration {s['vibration']} mm/s, "
                             f"bearing temperature {s['bearing_temp']} C, anomaly score {s['anomaly_score']}"
                             + (", alarm active." if s["alarm"] else ", no alarm."))
            if "predict_rul" in done:
                r = done["predict_rul"]
                parts.append(f"Predicted remaining useful life: {r['rul_hours']:.0f} hours"
                             + (f", which means priority {r['priority']} under the maintenance policy." if r["priority"]
                                else ", no work order needed yet."))
            if "propose_work_order" in done:
                p = done["propose_work_order"]
                parts.append(f"I proposed work order {p['proposal_id']}. It needs approval by the maintenance "
                             "engineer before it is created in the CMMS.")
            elif unit and any(w in question.lower() for w in self.ORDER):
                parts.append("No work order proposed, the unit does not need one under the maintenance policy.")
        docs = [] if unit and any("error" in done.get(n, {}) for n in ("get_unit_status", "predict_rul")) \
            else (done.get("search_docs") or [])
        if isinstance(docs, list) and docs:
            parts.append(f"From the documentation: {docs[0]['text'][:300]} [{docs[0]['id']}]")
        text = " ".join(parts) or "I could not find relevant information."
        prompt = sum(_approx_tokens(str(m.get("content") or "")) for m in messages)
        return LLMReply(text, [], {"prompt_tokens": prompt, "completion_tokens": _approx_tokens(text)})

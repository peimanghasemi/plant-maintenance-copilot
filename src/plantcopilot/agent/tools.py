"""Tools the assistant can call. Read-only, except propose_work_order, which only
creates a pending proposal. There is deliberately no tool that approves or writes."""
from __future__ import annotations

import json

from ..plant import Plant

SCHEMAS = [
    {"type": "function", "function": {
        "name": "get_unit_status",
        "description": "Latest sensor readings, anomaly score and alarm flag of one unit.",
        "parameters": {"type": "object", "properties": {"unit_id": {"type": "string"}}, "required": ["unit_id"]}}},
    {"type": "function", "function": {
        "name": "predict_rul",
        "description": "Predicted remaining useful life in hours and the work order priority from the maintenance policy.",
        "parameters": {"type": "object", "properties": {"unit_id": {"type": "string"}}, "required": ["unit_id"]}}},
    {"type": "function", "function": {
        "name": "search_docs",
        "description": "Search equipment manuals and the maintenance policy. Returns sections with ids to cite.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "propose_work_order",
        "description": "Propose a work order. It is NOT created until a maintenance engineer approves it.",
        "parameters": {"type": "object", "properties": {
            "unit_id": {"type": "string"}, "action": {"type": "string"},
            "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]},
            "reason": {"type": "string"}},
            "required": ["unit_id", "action", "priority", "reason"]}}},
]


class Tools:
    def __init__(self, plant: Plant):
        self.plant = plant

    def call(self, name: str, args: dict) -> str:
        try:
            if name == "get_unit_status":
                out = self.plant.status(args["unit_id"])
            elif name == "predict_rul":
                out = self.plant.rul(args["unit_id"])
            elif name == "search_docs":
                out = [{"id": c.id, "title": c.title, "text": c.text} for c in self.plant.retriever.search(args["query"])]
            elif name == "propose_work_order":
                p = self.plant.cmms.propose(args["unit_id"], args["action"], args["priority"], args["reason"])
                out = {"proposal_id": p.id, "status": p.status,
                       "note": "Pending approval by the maintenance engineer. Not created in the CMMS yet."}
            else:
                out = {"error": f"unknown tool {name}"}
        except (KeyError, ValueError) as e:
            out = {"error": str(e)}
        return json.dumps(out)

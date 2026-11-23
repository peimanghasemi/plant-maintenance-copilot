"""HTTP API: model endpoints, the assistant, and the approval step for work orders.

Run with:  uvicorn plantcopilot.api:app --port 8000
Set LLM_API_KEY (and optionally LLM_BASE_URL, LLM_MODEL) to use a real LLM,
otherwise the offline rule-based policy is used.
"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .agent.agent import Agent, Monitor
from .agent.llm import OpenAIChatLLM, RuleBasedLLM
from .agent.tools import Tools
from .plant import Plant

app = FastAPI(title="Plant maintenance copilot")
plant = Plant.demo(model_path=os.environ.get("RUL_MODEL_PATH", "models/rul.joblib"))
monitor = Monitor(os.environ.get("RUN_LOG", "runs/agent_runs.jsonl"))
llm = OpenAIChatLLM() if os.environ.get("LLM_API_KEY") else RuleBasedLLM()
agent = Agent(llm, Tools(plant), monitor=monitor)


class Question(BaseModel):
    question: str


class Decision(BaseModel):
    approver: str
    note: str = ""


@app.get("/health")
def health():
    return {"status": "ok", "llm": type(llm).__name__, "units": len(plant.units())}


@app.get("/units")
def units():
    return plant.units()


@app.get("/units/{unit_id}/status")
def status(unit_id: str):
    try:
        return plant.status(unit_id)
    except KeyError:
        raise HTTPException(404, f"unknown unit {unit_id}")


@app.get("/units/{unit_id}/rul")
def rul(unit_id: str):
    try:
        return plant.rul(unit_id)
    except KeyError:
        raise HTTPException(404, f"unknown unit {unit_id}")


@app.post("/ask")
def ask(q: Question):
    r = agent.run(q.question)
    return {"answer": r.answer, "tools": r.tools_used, "error": r.error,
            "tokens": r.prompt_tokens + r.completion_tokens}


@app.get("/proposals")
def proposals():
    return [p.__dict__ for p in plant.cmms.pending()]


@app.post("/proposals/{proposal_id}/approve")
def approve(proposal_id: str, d: Decision):
    try:
        return {"work_order": plant.cmms.approve(proposal_id, d.approver)}
    except KeyError:
        raise HTTPException(404, "unknown proposal")
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.post("/proposals/{proposal_id}/reject")
def reject(proposal_id: str, d: Decision):
    try:
        plant.cmms.reject(proposal_id, d.approver, d.note)
        return {"status": "rejected"}
    except KeyError:
        raise HTTPException(404, "unknown proposal")


@app.get("/monitoring")
def monitoring():
    return monitor.summary()

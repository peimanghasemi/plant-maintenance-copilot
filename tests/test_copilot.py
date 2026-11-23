import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from plantcopilot.agent.agent import Agent, Monitor
from plantcopilot.agent.llm import LLMReply, RuleBasedLLM
from plantcopilot.agent.tools import Tools
from plantcopilot.cmms import CMMS
from plantcopilot.evaluation import compare_to_baseline, evaluate
from plantcopilot.models.anomaly import evaluate_alarms
from plantcopilot.models.rul import RULModel
from plantcopilot.plant import Plant, priority_from_rul
from plantcopilot.rag.retriever import Retriever
from plantcopilot.simulate import simulate_fleet, truncate_for_test


@pytest.fixture(scope="module")
def plant():
    return Plant.demo()


def test_simulator_ground_truth():
    df = simulate_fleet(8, seed=3)
    for _, g in df.groupby("unit_id"):
        assert g["rul"].iloc[-1] == 0 and (np.diff(g["rul"].values) == -1).all()
    assert (simulate_fleet(4, seed=3, degrade=False)["rul"] == -1).all()


def test_rul_model_beats_a_constant_guess():
    model = RULModel.fit(simulate_fleet(30, seed=0))
    test = truncate_for_test(simulate_fleet(15, seed=1), seed=1)
    m = model.evaluate(test)
    y = test.groupby("unit_id").tail(1)["rul"].clip(upper=400)
    constant_rmse = float(np.sqrt(((y - y.mean()) ** 2).mean()))
    assert m["rmse"] < 0.6 * constant_rmse


def test_alarm_threshold_trades_lead_time_for_false_alarms():
    failing, healthy = simulate_fleet(16, seed=3), simulate_fleet(16, seed=2, degrade=False)
    low, high = evaluate_alarms(failing, healthy, 3), evaluate_alarms(failing, healthy, 10)
    assert low["false_alarm_rate"] >= high["false_alarm_rate"]
    assert low["median_lead_hours"] >= high["median_lead_hours"]


def test_policy_priorities():
    assert [priority_from_rul(x) for x in (10, 60, 200, 500)] == ["urgent", "high", "medium", None]


def test_retriever_finds_the_right_section():
    r = Retriever.from_folder("docs/manuals")
    assert r.search("raw mill pressure drop rising")[0].id == "raw_mill#rising-pressure-drop"
    assert r.search("approval of work orders")[0].id.startswith("maintenance_policy")


def test_work_orders_need_approval():
    cmms = CMMS()
    p = cmms.propose("CEM-000", "inspect", "high", "test")
    assert cmms.work_orders() == []
    wo = cmms.approve(p.id, "engineer")
    assert cmms.work_orders()[0]["id"] == wo
    with pytest.raises(ValueError):
        cmms.approve(p.id, "engineer")          # cannot approve twice
    assert [e["event"] for e in cmms.audit_log()] == ["proposed", "approved"]


def test_agent_only_proposes(plant):
    agent = Agent(RuleBasedLLM(), Tools(plant))
    r = agent.run("Please create a work order for RAW-021 if it needs one.")
    assert "propose_work_order" in r.tools_used and "approval" in r.answer
    assert plant.cmms.work_orders() == []


def test_unknown_unit_is_reported(plant):
    r = Agent(RuleBasedLLM(), Tools(plant)).run("How long until XYZ-999 fails?")
    assert "could not find" in r.answer.lower()


def test_step_limit_stops_a_looping_model(plant):
    class Looper:
        def chat(self, messages, tools):
            return LLMReply(None, [{"id": "x", "name": "search_docs", "arguments": {"query": "mill"}}], {})
    r = Agent(Looper(), Tools(plant), max_steps=3).run("anything")
    assert r.error == "step limit reached" and len(r.tool_calls) == 3


def test_evaluation_passes_and_catches_a_bad_policy(tmp_path):
    plant = Plant.demo()
    good = evaluate(Agent(RuleBasedLLM(), Tools(plant), monitor=Monitor(tmp_path / "a.jsonl")), plant)
    assert good["summary"]["task_success"] == 1.0
    assert good["summary"]["work_orders_written_without_approval"] == 0
    base = tmp_path / "baseline.json"
    base.write_text(json.dumps(good))

    class OverEager(RuleBasedLLM):
        """Proposes an urgent work order for every unit it sees."""
        def chat(self, messages, tools):
            reply = super().chat(messages, tools)
            if reply.tool_calls and reply.tool_calls[0]["name"] == "propose_work_order":
                reply.tool_calls[0]["arguments"]["priority"] = "urgent"
            return reply

    plant2 = Plant.demo()
    bad = evaluate(Agent(OverEager(), Tools(plant2), monitor=Monitor(tmp_path / "b.jsonl")), plant2)
    assert bad["summary"]["priority_accuracy"] < 1.0
    assert compare_to_baseline(bad, base)


def test_api_approval_flow():
    from plantcopilot import api
    client = TestClient(api.app)
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/units/NOPE-1/rul").status_code == 404
    r = client.post("/ask", json={"question": "Create a work order for KIL-022 if needed."}).json()
    assert "propose_work_order" in r["tools"]
    pid = client.get("/proposals").json()[0]["id"]
    assert client.post(f"/proposals/{pid}/approve", json={"approver": "M. Huber"}).json()["work_order"].startswith("WO-")
    assert client.get("/proposals").json() == []
    assert client.get("/monitoring").json()["runs"] >= 1

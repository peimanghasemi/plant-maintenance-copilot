# plant-maintenance-copilot

Predictive maintenance for rotating equipment in a cement plant, plus an LLM assistant that answers maintenance questions, cites the manuals and proposes work orders that a maintenance engineer has to approve before anything reaches the CMMS.

The point of the project is the whole path, not a single model. That means sensor data, then remaining useful life and anomaly alarms, then a tool-calling assistant, then a human approval gate, then evaluation and monitoring. Each part is kept simple enough to replace: the TF-IDF retriever with a vector index, the stand-in CMMS with a real one, the offline policy with any OpenAI-compatible model.

## Components

| Part | File | What it does |
|---|---|---|
| Data | `simulate.py` | Run-to-failure sensor data (vibration, bearing temperature, motor current, pressure drop) for cement mills, raw mills, kiln ID fans and cooler fans, with unit-to-unit differences, daily load cycles and sensor glitches |
| RUL model | `models/rul.py` | Gradient boosting on rolling features (level, spread, slope, drift against the unit's own first day) |
| Alarms | `models/anomaly.py` | Robust z-scores against each unit's healthy baseline; an alarm needs 6 consecutive hours above the threshold |
| Retrieval | `rag/retriever.py` | Manuals and the maintenance policy split by section, each section with a stable id to cite |
| Assistant | `agent/` | Tool-calling loop with a step limit. Tools: `get_unit_status`, `predict_rul`, `search_docs`, `propose_work_order` |
| Approval gate | `cmms.py` | The assistant can only create *proposals*. A work order exists only after a named person approves it. Every step goes to an audit log |
| Evaluation | `evaluation.py`, `eval/cases.yaml` | 18 cases with checks on tool use, citations, required statements, proposal decisions and priorities |
| Monitoring | `agent/agent.py` (`Monitor`) | One JSON line per run with tools, tokens, estimated cost, latency, errors and proposals |
| API | `api.py` | FastAPI endpoints for status, RUL, `/ask`, the proposal queue, approve/reject, and monitoring |

## Data

The sensor data is simulated, so the ground truth (remaining useful life) is known exactly. Real plant data is not public. The manuals in `docs/manuals` are short, generic texts written for this project, not vendor documentation. To use real data, replace `simulate_fleet` with a loader that returns the same columns.

## Results

All numbers come from `scripts/train.py` and `scripts/run_eval.py`.

**Remaining useful life** (trained on 60 simulated units, tested on 30 other units cut at a random point before failure; RUL capped at 400 h)

| RMSE (h) | MAE (h) | RMSE where true RUL < 250 h | share of too-optimistic predictions |
|---|---|---|---|
| 50.1 | 35.8 | 37.0 | 30% |

The error is largest far from failure. In the range where the maintenance policy acts (9 test units below 250 h), the error is smaller, but early and late predictions are about evenly split. The policy thresholds therefore keep a margin, and an anomaly alarm triggers an inspection even when the predicted RUL is still high.

**Alarm threshold** (40 failing units, 40 healthy units that never degrade)

| threshold | detection rate | median lead (h) | 10% lead (h) | false alarm rate |
|---|---|---|---|---|
| 3 | 1.00 | 706 | 474 | 0.175 |
| 4 | 1.00 | 551 | 388 | 0.075 |
| 5 | 1.00 | 466 | 316 | 0.000 |
| 6 | 1.00 | 424 | 247 | 0.000 |
| 8 | 1.00 | 329 | 211 | 0.000 |
| 10 | 1.00 | 278 | 197 | 0.000 |

The assistant uses threshold 6. That is one step above the lowest threshold without false alarms on healthy units, and it still warns at least 10 days ahead in 90% of the failures.

**Assistant evaluation**

| | task success | tool selection | citations | proposal decision | priority | work orders without approval |
|---|---|---|---|---|---|---|
| offline policy | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0 |
| faulty policy (always "urgent") | 0.83 | 1.00 | 1.00 | 1.00 | 0.40 | 0 |

The offline policy is a fixed tool plan, not a language model. It exists so that the loop, the tools, the approval gate and the evaluation run in CI without an API key. Its perfect score shows that the harness and the cases agree with the policy, not that an LLM is good. The faulty policy makes the same decisions but always says "urgent". It shows that the evaluation catches the problem: priority accuracy drops to 0.40 and the regression check fails. To evaluate a real model:

```bash
export LLM_API_KEY=...            # plus LLM_BASE_URL / LLM_MODEL for Azure OpenAI or other endpoints
python scripts/run_eval.py --llm openai
```

## Running it

```bash
pip install -e ".[dev,llm]"
python scripts/train.py           # RUL model, alarm table
pytest
python scripts/run_eval.py        # evaluation + regression check against eval/baseline.json
uvicorn plantcopilot.api:app --port 8000
```

```bash
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "Create a work order for KIL-022 if needed."}'
curl localhost:8000/proposals
curl -X POST localhost:8000/proposals/<id>/approve -H "Content-Type: application/json" -d '{"approver": "M. Huber"}'
curl localhost:8000/monitoring
```

Docker: `docker build -t plant-copilot . && docker run -p 8000:8000 plant-copilot`

CI (`.github/workflows/ci.yml`) trains the model, runs the tests and fails the build if an evaluation case that passed in the baseline now fails or if a work order was written without approval.

## Design choices

- **Propose, never write.** Only `cmms.approve` creates work orders, and no tool calls it. Writing back to an operational system is a decision for a person.
- **Simple where simple works.** The alarm is a robust z-score with a persistence rule, not a deep model. It is easy to explain to a shift leader, and the trade-off table shows what each threshold costs.
- **Policy in documents, numbers in tools.** Priorities come from the maintenance policy applied to the RUL tool output, so the assistant does not invent thresholds. The evaluation checks exactly that.
- **Replaceable parts.** The retriever, the LLM and the CMMS sit behind small interfaces, so Azure AI Search, Azure OpenAI or a real CMMS/SAP PM connection can be plugged in without changing the agent.

## License

MIT

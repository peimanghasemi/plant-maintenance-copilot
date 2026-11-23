"""A small CMMS stand-in with a human approval gate.

The assistant may only create *proposals*. A proposal becomes a work order in
the CMMS only after a named person approves it. Every step is written to an
audit log. Nothing in the agent code path can call `_write_work_order`.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass

PRIORITIES = ("low", "medium", "high", "urgent")


@dataclass
class Proposal:
    id: str
    unit_id: str
    action: str
    priority: str
    reason: str
    status: str = "pending"          # pending, approved, rejected
    created_at: float = 0.0
    decided_by: str | None = None


class CMMS:
    def __init__(self, path: str = ":memory:"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS proposals (id TEXT PRIMARY KEY, body TEXT);
            CREATE TABLE IF NOT EXISTS work_orders (
                id TEXT PRIMARY KEY, unit_id TEXT, action TEXT, priority TEXT,
                reason TEXT, approved_by TEXT, created_at REAL);
            CREATE TABLE IF NOT EXISTS audit (ts REAL, event TEXT, body TEXT);
        """)

    # --- used by the assistant ------------------------------------------------
    def propose(self, unit_id: str, action: str, priority: str, reason: str) -> Proposal:
        if priority not in PRIORITIES:
            raise ValueError(f"priority must be one of {PRIORITIES}")
        p = Proposal(uuid.uuid4().hex[:8], unit_id, action, priority, reason, created_at=time.time())
        self.db.execute("INSERT INTO proposals VALUES (?, ?)", (p.id, json.dumps(asdict(p))))
        self._audit("proposed", asdict(p))
        return p

    # --- used by people (API / UI) ---------------------------------------------
    def pending(self) -> list[Proposal]:
        return [p for p in self._proposals() if p.status == "pending"]

    def approve(self, proposal_id: str, approver: str) -> str:
        p = self._get(proposal_id)
        if p.status != "pending":
            raise ValueError(f"proposal {proposal_id} is already {p.status}")
        if not approver:
            raise ValueError("an approver name is required")
        p.status, p.decided_by = "approved", approver
        self._save(p)
        wo = self._write_work_order(p, approver)
        self._audit("approved", {"proposal": p.id, "work_order": wo, "by": approver})
        return wo

    def reject(self, proposal_id: str, approver: str, note: str = "") -> None:
        p = self._get(proposal_id)
        p.status, p.decided_by = "rejected", approver
        self._save(p)
        self._audit("rejected", {"proposal": p.id, "by": approver, "note": note})

    def work_orders(self) -> list[dict]:
        cur = self.db.execute("SELECT id, unit_id, action, priority, approved_by FROM work_orders")
        return [dict(zip(("id", "unit_id", "action", "priority", "approved_by"), r)) for r in cur]

    def audit_log(self) -> list[dict]:
        return [{"ts": ts, "event": e, **json.loads(b)} for ts, e, b in self.db.execute("SELECT * FROM audit")]

    # --- internal -------------------------------------------------------------
    def _write_work_order(self, p: Proposal, approver: str) -> str:
        wo = "WO-" + uuid.uuid4().hex[:6].upper()
        self.db.execute("INSERT INTO work_orders VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (wo, p.unit_id, p.action, p.priority, p.reason, approver, time.time()))
        self.db.commit()
        return wo

    def _proposals(self) -> list[Proposal]:
        return [Proposal(**json.loads(b)) for (b,) in self.db.execute("SELECT body FROM proposals")]

    def _get(self, pid: str) -> Proposal:
        row = self.db.execute("SELECT body FROM proposals WHERE id = ?", (pid,)).fetchone()
        if row is None:
            raise KeyError(pid)
        return Proposal(**json.loads(row[0]))

    def _save(self, p: Proposal) -> None:
        self.db.execute("UPDATE proposals SET body = ? WHERE id = ?", (json.dumps(asdict(p)), p.id))
        self.db.commit()

    def _audit(self, event: str, body: dict) -> None:
        self.db.execute("INSERT INTO audit VALUES (?, ?, ?)", (time.time(), event, json.dumps(body)))
        self.db.commit()

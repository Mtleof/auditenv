"""Demo: a small hand-scripted auditor policy in OPEN (training) mode.

It reads the payments table, picks payments whose allocations do not add up, opens the payment
document, double-checks the remainder with the calculator and submits. Run:
    PYTHONPATH=src python scripts/demo.py [seed]"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from auditenv.env import AuditEnv  # noqa: E402


class ScriptedAllocationAuditor:
    name = "scripted_allocation_auditor"

    def reset(self, obs):
        self.episode_id = obs["episode_id"]
        self.payments, self.cursor, self.done_reading = [], None, False
        self.suspects, self.findings, self.pending = [], [], None

    def act(self, obs):
        res = obs.get("result") or {}
        if obs.get("action") == "list_records":
            self.payments += res["records"]
            self.cursor = res["next_cursor"]
            if self.cursor is None:
                self.done_reading = True
                self.suspects = [p for p in self.payments
                                 if p["amount_minor"] != sum(x["amount_minor"] for x in p["invoice_refs"])]
        if not self.done_reading:
            return {"action": "list_records", "table": "payments", "cursor": self.cursor, "limit": 50}
        if obs.get("action") == "calculator" and self.pending:
            p = self.pending
            self.findings.append({
                "type": "missing_or_partial_allocation",
                "affected_ids": [p["payment_id"]] + [x["invoice_id"] for x in p["invoice_refs"]],
                "claimed_amount_minor": abs(int(res["value"])), "currency": p["currency"],
                "evidence_ids": [p["doc_id"]], "explanation": "payment amount exceeds its allocations"})
            self.pending = None
        if obs.get("action") == "get_document":
            self.pending = res["document"]
            refs = " + ".join(str(x["amount_minor"]) for x in self.pending["invoice_refs"]) or "0"
            return {"action": "calculator", "expression": f"{self.pending['amount_minor']} - ({refs})"}
        if self.suspects:
            return {"action": "get_document", "doc_id": self.suspects.pop(0)["doc_id"]}
        return {"action": "submit", "episode_id": self.episode_id, "findings": self.findings}


def main(seed: int = 2024) -> int:
    cfg = {"interactions": {"fx_partial_boundary": 1, "duplicate_cross_period": 0}}
    env = AuditEnv(mode="open")
    obs, info = env.reset(seed, cfg)
    m = obs["manifest"]
    print(f"episode {obs['episode_id']}: base {m['base_currency']}, period {m['period_start']}..{m['period_end']} "
          f"(tz {m['timezone']}), {m['counts']['ledger_rows']} ledger rows, {m['counts']['documents']} documents")
    pol = ScriptedAllocationAuditor()
    pol.reset(obs)
    while True:
        action = pol.act(obs)
        obs, r, term, trunc, info = env.step(action)
        brief = {k: v for k, v in action.items() if k != "findings"}
        res = obs["result"] or {}
        what = (f"{len(res['records'])} records" if "records" in res else res.get("value") or
                (res.get("document") or {}).get("payment_id") or res.get("status") or obs["error"])
        print(f"step {obs['step']:>3} | {json.dumps(brief)[:90]:<90} | {what}")
        if term or trunc:
            break
    print("grade:", json.dumps(info["grade"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 2024))

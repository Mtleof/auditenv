"""Baseline policies. All share one interface:

    policy.reset(first_observation) -> None
    policy.act(observation)         -> action (JSON dict)

random_explorer, rule_based and flag_all see only observations. oracle is evaluator-only: it is
constructed with the hidden truth and must never be shipped to an agent process."""
from __future__ import annotations

import random

from .. import FINDING_TYPES
from ..tools.money import convert_minor
from ..tools.timeutil import local_date

TABLE_ORDER = ("invoices", "payments", "fx", "ledger")


class Policy:
    name = "base"

    def reset(self, obs: dict) -> None:
        raise NotImplementedError

    def act(self, obs: dict) -> dict:
        raise NotImplementedError


def _submit(episode_id, findings):
    return {"action": "submit", "episode_id": episode_id, "findings": findings[:30]}


class _Reader(Policy):
    """Helper: reads whole tables page by page."""

    tables: tuple = TABLE_ORDER

    def reset(self, obs):
        self.episode_id = obs["episode_id"]
        self.manifest = obs["manifest"]
        self.data = {t: [] for t in TABLE_ORDER}
        self.queue = list(self.tables)
        self.cursor = None

    def _absorb(self, obs):
        res = obs.get("result") or {}
        if obs.get("action") == "list_records" and "records" in res:
            self.data[res["table"]].extend(res["records"])
            self.cursor = res["next_cursor"]
            if self.cursor is None:
                self.queue.pop(0)

    def _next_read(self):
        return {"action": "list_records", "table": self.queue[0], "cursor": self.cursor, "limit": 50}


class RuleBasedPolicy(_Reader):
    """Simple reconciliation: read everything, then apply four checks.

    Deliberate simplifications (documented baseline weaknesses):
      * duplicates are only detected when both invoices share the same local issue date;
      * FX is only re-checked on invoice postings, not on payment postings."""
    name = "rule_based"

    def act(self, obs):
        if obs.get("step", 0) > 0:
            self._absorb(obs)
        if self.queue and obs["steps_remaining"] > 1:
            return self._next_read()
        return _submit(self.episode_id, analyse(self.manifest, self.data))


def analyse(man: dict, data: dict) -> list[dict]:
    tz, base, exp = man["timezone"], man["base_currency"], man["currency_exponents"]
    start, end = man["period_start"], man["period_end"]
    mode = man["rounding"]["mode"]
    fx = {r["fx_id"]: r["rate"] for r in data["fx"]}
    ld = lambda ts: local_date(ts, tz).isoformat()
    findings = []

    def add(typ, ids, amount, cur, ev, why):
        findings.append({"type": typ, "affected_ids": sorted(ids), "claimed_amount_minor": int(amount),
                         "currency": cur, "evidence_ids": sorted(ev)[:10], "explanation": why})

    # 1. duplicates: same entity, currency, total, line items and same local issue date
    groups: dict = {}
    for inv in data["invoices"]:
        items = tuple(sorted(tuple(sorted(i.items())) for i in inv["line_items"]))
        groups.setdefault((inv["entity_id"], inv["currency"], inv["total_minor"], items, ld(inv["issued_at"])),
                          []).append(inv)
    for g in groups.values():
        if len(g) == 2:
            add("duplicate_invoice", [g[0]["invoice_id"], g[1]["invoice_id"]], g[0]["total_minor"],
                g[0]["currency"], [g[0]["doc_id"], g[1]["doc_id"]], "identical invoice issued twice")
    # 2. allocation
    for p in data["payments"]:
        s = sum(x["amount_minor"] for x in p["invoice_refs"])
        if s != p["amount_minor"]:
            add("missing_or_partial_allocation", [p["payment_id"]] + [x["invoice_id"] for x in p["invoice_refs"]],
                abs(p["amount_minor"] - s), p["currency"], [p["doc_id"]], "payment not fully allocated")
    # 3/4. invoice postings: FX rate and period
    rows_by_inv: dict = {}
    for r in data["ledger"]:
        if r["invoice_id"] and not r["payment_id"]:
            rows_by_inv.setdefault(r["invoice_id"], []).append(r)
    for inv in data["invoices"]:
        rows = rows_by_inv.get(inv["invoice_id"], [])
        dr = sorted((r for r in rows if r["debit_minor"]), key=lambda r: r["row_id"])
        if not dr:
            continue
        d = ld(inv["issued_at"])
        if not (start <= d <= end):
            add("period_misattribution", [inv["invoice_id"]], sum(r["debit_minor"] for r in dr), base,
                [inv["doc_id"], dr[0]["row_id"]], "document dated outside the period")
            continue
        if inv["currency"] == base:
            continue
        key = f"FX:{d}:{inv['currency']}:{base}"
        if key not in fx:
            continue
        expected = convert_minor(inv["total_minor"], exp[inv["currency"]], fx[key], exp[base], mode)
        diff = sum(abs(r["debit_minor"] - expected) for r in dr)
        if diff:
            add("wrong_fx_date_or_rate", [inv["invoice_id"]], diff, base,
                [inv["doc_id"], key] + [r["row_id"] for r in dr], "booked amount does not match policy rate")
    findings.sort(key=lambda f: (f["type"], f["affected_ids"]))
    return findings


class FlagAllPolicy(_Reader):
    """Attack: accuse every document of something, up to the 30-finding limit."""
    name = "flag_all"
    tables = ("invoices", "payments")

    def act(self, obs):
        if obs.get("step", 0) > 0:
            self._absorb(obs)
        if self.queue and obs["steps_remaining"] > 1:
            return self._next_read()
        base = self.manifest["base_currency"]
        out = []
        for p in self.data["payments"][:10]:
            out.append({"type": "missing_or_partial_allocation", "affected_ids": [p["payment_id"]],
                        "claimed_amount_minor": abs(p["amount_minor"]), "currency": p["currency"],
                        "evidence_ids": [p["doc_id"]], "explanation": "flag"})
        types = ("duplicate_invoice", "period_misattribution", "wrong_fx_date_or_rate")
        for k, inv in enumerate(self.data["invoices"]):
            t = types[k % 3]
            out.append({"type": t, "affected_ids": [inv["invoice_id"]], "claimed_amount_minor": inv["total_minor"],
                        "currency": inv["currency"] if t == "duplicate_invoice" else base,
                        "evidence_ids": [inv["doc_id"]], "explanation": "flag"})
        return _submit(self.episode_id, out[:30])


class RandomExplorerPolicy(Policy):
    """Random but legal-looking actions with a fixed seed, then a random submission."""
    name = "random"

    def __init__(self, seed: int = 20251121):
        self.seed = seed

    def reset(self, obs):
        self.episode_id = obs["episode_id"]
        self.base = obs["manifest"]["base_currency"]
        self.rng = random.Random(f"{self.seed}|{obs['episode_id']}")
        self.budget = self.rng.randint(3, 30)
        self.seen = {"ledger": [], "invoices": [], "payments": [], "fx": [], "docs": []}

    def _absorb(self, obs):
        res = obs.get("result") or {}
        recs = res.get("records") or ([res["record"]] if "record" in res else [])
        if "document" in res:
            recs = [res["document"]]
        for rec in recs:
            if "row_id" in rec:
                self.seen["ledger"].append(rec["row_id"])
            elif "fx_id" in rec:
                self.seen["fx"].append(rec["fx_id"])
            elif "line_items" in rec:
                self.seen["invoices"].append(rec["invoice_id"])
            elif "payment_id" in rec:
                self.seen["payments"].append(rec["payment_id"])
            if "doc_id" in rec:
                self.seen["docs"].append(rec["doc_id"])

    def act(self, obs):
        r = self.rng
        if obs.get("step", 0) > 0:
            self._absorb(obs)
        if obs["step"] >= self.budget or obs["steps_remaining"] <= 1:
            return self._random_submit()
        kind = r.choice(("list", "list", "get", "doc", "calc", "bad"))
        table = r.choice(("ledger", "invoices", "payments", "fx"))
        if kind == "list":
            return {"action": "list_records", "table": table, "cursor": f"o{r.randint(0, 300)}" if r.random() < .5
                    else None, "limit": r.randint(1, 80)}
        if kind == "get" and self.seen[table]:
            return {"action": "get_record", "table": table, "id": r.choice(self.seen[table])}
        if kind == "doc" and self.seen["docs"]:
            return {"action": "get_document", "doc_id": r.choice(self.seen["docs"])}
        if kind == "calc":
            return {"action": "calculator", "expression": f"{r.randint(1, 999)}*{r.randint(1, 99)}.{r.randint(0, 99)}"}
        return {"action": "get_record", "table": table, "id": f"X-{r.getrandbits(24):06X}"}

    def _random_submit(self):
        r = self.rng
        out = []
        pool = self.seen["invoices"] + self.seen["payments"]
        ev_pool = self.seen["docs"] + self.seen["ledger"] + self.seen["fx"]
        for _ in range(r.randint(0, 3) if pool else 0):
            out.append({"type": r.choice(FINDING_TYPES), "affected_ids": r.sample(pool, min(len(pool), r.randint(1, 2))),
                        "claimed_amount_minor": r.randint(0, 10**6), "currency": self.base,
                        "evidence_ids": r.sample(ev_pool, min(len(ev_pool), r.randint(0, 4))), "explanation": "random"})
        return _submit(self.episode_id, out)


class OraclePolicy(Policy):
    """EVALUATOR-ONLY upper bound: submits the truth with its first admissible evidence set."""
    name = "oracle"

    def __init__(self, truth: dict):
        self._truth = truth

    def reset(self, obs):
        self.episode_id = obs["episode_id"]

    def act(self, obs):
        return _submit(self.episode_id, [
            {"type": i["type"], "affected_ids": list(i["affected_ids"]), "claimed_amount_minor": i["expected_amount_minor"],
             "currency": i["currency"], "evidence_ids": list(i["evidence_sets"][0]), "explanation": "oracle"}
            for i in self._truth["issues"]])


AGENT_POLICIES = {"random": RandomExplorerPolicy, "rule_based": RuleBasedPolicy, "flag_all": FlagAllPolicy}


def make_policy(name: str, truth: dict | None = None) -> Policy:
    if name == "oracle":
        if truth is None:
            raise ValueError("oracle needs the truth and is evaluator-only")
        return OraclePolicy(truth)
    return AGENT_POLICIES[name]()


__all__ = ["Policy", "RuleBasedPolicy", "FlagAllPolicy", "RandomExplorerPolicy", "OraclePolicy",
           "AGENT_POLICIES", "make_policy", "analyse"]

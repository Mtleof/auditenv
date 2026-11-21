"""Independent validator. Works ONLY from serialized public files (plus truth for checking),
re-implementing parsing and conversion so that generator bugs are not masked by shared code.

(a) accounting identities hold on the clean (pre-injection) public files;
(b) after injection, an exhaustive detector finds exactly the truth issues (no more, no less);
(c) every admissible evidence set alone is sufficient to re-derive the issue (type, ids, amount)."""
from __future__ import annotations

import csv
import io
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from decimal import Decimal, localcontext, ROUND_HALF_EVEN, ROUND_HALF_UP

from ..schemas import FX_SCHEMA, INVOICE_SCHEMA, LEDGER_ROW_SCHEMA, PAYMENT_SCHEMA, validate

_ROUND = {"ROUND_HALF_EVEN": ROUND_HALF_EVEN, "ROUND_HALF_UP": ROUND_HALF_UP}


class View:
    """Parsed public records (possibly restricted to a subset of record ids)."""

    def __init__(self, files: dict[str, str]):
        self.manifest = json.loads(files["manifest.json"])
        self.policy = json.loads(files["policy.json"])
        self.ledger = []
        for r in csv.DictReader(io.StringIO(files["ledger.csv"])):
            r = {k: (None if v == "" else v) for k, v in r.items()}
            r["debit_minor"] = int(r["debit_minor"]); r["credit_minor"] = int(r["credit_minor"])
            self.ledger.append(r)
        self.invoices = [json.loads(x) for x in files["invoices.jsonl"].splitlines() if x]
        self.payments = [json.loads(x) for x in files["payments.jsonl"].splitlines() if x]
        self.fx = {}
        for r in csv.DictReader(io.StringIO(files["fx.csv"])):
            self.fx[f"FX:{r['date']}:{r['source_currency']}:{r['target_currency']}"] = r
        self._index()

    def _index(self):
        m = self.manifest
        sign = 1 if m["timezone"][0] == "+" else -1
        hh, mm = m["timezone"][1:].split(":")
        self.tz = timezone(sign * timedelta(hours=int(hh), minutes=int(mm)))
        self.base = m["base_currency"]
        self.exp = m["currency_exponents"]
        self.start = datetime.fromisoformat(m["period_start"]).date()
        self.end = datetime.fromisoformat(m["period_end"]).date()
        self.inv_by_id = {i["invoice_id"]: i for i in self.invoices}
        self.pay_by_id = {p["payment_id"]: p for p in self.payments}
        self.doc_by_id = {**{i["doc_id"]: ("inv", i) for i in self.invoices},
                          **{p["doc_id"]: ("pay", p) for p in self.payments}}
        self.row_by_id = {r["row_id"]: r for r in self.ledger}

    def restrict(self, ids: list[str]) -> "View":
        v = object.__new__(View)
        v.manifest, v.policy = self.manifest, self.policy
        ids = set(ids)
        v.ledger = [r for r in self.ledger if r["row_id"] in ids]
        v.invoices = [i for i in self.invoices if i["doc_id"] in ids]
        v.payments = [p for p in self.payments if p["doc_id"] in ids]
        v.fx = {k: r for k, r in self.fx.items() if k in ids}
        v._index()
        return v

    def all_ids(self) -> set[str]:
        return set(self.row_by_id) | set(self.doc_by_id) | set(self.fx)

    def local_date(self, ts: str):
        dt = datetime.fromisoformat(ts)
        assert dt.tzinfo is not None
        return dt.astimezone(self.tz).date()

    def in_period(self, ts: str) -> bool:
        return self.start <= self.local_date(ts) <= self.end

    def convert(self, amount: int, cur: str, ts: str):
        """Return the base-currency amount for a document amount, or None if the rate is not visible."""
        if cur == self.base:
            return abs(amount)
        key = f"FX:{self.local_date(ts).isoformat()}:{cur}:{self.base}"
        if key not in self.fx:
            return None
        with localcontext() as ctx:
            ctx.prec = 60
            q = Decimal(abs(amount)) * Decimal(self.fx[key]["rate"]) * Decimal(10) ** (self.exp[self.base] - self.exp[cur])
            return int(q.quantize(Decimal(1), rounding=_ROUND[self.manifest["rounding"]["mode"]]))

    # ------------------------------------------------------------------ helpers
    def doc_rows(self, kind: str, doc: dict) -> list[dict]:
        if kind == "inv":
            return [r for r in self.ledger if r["invoice_id"] == doc["invoice_id"] and r["payment_id"] is None]
        return [r for r in self.ledger if r["payment_id"] == doc["payment_id"]]

    def portion(self, kind: str, doc: dict, row: dict):
        if kind == "inv":
            return doc["total_minor"]
        refs = doc["invoice_refs"]
        if row["invoice_id"] is None:
            return abs(doc["amount_minor"] - sum(x["amount_minor"] for x in refs))
        match = [x["amount_minor"] for x in refs if x["invoice_id"] == row["invoice_id"]]
        return abs(match[0]) if len(match) == 1 else None

    def fx_discrepancy(self, kind: str, doc: dict, rows: list[dict]):
        ts = doc["issued_at"] if kind == "inv" else doc["timestamp"]
        out = {"dr": 0, "cr": 0}
        for r in rows:
            p = self.portion(kind, doc, r)
            exp = None if p is None else self.convert(p, doc["currency"], ts)
            if exp is None:
                return None
            side = "dr" if r["debit_minor"] else "cr"
            out[side] += abs(r["debit_minor"] + r["credit_minor"] - exp)
        return out


def dup_key(inv: dict):
    items = sorted((i["sku"], i["description"], i["quantity"], i["unit_price_minor"], i["amount_minor"])
                   for i in inv["line_items"])
    return (inv["entity_id"], inv["currency"], inv["total_minor"], tuple(items))


def _finding(typ, ids, amount, cur):
    return (typ, tuple(sorted(ids)), int(amount), cur)


# ---------------------------------------------------------------------------------------- (a)
def structural_errors(v: View) -> list[str]:
    errs = []
    for r in v.ledger:
        errs += [f"ledger {e}" for e in validate(r, LEDGER_ROW_SCHEMA)]
        if (r["debit_minor"] > 0) == (r["credit_minor"] > 0):
            errs.append("row must be debit xor credit")
        if r["currency"] != v.base:
            errs.append("ledger row not in base currency")
        if r["invoice_id"] is not None and r["invoice_id"] not in v.inv_by_id:
            errs.append("ledger references unknown invoice")
        if r["payment_id"] is not None and r["payment_id"] not in v.pay_by_id:
            errs.append("ledger references unknown payment")
    for i in v.invoices:
        errs += [f"invoice {e}" for e in validate(i, INVOICE_SCHEMA)]
        if i["total_minor"] != sum(x["amount_minor"] for x in i["line_items"]):
            errs.append("invoice total != sum(lines)")
        if any(x["amount_minor"] != x["quantity"] * x["unit_price_minor"] for x in i["line_items"]):
            errs.append("line amount != qty*price")
    for p in v.payments:
        errs += [f"payment {e}" for e in validate(p, PAYMENT_SCHEMA)]
        for x in p["invoice_refs"]:
            inv = v.inv_by_id.get(x["invoice_id"])
            if inv is None or inv["currency"] != p["currency"] or inv["entity_id"] != p["entity_id"]:
                errs.append("payment ref inconsistent")
    for r in v.fx.values():
        errs += [f"fx {e}" for e in validate(r, FX_SCHEMA)]
    ids = [r["row_id"] for r in v.ledger] + [d for d in v.doc_by_id]
    if len(ids) != len(set(ids)) or len(v.inv_by_id) != len(v.invoices) or len(v.pay_by_id) != len(v.payments):
        errs.append("ids not unique")
    c = v.manifest["counts"]
    if (c["ledger_rows"], c["invoices"], c["payments"], c["fx_rates"]) != (
            len(v.ledger), len(v.invoices), len(v.payments), len(v.fx)):
        errs.append("manifest counts mismatch")
    return errs


def identity_errors(v: View) -> list[str]:
    """Accounting identities that must hold on clean data."""
    errs = []
    if sum(r["debit_minor"] for r in v.ledger) != sum(r["credit_minor"] for r in v.ledger):
        errs.append("total debit != total credit")
    for kind, docs in (("inv", v.invoices), ("pay", v.payments)):
        for d in docs:
            rows = v.doc_rows(kind, d)
            if sum(r["debit_minor"] for r in rows) != sum(r["credit_minor"] for r in rows):
                errs.append("document postings unbalanced")
    alloc = Counter()
    for p in v.payments:
        if not v.in_period(p["timestamp"]):
            errs.append("payment outside period")
        if not v.doc_rows("pay", p):
            errs.append("payment not booked")
        for x in p["invoice_refs"]:
            alloc[x["invoice_id"]] += x["amount_minor"]
    for i in v.invoices:
        if not 0 <= alloc[i["invoice_id"]] <= i["total_minor"]:
            errs.append("invoice over-allocated")
        rows = v.doc_rows("inv", i)
        if v.in_period(i["issued_at"]):
            accts = sorted((r["account"], bool(r["debit_minor"])) for r in rows)
            if accts != [("1200", True), ("4000", False)]:
                errs.append("in-period invoice not booked exactly once")
    for r in v.ledger:
        if not v.in_period(r["timestamp"]):
            errs.append("ledger row outside period")
    return errs


# ---------------------------------------------------------------------------------------- (b)
def detect(v: View) -> tuple[list[tuple], list[str]]:
    """Exhaustive detector of the four issue types. Returns (findings, unexpected anomalies)."""
    found, odd = [], []
    for p in v.payments:
        s = sum(x["amount_minor"] for x in p["invoice_refs"])
        if s != p["amount_minor"]:
            found.append(_finding("missing_or_partial_allocation",
                                  [p["payment_id"]] + [x["invoice_id"] for x in p["invoice_refs"]],
                                  abs(p["amount_minor"] - s), p["currency"]))
    groups: dict = {}
    for i in v.invoices:
        groups.setdefault(dup_key(i), []).append(i)
    window = timedelta(days=7)
    for g in groups.values():
        for a in range(len(g)):
            for b in range(a + 1, len(g)):
                ta, tb = datetime.fromisoformat(g[a]["issued_at"]), datetime.fromisoformat(g[b]["issued_at"])
                if abs(ta - tb) <= window:
                    found.append(_finding("duplicate_invoice", [g[a]["invoice_id"], g[b]["invoice_id"]],
                                          g[a]["total_minor"], g[a]["currency"]))
    for i in v.invoices:
        rows = v.doc_rows("inv", i)
        if rows and not v.in_period(i["issued_at"]):
            found.append(_finding("period_misattribution", [i["invoice_id"]],
                                  sum(r["debit_minor"] for r in rows), v.base))
        elif not rows and v.in_period(i["issued_at"]):
            odd.append("unbooked in-period invoice")
        if len(rows) > 2:
            odd.append("invoice booked more than once")
    for kind, docs, key in (("inv", v.invoices, "invoice_id"), ("pay", v.payments, "payment_id")):
        for d in docs:
            rows = v.doc_rows(kind, d)
            if not rows:
                continue
            disc = v.fx_discrepancy(kind, d, rows)
            if disc is None:
                odd.append("fx rate not visible")
                continue
            if disc["dr"] != disc["cr"]:
                odd.append("asymmetric fx discrepancy")
            if disc["dr"]:
                found.append(_finding("wrong_fx_date_or_rate", [d[key]], disc["dr"], v.base))
    if sum(r["debit_minor"] for r in v.ledger) != sum(r["credit_minor"] for r in v.ledger):
        odd.append("ledger unbalanced")
    for r in v.ledger:
        if r["invoice_id"] is None and r["payment_id"] is None and not v.in_period(r["timestamp"]):
            odd.append("non-document row outside period")
    return found, odd


# ---------------------------------------------------------------------------------------- (c)
def prove(typ: str, v: View):
    """Re-derive an issue using only the records in a restricted view. Returns a finding tuple or None."""
    if typ == "duplicate_invoice":
        if len(v.invoices) != 2:
            return None
        a, b = v.invoices
        ta, tb = datetime.fromisoformat(a["issued_at"]), datetime.fromisoformat(b["issued_at"])
        if dup_key(a) == dup_key(b) and abs(ta - tb) <= timedelta(days=7):
            return _finding(typ, [a["invoice_id"], b["invoice_id"]], a["total_minor"], a["currency"])
        return None
    if typ == "missing_or_partial_allocation":
        if len(v.payments) != 1:
            return None
        p = v.payments[0]
        s = sum(x["amount_minor"] for x in p["invoice_refs"])
        if s == p["amount_minor"]:
            return None
        return _finding(typ, [p["payment_id"]] + [x["invoice_id"] for x in p["invoice_refs"]],
                        abs(p["amount_minor"] - s), p["currency"])
    if typ == "wrong_fx_date_or_rate":
        if len(v.doc_by_id) != 1 or not v.ledger:
            return None
        kind, d = next(iter(v.doc_by_id.values()))
        rows = v.doc_rows(kind, d)
        if len(rows) != len(v.ledger):
            return None
        sides = {bool(r["debit_minor"]) for r in rows}
        disc = v.fx_discrepancy(kind, d, rows)
        if disc is None or len(sides) != 1:
            return None
        amount = disc["dr"] + disc["cr"]
        return _finding(typ, [d["invoice_id" if kind == "inv" else "payment_id"]], amount, v.base) if amount else None
    if typ == "period_misattribution":
        if len(v.invoices) != 1 or len(v.ledger) != 1:
            return None
        inv, row = v.invoices[0], v.ledger[0]
        if row["invoice_id"] != inv["invoice_id"] or row["payment_id"] is not None or v.in_period(inv["issued_at"]):
            return None
        return _finding(typ, [inv["invoice_id"]], row["debit_minor"] + row["credit_minor"], v.base)
    return None


def validate_files(public: dict[str, str], clean_public: dict[str, str] | None, truth: dict) -> list[str]:
    errs = []
    if clean_public is not None:
        cv = View(clean_public)
        errs += [f"clean/structure: {e}" for e in structural_errors(cv)]
        errs += [f"clean/identity: {e}" for e in identity_errors(cv)]
        f, odd = detect(cv)
        errs += [f"clean/detect: {x[0]}" for x in f] + [f"clean/odd: {o}" for o in odd]
    v = View(public)
    errs += [f"public/structure: {e}" for e in structural_errors(v)]
    found, odd = detect(v)
    errs += [f"public/odd: {o}" for o in odd]
    expected = Counter(_finding(i["type"], i["affected_ids"], i["expected_amount_minor"], i["currency"])
                       for i in truth["issues"])
    got = Counter(found)
    if got != expected:
        errs.append(f"detect mismatch: missing={sorted(expected - got)} extra={sorted(got - expected)}")
    known = v.all_ids()
    blob = "".join(public.values())
    for i in truth["issues"]:
        if i["issue_id"] in blob:
            errs.append("issue id leaked into public files")
        if not i["evidence_sets"]:
            errs.append("issue without evidence")
        target = _finding(i["type"], i["affected_ids"], i["expected_amount_minor"], i["currency"])
        for es in i["evidence_sets"]:
            if not set(es) <= known:
                errs.append("evidence references unknown record")
                continue
            if prove(i["type"], v.restrict(es)) != target:
                errs.append(f"evidence set insufficient for {i['type']}")
    return errs


def validate_episode(ep) -> list[str]:
    return validate_files(ep.public_files(), ep.clean_public_files(), ep.truth)


__all__ = ["View", "validate_episode", "validate_files", "detect", "prove", "structural_errors", "identity_errors"]

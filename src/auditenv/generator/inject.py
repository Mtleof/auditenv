"""Step 3-4 of generation: inject errors (single and interacting) into the clean history and
record the causal change in provenance. Each injected issue gets a truth record whose
expected amount and evidence sets are computed from the *public* records it changes."""
from __future__ import annotations

import copy
from datetime import timedelta
from decimal import Decimal

from .. import FINDING_TYPES
from ..tools.money import CURRENCY_EXPONENTS, convert_minor
from ..tools.timeutil import fmt_ts
from .history import ROUNDING, Builder, GeneratorError

MIN_FX_DISCREPANCY = 10


def fx_id(d, cur: str, base: str) -> str:
    return f"FX:{d.isoformat()}:{cur}:{base}"


class Injector:
    def __init__(self, b: Builder):
        self.b = b
        self.issues: list[dict] = []
        self.mutations: list[dict] = []
        self.used_inv: set[str] = set()
        self.used_pay: set[str] = set()
        for s in b.slots["fx_partial_boundary"]:
            self.used_inv.add(s["invoice"]); self.used_pay.add(s["payment"])
        for s in b.slots["duplicate_cross_period"]:
            self.used_inv.add(s["invoice"])

    # ------------------------------------------------------------------ helpers
    def issue(self, typ, affected, amount, cur, evidence_sets, scenario) -> dict:
        iss = {"issue_id": self.b.new_id("ISS", 8), "type": typ, "affected_ids": sorted(affected),
               "expected_amount_minor": int(amount), "currency": cur,
               "evidence_sets": [sorted(s) for s in evidence_sets], "scenario": scenario}
        self.issues.append(iss)
        return iss

    def _sorted(self, items, meta):
        return sorted(items, key=lambda x: meta[x]["seq"])

    def pool(self, kind: str) -> list[str]:
        b = self.b
        if kind == "regular_inv":
            ids = [i for i, m in b.inv_meta.items() if "regular" in m["tags"] and i not in self.used_inv]
            return self._sorted(ids, b.inv_meta)
        if kind == "regular_pay":
            ids = [p for p, m in b.pay_meta.items() if "regular_full" in m["tags"] and p not in self.used_pay
                   and b.pay(p)["invoice_refs"][0]["invoice_id"] not in self.used_inv]
            return self._sorted(ids, b.pay_meta)
        if kind == "next_inv":
            ids = [i for i, m in b.inv_meta.items() if m["kind"] == "next" and i not in self.used_inv]
            return self._sorted(ids, b.inv_meta)
        raise KeyError(kind)

    def side_rows(self, jes: list[dict], side: str) -> list[str]:
        out = []
        for je in jes:
            for r in self.b.je_rows(je):
                if (side == "dr" and r["debit_minor"]) or (side == "cr" and r["credit_minor"]):
                    out.append(r["row_id"])
        return out

    # ------------------------------------------------------------------ duplicate_invoice
    def duplicate(self, inv_id: str, cross_period: bool, scenario: str) -> dict | None:
        b, r = self.b, self.b.rng
        src = b.inv(inv_id)
        m = b.inv_meta[inv_id]
        if cross_period:
            t = b.end_instant + timedelta(minutes=r.randint(60, 3 * 1440))
        else:
            room = int((min(m["instant"] + timedelta(days=2), b.end_instant - timedelta(minutes=1))
                        - m["instant"]).total_seconds() // 60)
            if room < 30:
                return None
            t = m["instant"] + timedelta(minutes=r.randint(30, room))
        items = copy.deepcopy(src["line_items"])
        r.shuffle(items)
        dup = b.add_invoice(src["entity_id"], src["currency"], t, items, "next" if cross_period else "in",
                            ("injected_duplicate",))
        self.used_inv.update({inv_id, dup["invoice_id"]})
        iss = self.issue("duplicate_invoice", [inv_id, dup["invoice_id"]], src["total_minor"], src["currency"],
                         [[src["doc_id"], dup["doc_id"]]], scenario)
        mut = {"op": "insert_duplicate_invoice", "issue_id": iss["issue_id"], "source_invoice": inv_id,
               "duplicate_invoice": dup["invoice_id"], "duplicate_doc": dup["doc_id"]}
        if not cross_period:
            je = b.book_invoice(dup)
            mut["ledger_rows_added"] = list(je["rows"])
        self.mutations.append(mut)
        return dup

    # ------------------------------------------------------------------ period_misattribution
    def misbook(self, inv_id: str, scenario: str) -> None:
        b, r = self.b, self.b.rng
        inv = b.inv(inv_id)
        m = b.inv_meta[inv_id]
        variant = r.choice(("backdated", "true_timestamp"))
        t = b.end_instant - timedelta(minutes=r.randint(1, 600)) if variant == "backdated" else m["instant"]
        je = b.book_invoice(inv, instant=t)
        self.used_inv.add(inv_id)
        amt = b.je_rows(je)[0]["debit_minor"]
        iss = self.issue("period_misattribution", [inv_id], amt, b.base,
                         [[inv["doc_id"], je["rows"][0]], [inv["doc_id"], je["rows"][1]]], scenario)
        self.mutations.append({"op": "book_out_of_period_invoice", "variant": variant, "issue_id": iss["issue_id"],
                               "invoice": inv_id, "ledger_rows_added": list(je["rows"]),
                               "ledger_timestamp": fmt_ts(t, b.tz)})

    # ------------------------------------------------------------------ missing_or_partial_allocation
    def misallocate(self, pay_id: str, variant: str, scenario: str) -> None:
        b, r = self.b, self.b.rng
        p = b.pay(pay_id)
        ref = p["invoice_refs"][0]
        inv_id = ref["invoice_id"]
        jes = b.doc_jes("pay", pay_id)
        assert len(jes) == 1
        je = jes[0]
        before = copy.deepcopy(p["invoice_refs"])
        if variant == "missing":
            p["invoice_refs"] = []
            for row in b.je_rows(je):
                row["invoice_id"] = None
                if row["account"] == "1200":
                    row["account"] = "2900"
            amount, affected = p["amount_minor"], [pay_id]
        else:
            a = ref["amount_minor"]
            new_a = max(1, a * r.randint(30, 80) // 100)
            rem = a - new_a
            ref["amount_minor"] = new_a
            d = je["rate_date"]
            je["portion"] = new_a
            b.set_je_amount(je, b.conv(new_a, p["currency"], d))
            m = b.pay_meta[pay_id]
            b.post_je(m["instant"], "1000", "2900", b.conv(rem, p["currency"], d), p["entity_id"], None, pay_id,
                      ("pay", pay_id), rem, p["currency"], d)
            amount, affected = rem, [pay_id, inv_id]
        self.used_pay.add(pay_id); self.used_inv.add(inv_id)
        iss = self.issue("missing_or_partial_allocation", affected, amount, p["currency"], [[p["doc_id"]]], scenario)
        self.mutations.append({"op": "alter_payment_allocation", "variant": variant, "issue_id": iss["issue_id"],
                               "payment": pay_id, "refs_before": before, "refs_after": copy.deepcopy(p["invoice_refs"])})

    # ------------------------------------------------------------------ wrong_fx_date_or_rate
    def wrong_fx(self, kind: str, doc_key: str, scenario: str, forced_date=None) -> bool:
        b, r = self.b, self.b.rng
        jes = b.doc_jes(kind, doc_key)
        if not jes:
            return False
        cur, d = jes[0]["cur"], jes[0]["rate_date"]
        if cur == b.base:
            return False
        correct = b.rate(cur, d)
        attempts = []
        if forced_date is not None:
            attempts.append(("wrong_date", forced_date))
        else:
            variant = r.choice(("wrong_date", "wrong_rate"))
            if variant == "wrong_date":
                attempts.append(("wrong_date", d + timedelta(days=r.choice((-1, 1)))))
            for k in (r.randint(2, 6), 8, 12):
                attempts.append(("wrong_rate", k * r.choice((-1, 1))))
        for variant, param in attempts:
            if variant == "wrong_date":
                wrong_rate, wrong_date = b.rate(cur, param), param
            else:
                q = Decimal(correct).as_tuple().exponent
                wrong_rate = format((Decimal(correct) * (1 + Decimal(param) / 100)).quantize(Decimal(1).scaleb(q)), "f")
                wrong_date = None
            new = [convert_minor(je["portion"], CURRENCY_EXPONENTS[cur], wrong_rate, CURRENCY_EXPONENTS[b.base],
                                 ROUNDING) for je in jes]
            old = [b.conv(je["portion"], cur, d) for je in jes]
            disc = sum(abs(x - y) for x, y in zip(new, old))
            if disc >= MIN_FX_DISCREPANCY and all(x > 0 for x in new):
                break
        else:
            return False
        for je, amt in zip(jes, new):
            b.set_je_amount(je, amt)
        doc = b.inv(doc_key) if kind == "inv" else b.pay(doc_key)
        (self.used_inv if kind == "inv" else self.used_pay).add(doc_key)
        fid = fx_id(d, cur, b.base)
        iss = self.issue("wrong_fx_date_or_rate", [doc_key], disc, b.base,
                         [[doc["doc_id"], fid, *self.side_rows(jes, "dr")],
                          [doc["doc_id"], fid, *self.side_rows(jes, "cr")]], scenario)
        self.mutations.append({"op": "rebook_with_wrong_fx", "variant": variant, "issue_id": iss["issue_id"],
                               "document": doc_key, "correct_rate": correct, "correct_date": d.isoformat(),
                               "used_rate": wrong_rate,
                               "used_date": wrong_date.isoformat() if wrong_date else None})
        return True

    # ------------------------------------------------------------------ orchestration
    def run(self, counts: dict[str, int]) -> None:
        b, r = self.b, self.b.rng
        # interacting scenario 1: partial allocation of a foreign-currency payment at the period
        # boundary, booked with the rate of its UTC date instead of its local (manifest tz) date
        for s in b.slots["fx_partial_boundary"]:
            self.misallocate(s["payment"], "partial", "fx_partial_boundary")
            if not self.wrong_fx("pay", s["payment"], "fx_partial_boundary", forced_date=s["wrong_date"]):
                raise GeneratorError("fx boundary scenario not provable")
        # interacting scenario 2: invoice re-issued after period end (duplicate) and the duplicate
        # booked into this period's ledger (period misattribution of the duplicate)
        for s in b.slots["duplicate_cross_period"]:
            dup = self.duplicate(s["invoice"], True, "duplicate_cross_period")
            self.misbook(dup["invoice_id"], "duplicate_cross_period")
        for typ in FINDING_TYPES:
            for _ in range(counts.get(typ, 0)):
                ok = False
                if typ == "duplicate_invoice":
                    for cand in r.sample(self.pool("regular_inv"), len(self.pool("regular_inv"))):
                        if self.duplicate(cand, False, "single") is not None:
                            ok = True; break
                elif typ == "missing_or_partial_allocation":
                    pool = self.pool("regular_pay")
                    if pool:
                        self.misallocate(r.choice(pool), r.choice(("missing", "partial")), "single"); ok = True
                elif typ == "wrong_fx_date_or_rate":
                    cands = [("inv", i) for i in self.pool("regular_inv") if b.inv(i)["currency"] != b.base]
                    cands += [("pay", p) for p in self.pool("regular_pay") if b.pay(p)["currency"] != b.base]
                    r.shuffle(cands)
                    for kind, key in cands:
                        if self.wrong_fx(kind, key, "single"):
                            ok = True; break
                elif typ == "period_misattribution":
                    pool = self.pool("next_inv")
                    if pool:
                        self.misbook(r.choice(pool), "single"); ok = True
                if not ok:
                    raise GeneratorError(f"no target for {typ}")

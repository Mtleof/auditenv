"""Step 1-2 of generation: a consistent, hidden *clean* company history and the documents /
ledger postings derived from it. All randomness comes from one local random.Random."""
from __future__ import annotations

import copy
import random
from datetime import date, datetime, timedelta
from decimal import Decimal

from ..tools.money import CURRENCY_EXPONENTS, convert_minor
from ..tools.timeutil import fmt_ts, offset_minutes, tz_from_offset

USD_VALUE = {"USD": Decimal("1"), "EUR": Decimal("1.08"), "GBP": Decimal("1.27"),
             "CHF": Decimal("1.12"), "SEK": Decimal("0.095"), "JPY": Decimal("0.0067")}
CURRENCIES = sorted(USD_VALUE)
OFFSETS = ["-08:00", "-05:00", "-03:00", "+00:00", "+01:00", "+02:00", "+03:00", "+05:30", "+09:00"]
PRODUCTS = ["Consulting hours", "Cloud storage plan", "Maintenance visit", "Freight handling",
            "Software license", "Training seat", "Spare parts kit", "Data processing batch",
            "Support retainer", "Packaging units", "Calibration service", "Translation pages",
            "Hosting plan", "Design sprint", "Inspection report"]
EXPENSE_ACCOUNTS = ["6100", "6200", "6300", "6400", "6500"]
CHART_OF_ACCOUNTS = {"1000": "Cash", "1200": "Accounts receivable", "2900": "Unapplied cash",
                     "4000": "Revenue", "6100": "Rent", "6200": "Utilities", "6300": "Software",
                     "6400": "Travel", "6500": "Office supplies"}
DUPLICATE_WINDOW_DAYS = 7
ROUNDING = "ROUND_HALF_EVEN"
FX_MARGIN_DAYS = 12


class GeneratorError(RuntimeError):
    pass


class Builder:
    """Holds the hidden history plus metadata needed for error injection."""

    def __init__(self, seed: int, cfg: dict, rng_key: str):
        self.seed = seed
        self.cfg = cfg
        self.rng = random.Random(rng_key)
        self._ids: set[str] = set()
        self._seq = 0
        self.invoices: list[dict] = []
        self.payments: list[dict] = []
        self.ledger: list[dict] = []
        self.inv_meta: dict[str, dict] = {}
        self.pay_meta: dict[str, dict] = {}
        self.jes: dict[str, dict] = {}
        self.slots: dict[str, list[dict]] = {"fx_partial_boundary": [], "duplicate_cross_period": []}
        self.distractor_log: list[dict] = []

    # ------------------------------------------------------------------ ids / basics
    def new_id(self, prefix: str, n: int = 8) -> str:
        while True:
            s = f"{prefix}-{self.rng.getrandbits(4 * n):0{n}X}"
            if s not in self._ids:
                self._ids.add(s)
                return s

    def seq(self) -> int:
        self._seq += 1
        return self._seq

    def conv(self, amount_minor: int, cur: str, d: date) -> int:
        if cur == self.base:
            return abs(amount_minor)
        return convert_minor(abs(amount_minor), CURRENCY_EXPONENTS[cur], self.rate(cur, d),
                             CURRENCY_EXPONENTS[self.base], ROUNDING)

    def rate(self, cur: str, d: date) -> str:
        return self.fx[(d.isoformat(), cur)]

    def local_date(self, instant: datetime) -> date:
        return instant.astimezone(self.tz).date()

    def usd_cents_to_minor(self, cents: int, cur: str) -> int:
        v = Decimal(cents) / Decimal(100) / USD_VALUE[cur] * (Decimal(10) ** CURRENCY_EXPONENTS[cur])
        return max(1, int(v.quantize(Decimal(1))))

    def period_instant(self, day_lo: int = 0, day_hi: int | None = None) -> datetime:
        day_hi = self.ndays - 1 if day_hi is None else day_hi
        d = self.period_start + timedelta(days=self.rng.randint(day_lo, max(day_lo, day_hi)))
        return datetime(d.year, d.month, d.day, self.rng.randint(0, 23), self.rng.randint(0, 59),
                        tzinfo=self.tz)

    @property
    def start_instant(self) -> datetime:
        d = self.period_start
        return datetime(d.year, d.month, d.day, tzinfo=self.tz)

    @property
    def end_instant(self) -> datetime:
        """First instant *after* the period (exclusive bound)."""
        d = self.period_end + timedelta(days=1)
        return datetime(d.year, d.month, d.day, tzinfo=self.tz)

    # ------------------------------------------------------------------ company
    def build_company(self) -> None:
        r = self.rng
        self.episode_id = self.new_id("EP", 10)
        self.company_id = self.new_id("CO", 6)
        self.base = r.choice(CURRENCIES)
        self.foreign = sorted(r.sample([c for c in CURRENCIES if c != self.base], r.randint(1, 3)))
        self.tz_offset = r.choice(OFFSETS)
        self.tz = tz_from_offset(self.tz_offset)
        year, month = r.randint(2023, 2026), r.randint(1, 12)
        self.period_start = date(year, month, 1)
        nxt = date(year + (month == 12), month % 12 + 1, 1)
        self.period_end = nxt - timedelta(days=1)
        self.ndays = (self.period_end - self.period_start).days + 1
        # FX: one rate per calendar day per foreign currency, random walk that never stands still
        self.fx: dict[tuple[str, str], str] = {}
        first = self.period_start - timedelta(days=FX_MARGIN_DAYS * 3)
        days = (self.period_end - first).days + FX_MARGIN_DAYS + 1
        for cur in self.foreign:
            rate = USD_VALUE[cur] / USD_VALUE[self.base] * (1 + Decimal(r.randint(-300, 300)) / 10000)
            prev = None
            for i in range(days):
                step = Decimal(r.randint(20, 90)) / 10000 * r.choice((-1, 1))
                rate = rate * (1 + step)
                q = Decimal("0.000001") if rate >= 1 else Decimal("0.00000001")
                s = format(rate.quantize(q), "f")
                if s == prev:
                    rate = rate + q
                    s = format(rate.quantize(q), "f")
                prev = s
                self.fx[((first + timedelta(days=i)).isoformat(), cur)] = s
        self.fx_first = self.period_start - timedelta(days=FX_MARGIN_DAYS)
        self.fx_last = self.period_end + timedelta(days=FX_MARGIN_DAYS)
        # entities
        n_c = r.randint(*self.cfg["n_customers"])
        self.customers = [{"id": self.new_id("C", 6), "tz": r.choice(OFFSETS)} for _ in range(n_c)]
        if all(c["tz"] == self.tz_offset for c in self.customers):
            self.customers[0]["tz"] = next(o for o in OFFSETS if o != self.tz_offset)
        self.vendors = [self.new_id("V", 6) for _ in range(r.randint(3, 6))]
        names = r.sample(PRODUCTS, r.randint(6, len(PRODUCTS)))
        self.products = [{"sku": self.new_id("SKU", 5), "name": n, "usd_cents": r.randint(900, 180000)}
                         for n in names]

    # ------------------------------------------------------------------ documents
    def entity_tz(self, entity_id: str):
        for c in self.customers:
            if c["id"] == entity_id:
                return tz_from_offset(c["tz"])
        raise GeneratorError("unknown entity")

    def make_items(self, cur: str, k: int | None = None, min_total_usd_cents: int = 0) -> list[dict]:
        r = self.rng
        items = []
        for p in r.sample(self.products, k or r.randint(1, 4)):
            unit = self.usd_cents_to_minor(p["usd_cents"] * r.randint(80, 125) // 100, cur)
            qty = r.randint(1, 20)
            items.append({"sku": p["sku"], "description": p["name"], "quantity": qty,
                          "unit_price_minor": unit, "amount_minor": unit * qty})
        if min_total_usd_cents:
            floor = self.usd_cents_to_minor(min_total_usd_cents, cur)
            while sum(i["amount_minor"] for i in items) < floor:
                items[0]["quantity"] += 5
                items[0]["amount_minor"] = items[0]["quantity"] * items[0]["unit_price_minor"]
        return items

    def add_invoice(self, entity_id: str, cur: str, instant: datetime, items: list[dict],
                    kind: str, tags: tuple[str, ...] = ()) -> dict:
        etz = self.entity_tz(entity_id)
        net = self.rng.choice((14, 30, 45))
        inv = {
            "doc_id": self.new_id("D", 10),
            "invoice_id": self.new_id("INV", 8),
            "entity_id": entity_id,
            "issued_at": fmt_ts(instant, etz),
            "due_at": fmt_ts(instant + timedelta(days=net), etz),
            "currency": cur,
            "total_minor": sum(i["amount_minor"] for i in items),
            "line_items": copy.deepcopy(items),
        }
        self.invoices.append(inv)
        self.inv_meta[inv["invoice_id"]] = {"instant": instant, "kind": kind, "tags": set(tags),
                                            "seq": self.seq(), "net": net}
        return inv

    def add_payment(self, entity_id: str, cur: str, instant: datetime, refs: list[tuple[str, int]],
                    tags: tuple[str, ...] = ()) -> dict:
        if not (self.start_instant <= instant < self.end_instant):
            raise GeneratorError("payment outside period")
        pay = {
            "doc_id": self.new_id("D", 10),
            "payment_id": self.new_id("PAY", 8),
            "entity_id": entity_id,
            "timestamp": fmt_ts(instant, self.entity_tz(entity_id)),
            "currency": cur,
            "amount_minor": sum(a for _, a in refs),
            "invoice_refs": [{"invoice_id": i, "amount_minor": a} for i, a in refs],
        }
        self.payments.append(pay)
        self.pay_meta[pay["payment_id"]] = {"instant": instant, "tags": set(tags), "seq": self.seq()}
        return pay

    def inv(self, invoice_id: str) -> dict:
        return next(i for i in self.invoices if i["invoice_id"] == invoice_id)

    def pay(self, payment_id: str) -> dict:
        return next(p for p in self.payments if p["payment_id"] == payment_id)

    # ------------------------------------------------------------------ ledger
    def post_je(self, instant: datetime, dr: str, cr: str, amount_base: int, entity_id,
                invoice_id, payment_id, doc: tuple | None, portion: int | None, cur: str | None,
                rate_date: date | None) -> dict:
        if amount_base <= 0:
            raise GeneratorError("non-positive posting")
        je_id = f"JE{self.seq():06d}"
        ts = fmt_ts(instant, self.tz)
        rows = []
        for account, debit, credit in ((dr, amount_base, 0), (cr, 0, amount_base)):
            row = {"row_id": self.new_id("L", 8), "timestamp": ts, "account": account,
                   "debit_minor": debit, "credit_minor": credit, "currency": self.base,
                   "entity_id": entity_id, "invoice_id": invoice_id, "payment_id": payment_id}
            row["_instant"] = instant
            row["_seq"] = self.seq()
            row["_je"] = je_id
            rows.append(row)
            self.ledger.append(row)
        je = {"id": je_id, "rows": [rows[0]["row_id"], rows[1]["row_id"]], "doc": doc,
              "portion": portion, "cur": cur, "rate_date": rate_date}
        self.jes[je_id] = je
        return je

    def je_rows(self, je: dict) -> list[dict]:
        ids = set(je["rows"])
        return [r for r in self.ledger if r["row_id"] in ids]

    def set_je_amount(self, je: dict, amount: int) -> None:
        if amount <= 0:
            raise GeneratorError("non-positive posting")
        for r in self.je_rows(je):
            if r["debit_minor"]:
                r["debit_minor"] = amount
            else:
                r["credit_minor"] = amount

    def doc_jes(self, kind: str, doc_id: str) -> list[dict]:
        return [je for je in self.jes.values() if je["doc"] == (kind, doc_id)]

    def book_invoice(self, inv: dict, instant: datetime | None = None) -> dict:
        m = self.inv_meta[inv["invoice_id"]]
        d = self.local_date(m["instant"])
        amt = self.conv(inv["total_minor"], inv["currency"], d)
        return self.post_je(instant or m["instant"], "1200", "4000", amt, inv["entity_id"],
                            inv["invoice_id"], None, ("inv", inv["invoice_id"]), inv["total_minor"],
                            inv["currency"], d)

    def book_payment(self, pay: dict) -> None:
        m = self.pay_meta[pay["payment_id"]]
        d = self.local_date(m["instant"])
        cur = pay["currency"]
        for ref in pay["invoice_refs"]:
            a = ref["amount_minor"]
            dr, cr = ("1000", "1200") if a > 0 else ("1200", "1000")
            self.post_je(m["instant"], dr, cr, self.conv(a, cur, d), pay["entity_id"], ref["invoice_id"],
                         pay["payment_id"], ("pay", pay["payment_id"]), abs(a), cur, d)
        rem = pay["amount_minor"] - sum(x["amount_minor"] for x in pay["invoice_refs"])
        if rem:
            self.post_je(m["instant"], "1000", "2900", self.conv(rem, cur, d), pay["entity_id"], None,
                         pay["payment_id"], ("pay", pay["payment_id"]), abs(rem), cur, d)

    # ------------------------------------------------------------------ clean history
    def build_clean(self, plan: dict) -> None:
        r = self.rng
        dist = self.cfg["distractors"]
        cust = self.customers
        pick_cur = lambda: (r.choice(self.foreign) if r.randint(1, 100) <= self.cfg["foreign_share_pct"]
                            else self.base)

        def pay_after(inv: dict, frac: tuple[int, int] = (2, 60 * 24 * 20), tags=(), refs=None):
            m = self.inv_meta[inv["invoice_id"]]
            lo = max(m["instant"], self.start_instant) + timedelta(minutes=frac[0])
            hi = min(m["instant"] + timedelta(minutes=frac[1]), self.end_instant - timedelta(minutes=1))
            if hi <= lo:
                return None
            t = lo + timedelta(minutes=r.randint(0, int((hi - lo).total_seconds() // 60)))
            return self.add_payment(inv["entity_id"], inv["currency"], t,
                                    refs or [(inv["invoice_id"], inv["total_minor"])], tags)

        # --- regular in-period invoices
        n_regular = plan["n_regular"]
        for k in range(n_regular):
            cur = self.foreign[k % len(self.foreign)] if k < plan["forced_foreign"] else pick_cur()
            c = r.choice(cust)
            self.add_invoice(c["id"], cur, self.period_instant(), self.make_items(cur), "in", ("regular",))
        regular = [i for i in self.invoices]
        # payment plans for regular invoices: forced full payments first, then 75% full
        order = sorted(regular, key=lambda i: self.inv_meta[i["invoice_id"]]["instant"])
        r.shuffle(order)
        n_full_needed = plan["forced_full"]
        for inv in order:
            full = n_full_needed > 0 or r.randint(1, 100) <= 75
            if full and pay_after(inv, tags=("regular_full",)) is not None:
                n_full_needed -= 1
        if n_full_needed > 0:
            raise GeneratorError("not enough payable regular invoices")

        early = lambda: self.period_instant(0, self.ndays - 7)
        # --- split payments (legit): one invoice, two payments
        for _ in range(dist["split_payments"]):
            c, cur = r.choice(cust), pick_cur()
            inv = self.add_invoice(c["id"], cur, early(), self.make_items(cur), "in", ("split",))
            a1 = inv["total_minor"] * r.randint(30, 70) // 100
            p1 = pay_after(inv, (60, 60 * 24 * 3), ("split",), [(inv["invoice_id"], a1)])
            p2 = pay_after(inv, (60 * 24 * 3 + 60, 60 * 24 * 6), ("split",),
                           [(inv["invoice_id"], inv["total_minor"] - a1)])
            self.distractor_log.append({"kind": "split_payment", "invoice": inv["invoice_id"],
                                        "payments": [p["payment_id"] for p in (p1, p2) if p]})
        # --- legit partial payment (invoice stays partially open, payment fully allocated)
        for _ in range(dist["legit_partial"]):
            c, cur = r.choice(cust), pick_cur()
            inv = self.add_invoice(c["id"], cur, early(), self.make_items(cur), "in", ("legit_partial",))
            a1 = max(1, inv["total_minor"] * r.randint(20, 80) // 100)
            p = pay_after(inv, tags=("legit_partial",), refs=[(inv["invoice_id"], a1)])
            self.distractor_log.append({"kind": "legit_partial_payment", "invoice": inv["invoice_id"],
                                        "payment": p["payment_id"] if p else None})
        # --- one payment covering several invoices of the same entity
        for _ in range(dist["multi_invoice_payments"]):
            c, cur = r.choice(cust), pick_cur()
            a = self.add_invoice(c["id"], cur, early(), self.make_items(cur), "in", ("multi",))
            b = self.add_invoice(c["id"], cur, early(), self.make_items(cur), "in", ("multi",))
            later = max((a, b), key=lambda i: self.inv_meta[i["invoice_id"]]["instant"])
            p = pay_after(later, (60, 60 * 24 * 5), ("multi",),
                          [(a["invoice_id"], a["total_minor"]), (b["invoice_id"], b["total_minor"])])
            self.distractor_log.append({"kind": "multi_invoice_payment",
                                        "invoices": [a["invoice_id"], b["invoice_id"]],
                                        "payment": p["payment_id"] if p else None})
        # --- refund: paid twice, second payment refunded (legit repeated amount + negative payment)
        for _ in range(dist["refunds"]):
            c, cur = r.choice(cust), pick_cur()
            inv = self.add_invoice(c["id"], cur, early(), self.make_items(cur), "in", ("refund",))
            t = inv["total_minor"]
            p1 = pay_after(inv, (60, 60 * 24 * 2), ("refund",))
            p2 = pay_after(inv, (60 * 24 * 2 + 60, 60 * 24 * 4), ("refund",))
            rf = pay_after(inv, (60 * 24 * 4 + 60, 60 * 24 * 6), ("refund",), [(inv["invoice_id"], -t)])
            self.distractor_log.append({"kind": "refund", "invoice": inv["invoice_id"],
                                        "payments": [p["payment_id"] for p in (p1, p2, rf) if p]})
        # --- recurring invoices: same entity/amount, different service period label
        for _ in range(dist["recurring"]):
            c, cur = r.choice(cust), pick_cur()
            items = self.make_items(cur, 1)
            day = r.randint(0, self.ndays - 9)
            for wk in (1, 2):
                it = copy.deepcopy(items)
                it[0]["description"] = f"{it[0]['description']} - service week {day // 7 + wk}"
                t0 = self.period_instant(day + 7 * (wk - 1), day + 7 * (wk - 1))
                inv = self.add_invoice(c["id"], cur, t0, it, "in", ("recurring",))
                if wk == 1:
                    pay_after(inv, (60, 60 * 24 * 5), ("recurring",))
            self.distractor_log.append({"kind": "recurring_amount", "entity": c["id"]})
        # --- same items, same day, different customer (not a duplicate)
        for _ in range(dist["same_amount_other_entity"]):
            src = r.choice(regular)
            others = [c for c in cust if c["id"] != src["entity_id"]]
            if not others:
                continue
            m = self.inv_meta[src["invoice_id"]]
            t0 = m["instant"].replace(hour=r.randint(0, 23), minute=r.randint(0, 59))
            if not (self.start_instant <= t0 < self.end_instant):
                continue
            inv = self.add_invoice(r.choice(others)["id"], src["currency"], t0, src["line_items"], "in",
                                   ("same_amount",))
            self.distractor_log.append({"kind": "same_items_other_entity",
                                        "invoices": [src["invoice_id"], inv["invoice_id"]]})
        # --- timezone boundary: in period in manifest tz, but the written date is outside
        for k in range(dist["tz_boundary"]):
            mo = offset_minutes(self.tz_offset)
            lower = [c for c in cust if offset_minutes(c["tz"]) < mo]
            higher = [c for c in cust if offset_minutes(c["tz"]) > mo]
            choices = ([("start", c) for c in lower] + [("end", c) for c in higher])
            if not choices:
                continue
            side, c = r.choice(choices)
            diff = abs(mo - offset_minutes(c["tz"]))
            m = r.randint(0, diff - 1)
            t0 = (self.start_instant + timedelta(minutes=m) if side == "start"
                  else self.end_instant - timedelta(minutes=m + 1))
            cur = pick_cur()
            inv = self.add_invoice(c["id"], cur, t0, self.make_items(cur), "in", ("tz_boundary",))
            if side == "start":
                pay_after(inv, tags=("tz_boundary",))
            self.distractor_log.append({"kind": "tz_boundary", "invoice": inv["invoice_id"], "side": side})
        # --- reserved slot: FX payment at the period boundary (for interaction scenario)
        mo = offset_minutes(self.tz_offset)
        for _ in range(plan["slots_fx"]):
            cur = r.choice(self.foreign)
            c = r.choice(cust)
            items = self.make_items(cur, min_total_usd_cents=300000)
            if mo > 0:   # first local day, early hours -> UTC date is the previous day
                t_inv = self.start_instant - timedelta(days=r.randint(3, 15), hours=r.randint(0, 20))
                inv = self.add_invoice(c["id"], cur, t_inv, items, "prior", ("slot_fx",))
                t_pay = self.start_instant + timedelta(minutes=r.randint(0, mo - 1))
            else:        # last local day, late hours -> UTC date is the next day (or next day bug)
                t_inv = self.period_instant(0, self.ndays - 4)
                inv = self.add_invoice(c["id"], cur, t_inv, items, "in", ("slot_fx",))
                lo = 1440 + mo if mo < 0 else 20 * 60
                t_pay = self.end_instant - timedelta(minutes=1440 - r.randint(lo, 1439))
            p = self.add_payment(c["id"], cur, t_pay, [(inv["invoice_id"], inv["total_minor"])], ("slot_fx",))
            utc_d = t_pay.astimezone(tz_from_offset("+00:00")).date()
            loc_d = self.local_date(t_pay)
            wrong = utc_d if utc_d != loc_d else loc_d + timedelta(days=1)
            self.slots["fx_partial_boundary"].append({"invoice": inv["invoice_id"], "payment": p["payment_id"],
                                                      "wrong_date": wrong})
        # --- reserved slot: invoice at period end that will be re-issued after the period
        for _ in range(plan["slots_dup"]):
            c, cur = r.choice(cust), pick_cur()
            t_inv = self.period_instant(self.ndays - 2, self.ndays - 1)
            inv = self.add_invoice(c["id"], cur, t_inv, self.make_items(cur), "in", ("slot_dup",))
            self.slots["duplicate_cross_period"].append({"invoice": inv["invoice_id"]})
        # --- prior-period invoices (booked last period) paid in this period
        for _ in range(dist["prior_period_invoices"]):
            c, cur = r.choice(cust), pick_cur()
            t0 = self.start_instant - timedelta(days=r.randint(5, 25), minutes=r.randint(0, 1439))
            inv = self.add_invoice(c["id"], cur, t0, self.make_items(cur), "prior", ("prior",))
            pay_after(inv, (60, 60 * 24 * 40), ("prior",))
            self.distractor_log.append({"kind": "prior_period_invoice", "invoice": inv["invoice_id"]})
        # --- next-period invoices (issued after the period end, must NOT be booked)
        for _ in range(dist["next_period_invoices"] + plan["extra_next"]):
            c, cur = r.choice(cust), pick_cur()
            t0 = self.end_instant + timedelta(days=r.randint(0, 5), minutes=r.randint(1, 1439))
            self.add_invoice(c["id"], cur, t0, self.make_items(cur), "next", ("next",))
        self._fix_accidental_duplicates()
        # --- ledger postings for documents
        for inv in sorted(self.invoices, key=lambda i: self.inv_meta[i["invoice_id"]]["seq"]):
            if self.inv_meta[inv["invoice_id"]]["kind"] == "in":
                self.book_invoice(inv)
        for pay in sorted(self.payments, key=lambda p: self.pay_meta[p["payment_id"]]["seq"]):
            self.book_payment(pay)
        # --- operating expenses (noise) to reach the target ledger size
        target = min(r.randint(*self.cfg["ledger_rows"]), 2000 - 4 * 12 - 4)
        while len(self.ledger) + 2 <= max(target, 200 + 2):
            amt = self.usd_cents_to_minor(r.randint(1500, 400000), self.base)
            self.post_je(self.period_instant(), r.choice(EXPENSE_ACCOUNTS), "1000", amt, r.choice(self.vendors),
                         None, None, None, None, None, None)
        self.assert_clean()

    def dup_key(self, inv: dict):
        items = sorted((i["sku"], i["description"], i["quantity"], i["unit_price_minor"], i["amount_minor"])
                       for i in inv["line_items"])
        return (inv["entity_id"], inv["currency"], inv["total_minor"], tuple(items))

    def _fix_accidental_duplicates(self) -> None:
        seen: dict = {}
        for inv in self.invoices:
            k = self.dup_key(inv)
            for other in seen.get(k, []):
                dt = abs(self.inv_meta[inv["invoice_id"]]["instant"] - self.inv_meta[other]["instant"])
                if dt <= timedelta(days=DUPLICATE_WINDOW_DAYS):
                    raise GeneratorError("accidental duplicate")  # astronomically rare; caller retries
            seen.setdefault(k, []).append(inv["invoice_id"])

    # ------------------------------------------------------------------ assertions on the clean history
    def assert_clean(self) -> None:
        assert sum(r["debit_minor"] for r in self.ledger) == sum(r["credit_minor"] for r in self.ledger)
        for je in self.jes.values():
            rows = self.je_rows(je)
            assert len(rows) == 2 and rows[0]["debit_minor"] == rows[1]["credit_minor"] > 0
        allocated: dict[str, int] = {}
        for p in self.payments:
            assert p["amount_minor"] == sum(x["amount_minor"] for x in p["invoice_refs"]), "unallocated cash"
            for x in p["invoice_refs"]:
                assert self.inv(x["invoice_id"])["currency"] == p["currency"]
                allocated[x["invoice_id"]] = allocated.get(x["invoice_id"], 0) + x["amount_minor"]
        for inv in self.invoices:
            assert inv["total_minor"] == sum(i["amount_minor"] for i in inv["line_items"])
            assert 0 <= allocated.get(inv["invoice_id"], 0) <= inv["total_minor"], "over-allocation"
            booked = self.doc_jes("inv", inv["invoice_id"])
            in_period = self.inv_meta[inv["invoice_id"]]["kind"] == "in"
            assert len(booked) == (1 if in_period else 0)
        for row in self.ledger:
            assert self.start_instant <= row["_instant"] < self.end_instant
        for je in self.jes.values():
            if je["doc"] is not None:
                assert self.je_rows(je)[0]["debit_minor"] == self.conv(je["portion"], je["cur"], je["rate_date"])

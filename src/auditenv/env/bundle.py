"""Read-only view of the six PUBLIC episode files. This is the only data the environment holds."""
from __future__ import annotations

import csv
import io
import json

from ..generator.episode import PUBLIC_FILES


class PublicBundle:
    def __init__(self, files: dict[str, str]):
        extra = set(files) - set(PUBLIC_FILES)
        if extra or set(files) != set(PUBLIC_FILES):
            raise ValueError("bundle must contain exactly the public files")
        self.manifest = json.loads(files["manifest.json"])
        self.policy = json.loads(files["policy.json"])
        ledger = []
        for r in csv.DictReader(io.StringIO(files["ledger.csv"])):
            r = {k: (None if v == "" else v) for k, v in r.items()}
            r["debit_minor"] = int(r["debit_minor"])
            r["credit_minor"] = int(r["credit_minor"])
            ledger.append(r)
        fx = []
        for r in csv.DictReader(io.StringIO(files["fx.csv"])):
            fx.append({"fx_id": f"FX:{r['date']}:{r['source_currency']}:{r['target_currency']}", **r})
        self.tables = {
            "ledger": ledger,
            "invoices": [json.loads(x) for x in files["invoices.jsonl"].splitlines() if x],
            "payments": [json.loads(x) for x in files["payments.jsonl"].splitlines() if x],
            "fx": fx,
        }
        keys = {"ledger": "row_id", "invoices": "invoice_id", "payments": "payment_id", "fx": "fx_id"}
        self.index = {t: {r[k]: r for r in rows} for t, rows in self.tables.items() for k in [keys[t]]}
        self.docs = {r["doc_id"]: ("invoice", r) for r in self.tables["invoices"]}
        self.docs.update({r["doc_id"]: ("payment", r) for r in self.tables["payments"]})

    @property
    def episode_id(self) -> str:
        return self.manifest["episode_id"]

    def known_ids(self) -> set[str]:
        ids = set(self.docs)
        for t in self.index.values():
            ids |= set(t)
        return ids

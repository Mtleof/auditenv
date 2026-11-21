"""Episode container and deterministic serialization of public / private files."""
from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
from dataclasses import dataclass, field

PUBLIC_FILES = ("manifest.json", "ledger.csv", "invoices.jsonl", "payments.jsonl", "fx.csv", "policy.json")
PRIVATE_FILES = ("truth.json", "provenance.json")
LEDGER_COLUMNS = ["row_id", "timestamp", "account", "debit_minor", "credit_minor", "currency",
                  "entity_id", "invoice_id", "payment_id"]
FX_COLUMNS = ["date", "source_currency", "target_currency", "rate"]


def dump_json(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"


def dump_jsonl(rows) -> str:
    return "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows)


def dump_csv(rows, columns) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(columns)
    for r in rows:
        w.writerow(["" if r[c] is None else r[c] for c in columns])
    return buf.getvalue()


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclass
class Episode:
    manifest: dict
    policy: dict
    ledger: list
    invoices: list
    payments: list
    fx: list
    truth: dict
    provenance: dict
    clean_tables: dict = field(default_factory=dict)   # public tables before injection (validator input)

    @property
    def episode_id(self) -> str:
        return self.manifest["episode_id"]

    def tables(self) -> dict:
        return {"ledger": self.ledger, "invoices": self.invoices, "payments": self.payments, "fx": self.fx}

    def public_files(self) -> dict[str, str]:
        return render_public(self.manifest, self.policy, self.tables())

    def clean_public_files(self) -> dict[str, str]:
        t = self.clean_tables
        man = copy.deepcopy(self.manifest)
        man["counts"] = counts(t)
        return render_public(man, self.policy, t)

    def private_files(self) -> dict[str, str]:
        return {"truth.json": dump_json(self.truth), "provenance.json": dump_json(self.provenance)}


def counts(t: dict) -> dict:
    return {"ledger_rows": len(t["ledger"]), "invoices": len(t["invoices"]), "payments": len(t["payments"]),
            "documents": len(t["invoices"]) + len(t["payments"]), "fx_rates": len(t["fx"])}


def render_public(manifest: dict, policy: dict, t: dict) -> dict[str, str]:
    return {
        "manifest.json": dump_json(manifest),
        "ledger.csv": dump_csv(t["ledger"], LEDGER_COLUMNS),
        "invoices.jsonl": dump_jsonl(t["invoices"]),
        "payments.jsonl": dump_jsonl(t["payments"]),
        "fx.csv": dump_csv(t["fx"], FX_COLUMNS),
        "policy.json": dump_json(policy),
    }

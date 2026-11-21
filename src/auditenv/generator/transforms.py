"""Metamorphic transforms for paired checks. Both preserve the audit answer:
  * permute(ep, seed)   - reorder rows, documents, line items and allocations;
  * remap_ids(ep, seed) - consistently rename every identifier (truth is renamed too).
Neither changes amounts, dates, currencies or the relation between records."""
from __future__ import annotations

import copy
import random
import re

from .episode import Episode

_ID_RE = re.compile(r"^(EP|CO|C|V|INV|PAY|D|L|SKU)-([0-9A-F]+)$")


def _shuffle_tables(t: dict, r: random.Random) -> None:
    for key in ("ledger", "invoices", "payments", "fx"):
        r.shuffle(t[key])
    for inv in t["invoices"]:
        r.shuffle(inv["line_items"])
    for p in t["payments"]:
        r.shuffle(p["invoice_refs"])


def permute(ep: Episode, seed: int) -> Episode:
    r = random.Random(f"permute|{seed}")
    out = copy.deepcopy(ep)
    t = out.tables()
    _shuffle_tables(t, r)
    if out.clean_tables:
        _shuffle_tables(out.clean_tables, r)
    return out


def _collect(obj, acc: set) -> None:
    if isinstance(obj, dict):
        for v in obj.values():
            _collect(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            _collect(v, acc)
    elif isinstance(obj, str) and _ID_RE.match(obj):
        acc.add(obj)


def _apply(obj, mapping: dict):
    if isinstance(obj, dict):
        return {k: _apply(v, mapping) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_apply(v, mapping) for v in obj]
    if isinstance(obj, str):
        return mapping.get(obj, obj)
    return obj


def remap_ids(ep: Episode, seed: int) -> tuple[Episode, dict]:
    r = random.Random(f"remap|{seed}")
    ids: set[str] = set()
    for part in (ep.manifest, ep.tables(), ep.truth):
        _collect(part, ids)
    mapping, used = {}, set(ids)
    for old in sorted(ids):
        prefix, body = _ID_RE.match(old).groups()
        while True:
            new = f"{prefix}-{r.getrandbits(4 * len(body)):0{len(body)}X}"
            if new not in used:
                used.add(new)
                mapping[old] = new
                break
    out = Episode(_apply(ep.manifest, mapping), copy.deepcopy(ep.policy), _apply(ep.ledger, mapping),
                  _apply(ep.invoices, mapping), _apply(ep.payments, mapping), copy.deepcopy(ep.fx),
                  _apply(ep.truth, mapping), _apply(ep.provenance, mapping), _apply(ep.clean_tables, mapping))
    for iss in out.truth["issues"]:
        iss["affected_ids"] = sorted(iss["affected_ids"])
        iss["evidence_sets"] = [sorted(s) for s in iss["evidence_sets"]]
    return out, mapping

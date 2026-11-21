"""Generator configuration: defaults, deep-merge, validation and canonical hashing."""
from __future__ import annotations

import copy
import hashlib
import json

from .. import FINDING_TYPES

CONFIG_VERSION = "1.0.0"
INTERACTIONS = ("fx_partial_boundary", "duplicate_cross_period")
MAX_ERRORS = 12

DEFAULT_CONFIG: dict = {
    "config_version": CONFIG_VERSION,
    "n_customers": [6, 18],
    "n_invoices": [24, 60],
    "ledger_rows": [200, 700],
    "foreign_share_pct": 35,
    "errors": {t: {"enabled": True, "count": 1} for t in FINDING_TYPES},
    "interactions": {k: 0 for k in INTERACTIONS},
    "distractors": {
        "split_payments": 2,
        "multi_invoice_payments": 1,
        "refunds": 1,
        "recurring": 1,
        "same_amount_other_entity": 1,
        "tz_boundary": 2,
        "prior_period_invoices": 3,
        "next_period_invoices": 2,
        "legit_partial": 2,
    },
    "row_order": "chronological",
}


class ConfigError(ValueError):
    pass


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if k not in out:
            raise ConfigError(f"unknown config key: {k}")
        if isinstance(v, dict) and isinstance(out[k], dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def clean_config(**over) -> dict:
    """Config with every error type disabled (a clean company)."""
    cfg = {"errors": {t: {"enabled": False, "count": 0} for t in FINDING_TYPES}}
    cfg.update(over)
    return cfg


def resolve_config(config: dict | None) -> dict:
    cfg = _merge(DEFAULT_CONFIG, config or {})
    if cfg["config_version"] != CONFIG_VERSION:
        raise ConfigError("unsupported config_version")
    for key in ("n_customers", "n_invoices", "ledger_rows"):
        lo, hi = cfg[key]
        if not (isinstance(lo, int) and isinstance(hi, int) and 0 < lo <= hi):
            raise ConfigError(f"bad range {key}")
    if cfg["n_invoices"][0] < 14 or cfg["n_invoices"][1] > 100:
        raise ConfigError("n_invoices must be within [14, 100]")
    if cfg["ledger_rows"][0] < 200 or cfg["ledger_rows"][1] > 2000:
        raise ConfigError("ledger_rows must be within [200, 2000]")
    if not 0 <= cfg["foreign_share_pct"] <= 100:
        raise ConfigError("foreign_share_pct")
    total = 0
    for t, spec in cfg["errors"].items():
        if t not in FINDING_TYPES or set(spec) != {"enabled", "count"}:
            raise ConfigError("bad errors section")
        if not isinstance(spec["count"], int) or spec["count"] < 0:
            raise ConfigError("bad error count")
        total += spec["count"] if spec["enabled"] else 0
    for k, n in cfg["interactions"].items():
        if not isinstance(n, int) or n < 0 or n > 2:
            raise ConfigError("bad interaction count")
        total += 2 * n
    for k, n in cfg["distractors"].items():
        if not isinstance(n, int) or n < 0 or n > 6:
            raise ConfigError("bad distractor count")
    if total > MAX_ERRORS:
        raise ConfigError("too many errors (max 12)")
    if cfg["row_order"] not in ("chronological", "shuffled"):
        raise ConfigError("row_order")
    return cfg


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(canonical_json(cfg).encode()).hexdigest()[:16]

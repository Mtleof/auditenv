"""Synthetic episode generator.

Order: (1) clean hidden history with accounting identities asserted, (2) documents and ledger
postings, (3) error injection with provenance, (4) interacting scenarios, (5) clean companies and
plausible distractors, (6) self-check with the independent validator."""
from __future__ import annotations

import copy

from .. import ENV_VERSION, FINDING_TYPES, GENERATOR_VERSION, SCHEMA_VERSION
from .config import (CONFIG_VERSION, DEFAULT_CONFIG, INTERACTIONS, ConfigError, canonical_json, clean_config,
                     config_hash, resolve_config)
from .episode import PRIVATE_FILES, PUBLIC_FILES, Episode, counts, sha256
from .history import CHART_OF_ACCOUNTS, DUPLICATE_WINDOW_DAYS, ROUNDING, Builder, GeneratorError
from .inject import Injector

MAX_ATTEMPTS = 8


def _plan(b: Builder, cfg: dict, counts_: dict) -> dict:
    d = cfg["distractors"]
    specials = (d["split_payments"] + d["legit_partial"] + 2 * d["multi_invoice_payments"] + d["refunds"]
                + 2 * d["recurring"] + d["same_amount_other_entity"] + d["tz_boundary"]
                + cfg["interactions"]["duplicate_cross_period"]
                + cfg["interactions"]["fx_partial_boundary"])
    n_inv = b.rng.randint(*cfg["n_invoices"])
    c_d, c_a, c_f = (counts_["duplicate_invoice"], counts_["missing_or_partial_allocation"],
                     counts_["wrong_fx_date_or_rate"])
    return {
        "n_regular": max(n_inv - specials, c_d + c_a + c_f + 8),
        "forced_foreign": c_f + 2,
        "forced_full": c_a + c_f + 2,
        "slots_fx": cfg["interactions"]["fx_partial_boundary"],
        "slots_dup": cfg["interactions"]["duplicate_cross_period"],
        "extra_next": counts_["period_misattribution"] + 1,
    }


def _public_tables(b: Builder) -> dict:
    ledger = sorted(b.ledger, key=lambda r: (r["_instant"], r["_seq"]))
    ledger = [{k: v for k, v in r.items() if not k.startswith("_")} for r in ledger]
    if b.cfg["row_order"] == "shuffled":
        b.rng.shuffle(ledger)
    invoices = sorted(b.invoices, key=lambda i: (b.inv_meta[i["invoice_id"]]["instant"],
                                                 b.inv_meta[i["invoice_id"]]["seq"]))
    payments = sorted(b.payments, key=lambda p: (b.pay_meta[p["payment_id"]]["instant"],
                                                 b.pay_meta[p["payment_id"]]["seq"]))
    fx = []
    day = b.fx_first
    while day <= b.fx_last:
        for cur in b.foreign:
            fx.append({"date": day.isoformat(), "source_currency": cur, "target_currency": b.base,
                       "rate": b.rate(cur, day)})
        day = day.fromordinal(day.toordinal() + 1)
    return copy.deepcopy({"ledger": ledger, "invoices": invoices, "payments": payments, "fx": fx})


def _policy(b: Builder) -> dict:
    return {
        "policy_version": "1.0.0",
        "fx": {
            "rate_source": "fx.csv (one rate per calendar day, source_currency -> target_currency)",
            "rate_date": "calendar date of the document timestamp (invoice issued_at / payment timestamp) "
                         "converted to the manifest timezone",
            "conversion": "base_minor = round(amount_minor / 10**exp(source) * rate * 10**exp(base))",
            "record_key": "fx records are addressed as FX:<date>:<source_currency>:<target_currency>",
        },
        "period": {
            "start": b.period_start.isoformat(), "end": b.period_end.isoformat(), "inclusive": True,
            "timezone": b.tz_offset,
            "basis": "document timestamp converted to the manifest timezone",
            "ledger_scope": "the ledger holds only postings for documents dated inside the period; "
                            "invoices issued before the period were booked in their own period",
        },
        "rounding": {"mode": ROUNDING, "unit": "minor unit of the target currency", "applied": "once per ledger row"},
        "allocation": "every payment must be fully allocated: amount_minor == sum(invoice_refs[].amount_minor); "
                      "refunds are negative payments with negative allocations",
        "duplicates": f"an invoice duplicates another if entity_id, currency, total_minor and the multiset of "
                      f"line items are identical and issue instants are within {DUPLICATE_WINDOW_DAYS} days",
        "chart_of_accounts": CHART_OF_ACCOUNTS,
        "postings": {
            "invoice": "Dr 1200 / Cr 4000 (invoice_id set, payment_id null)",
            "payment": "per allocation Dr 1000 / Cr 1200 (invoice_id = allocated invoice); unallocated remainder "
                       "Dr 1000 / Cr 2900 (invoice_id null)",
            "refund": "Dr 1200 / Cr 1000",
            "currency": "all ledger rows are in the base currency",
        },
        "findings": {
            "duplicate_invoice": {"affected_ids": "both invoice_ids", "amount": "invoice total_minor, invoice currency",
                                  "evidence": "doc_ids of both invoices"},
            "missing_or_partial_allocation": {
                "affected_ids": "payment_id plus every invoice_id the payment references",
                "amount": "amount_minor - sum(allocations), payment currency", "evidence": "doc_id of the payment"},
            "wrong_fx_date_or_rate": {
                "affected_ids": "the invoice_id or payment_id whose ledger rows use a wrong rate",
                "amount": "sum over the document's debit rows of |booked - correct|, base currency",
                "evidence": "doc_id, fx key of the correct rate, and all debit rows (or all credit rows) of the document"},
            "period_misattribution": {
                "affected_ids": "invoice_id booked in the wrong period",
                "amount": "booked debit amount, base currency",
                "evidence": "doc_id and one ledger row of that booking"},
            "amount_tolerance_minor": 1,
        },
    }


def generate(seed: int, config: dict | None = None) -> Episode:
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ConfigError("seed must be a non-negative int")
    cfg = resolve_config(config)
    chash = config_hash(cfg)
    counts_ = {t: (s["count"] if s["enabled"] else 0) for t, s in cfg["errors"].items()}
    last_err = None
    for attempt in range(MAX_ATTEMPTS):
        rng_key = f"auditenv|{GENERATOR_VERSION}|{seed}|{chash}|{attempt}"
        b = Builder(seed, cfg, rng_key)
        try:
            b.build_company()
            plan = _plan(b, cfg, counts_)
            b.build_clean(plan)
            clean_tables = _public_tables(b)
            inj = Injector(b)
            inj.run(counts_)
            tables = _public_tables(b)
            c = counts(tables)
            if not (20 <= c["documents"] <= 200 and 200 <= c["ledger_rows"] <= 2000):
                raise GeneratorError("size out of range")
        except GeneratorError as e:
            last_err = e
            continue
        break
    else:
        raise GeneratorError(f"could not generate episode: {last_err}")
    from ..tools.money import CURRENCY_EXPONENTS
    curs = sorted({b.base, *b.foreign})
    manifest = {
        "schema_version": SCHEMA_VERSION, "env_version": ENV_VERSION, "generator_version": GENERATOR_VERSION,
        "config_version": CONFIG_VERSION, "episode_id": b.episode_id, "company_id": b.company_id,
        "base_currency": b.base, "currency_exponents": {k: CURRENCY_EXPONENTS[k] for k in curs},
        "period_start": b.period_start.isoformat(), "period_end": b.period_end.isoformat(),
        "timezone": b.tz_offset, "counts": c,
        "rounding": {"mode": ROUNDING, "unit": "target currency minor unit", "applied": "once per ledger row"},
        "csv_null": "empty field",
        "files": list(PUBLIC_FILES),
    }
    policy = _policy(b)
    truth = {"episode_id": b.episode_id, "issues": inj.issues}
    ep = Episode(manifest, policy, tables["ledger"], tables["invoices"], tables["payments"], tables["fx"],
                 truth, {}, clean_tables)
    ep.provenance = {
        "generator_version": GENERATOR_VERSION, "config_version": CONFIG_VERSION, "config_hash": chash,
        "seed": seed, "attempt": attempt, "config": cfg,
        "clean_public_sha256": {k: sha256(v) for k, v in ep.clean_public_files().items()},
        "distractors": b.distractor_log, "mutations": inj.mutations,
    }
    return ep


__all__ = ["generate", "Episode", "DEFAULT_CONFIG", "INTERACTIONS", "FINDING_TYPES", "ConfigError",
           "GeneratorError", "clean_config", "resolve_config", "config_hash", "canonical_json",
           "PUBLIC_FILES", "PRIVATE_FILES"]

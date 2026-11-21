import copy
import json
import random
import re
from collections import Counter

import pytest

from auditenv import FINDING_TYPES
from auditenv.generator import ConfigError, clean_config, generate
from auditenv.validator import View, detect, identity_errors, validate_episode

TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$")


def random_config(i):
    r = random.Random(f"cfg{i}")
    errs = {t: {"enabled": r.random() < 0.8, "count": r.randint(0, 2)} for t in FINDING_TYPES}
    inter = {"fx_partial_boundary": r.randint(0, 1), "duplicate_cross_period": r.randint(0, 1)}
    lo = r.choice([200, 300, 900])
    return {"errors": errs, "interactions": inter, "ledger_rows": [lo, lo + r.choice([50, 1000])],
            "n_invoices": [14, r.choice([30, 60, 90])], "row_order": r.choice(["chronological", "shuffled"]),
            "foreign_share_pct": r.choice([0, 35, 80])}


SEEDS = list(range(100, 124))


@pytest.mark.parametrize("seed", SEEDS)
def test_validator_on_many_seeds(seed):
    """(a) identities before injection, (b) discrepancy after, (c) sufficiency of every evidence set."""
    cfg = random_config(seed)
    ep = generate(seed, cfg)
    assert validate_episode(ep) == []
    c = ep.manifest["counts"]
    assert 200 <= c["ledger_rows"] <= 2000 and 20 <= c["documents"] <= 200
    n = len(ep.truth["issues"])
    exp = sum(s["count"] for s in cfg["errors"].values() if s["enabled"]) + 2 * sum(cfg["interactions"].values())
    assert n == exp and 0 <= n <= 12
    # clean snapshot really is clean, injected version really is not (unless no errors requested)
    clean = View(ep.clean_public_files())
    assert identity_errors(clean) == [] and detect(clean) == ([], [])
    if n:
        assert ep.public_files() != ep.clean_public_files()


def test_big_episode_and_clean_company():
    ep = generate(7, {"ledger_rows": [1900, 2000], "n_invoices": [95, 100]})
    assert validate_episode(ep) == [] and 1900 <= len(ep.ledger) <= 2000
    cl = generate(8, clean_config())
    assert cl.truth["issues"] == [] and validate_episode(cl) == []
    assert cl.public_files() == cl.clean_public_files()


def test_distractors_present_in_clean_company():
    ep = generate(9, clean_config(distractors={"tz_boundary": 3}))
    kinds = Counter(d["kind"] for d in ep.provenance["distractors"])
    for k in ("split_payment", "refund", "recurring_amount", "multi_invoice_payment", "legit_partial_payment",
              "prior_period_invoice", "same_items_other_entity"):
        assert kinds[k] >= 1, k
    # a refund is a negative payment with negative allocation; repeated equal payments exist
    assert any(p["amount_minor"] < 0 for p in ep.payments)
    amounts = Counter((p["entity_id"], p["amount_minor"]) for p in ep.payments if p["amount_minor"] > 0)
    assert max(amounts.values()) >= 2
    # timezone-boundary invoices: written date outside the period, yet inside in manifest tz
    v = View(ep.public_files())
    tricky = [i for i in ep.invoices if v.in_period(i["issued_at"]) and not
              (ep.manifest["period_start"] <= i["issued_at"][:10] <= ep.manifest["period_end"])]
    assert tricky or ep.manifest["timezone"] in ("-08:00", "+09:00")
    # documents issued outside the period exist but are not booked
    outside = [i for i in ep.invoices if not v.in_period(i["issued_at"])]
    assert outside and not any(r["invoice_id"] in {i["invoice_id"] for i in outside} and r["payment_id"] is None
                               for r in ep.ledger)


@pytest.mark.parametrize("typ", FINDING_TYPES)
def test_each_type_independently_configurable(typ):
    cfg = clean_config()
    cfg["errors"] = copy.deepcopy(cfg["errors"])
    cfg["errors"][typ] = {"enabled": True, "count": 3}
    ep = generate(31, cfg)
    assert [i["type"] for i in ep.truth["issues"]] == [typ] * 3
    assert validate_episode(ep) == []
    cfg["errors"][typ]["enabled"] = False
    assert generate(31, cfg).truth["issues"] == []


@pytest.mark.parametrize("scenario,types", [
    ("fx_partial_boundary", ["missing_or_partial_allocation", "wrong_fx_date_or_rate"]),
    ("duplicate_cross_period", ["duplicate_invoice", "period_misattribution"]),
])
def test_interaction_scenarios(scenario, types):
    for seed in range(40, 52):
        cfg = clean_config(interactions={scenario: 1})
        ep = generate(seed, cfg)
        iss = ep.truth["issues"]
        assert sorted(i["type"] for i in iss) == sorted(types)
        assert {i["scenario"] for i in iss} == {scenario}
        # the two issues share a document
        a, b = iss
        assert set(a["affected_ids"]) & set(b["affected_ids"])
        assert validate_episode(ep) == []
        if scenario == "fx_partial_boundary":
            pay = next(p for p in ep.payments if p["payment_id"] in a["affected_ids"])
            assert pay["currency"] != ep.manifest["base_currency"]
            v = View(ep.public_files())
            d = v.local_date(pay["timestamp"]).isoformat()
            assert d in (ep.manifest["period_start"], ep.manifest["period_end"])   # at the boundary


def test_formats_money_fx_time():
    ep = generate(13)
    for r in ep.ledger:
        assert isinstance(r["debit_minor"], int) and isinstance(r["credit_minor"], int)
        assert TS_RE.match(r["timestamp"])
    for i in ep.invoices:
        assert TS_RE.match(i["issued_at"]) and isinstance(i["total_minor"], int)
    for p in ep.payments:
        assert TS_RE.match(p["timestamp"]) and isinstance(p["amount_minor"], int)
    for f in ep.fx:
        assert re.match(r"^\d+\.\d+$", f["rate"])
    files = ep.public_files()
    assert "e-" not in files["fx.csv"].lower() and "nan" not in files["fx.csv"].lower()
    # explicit nulls: missing refs are empty in CSV and never guessed
    assert ",," in files["ledger.csv"] or files["ledger.csv"].rstrip().endswith(",")
    for line in files["payments.jsonl"].splitlines():
        json.loads(line)


def test_provenance_records_every_mutation():
    ep = generate(14, {"interactions": {"fx_partial_boundary": 1, "duplicate_cross_period": 1}})
    ids = {m["issue_id"] for m in ep.provenance["mutations"]}
    assert ids == {i["issue_id"] for i in ep.truth["issues"]}
    assert ep.provenance["seed"] == 14 and len(ep.provenance["clean_public_sha256"]) == 6


def test_varies_across_seeds():
    eps = [generate(s) for s in range(200, 212)]
    assert len({e.manifest["base_currency"] for e in eps}) >= 3
    assert len({e.manifest["timezone"] for e in eps}) >= 4
    assert len({e.manifest["period_start"] for e in eps}) >= 8
    assert len({len(e.ledger) for e in eps}) == len(eps)


def test_validator_rejects_tampering():
    ep = generate(15)
    bad = copy.deepcopy(ep); bad.truth["issues"].pop()
    assert any("detect mismatch" in e for e in validate_episode(bad))
    bad = copy.deepcopy(ep); bad.truth["issues"][0]["expected_amount_minor"] += 5
    assert validate_episode(bad)
    bad = copy.deepcopy(ep)
    fxi = next(i for i in bad.truth["issues"] if i["type"] == "wrong_fx_date_or_rate")
    fxi["evidence_sets"] = [[e for e in s if not e.startswith("FX:")] for s in fxi["evidence_sets"]]
    assert any("insufficient" in e for e in validate_episode(bad))
    bad = copy.deepcopy(ep); bad.payments[0]["amount_minor"] += 1
    assert validate_episode(bad)


@pytest.mark.parametrize("cfg", [
    {"errors": {t: {"enabled": True, "count": 4} for t in FINDING_TYPES}},
    {"ledger_rows": [10, 20]}, {"n_invoices": [5, 500]}, {"bogus": 1},
    {"errors": {"duplicate_invoice": {"enabled": True, "count": -1}}},
    {"config_version": "0.1"},
])
def test_bad_configs(cfg):
    with pytest.raises(ConfigError):
        generate(1, cfg)

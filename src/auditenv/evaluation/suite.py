"""The fixed train/dev suite (10 + 10 episodes): 5 clean companies, 5 with interacting errors,
10 mixed. Seeds and categories are evaluator-side metadata and are never written to public files."""
from __future__ import annotations

from ..generator import clean_config

E = lambda d=1, a=1, f=1, p=1: {
    "duplicate_invoice": {"enabled": d > 0, "count": d},
    "missing_or_partial_allocation": {"enabled": a > 0, "count": a},
    "wrong_fx_date_or_rate": {"enabled": f > 0, "count": f},
    "period_misattribution": {"enabled": p > 0, "count": p},
}

SUITE = [
    # ---------------------------------------------------------------- train
    {"split": "train", "name": "train-01", "category": "clean", "seed": 7340021, "config": clean_config()},
    {"split": "train", "name": "train-02", "category": "clean", "seed": 5512093,
     "config": clean_config(ledger_rows=[900, 1200], row_order="shuffled")},
    {"split": "train", "name": "train-03", "category": "clean", "seed": 8820117,
     "config": clean_config(distractors={"tz_boundary": 4, "refunds": 2, "split_payments": 3})},
    {"split": "train", "name": "train-04", "category": "interaction", "seed": 6140988,
     "config": {"errors": E(0, 0, 0, 0), "interactions": {"fx_partial_boundary": 1, "duplicate_cross_period": 0}}},
    {"split": "train", "name": "train-05", "category": "interaction", "seed": 9031777,
     "config": {"errors": E(1, 0, 1, 0), "interactions": {"fx_partial_boundary": 0, "duplicate_cross_period": 1}}},
    {"split": "train", "name": "train-06", "category": "mixed", "seed": 1209331, "config": {"errors": E()}},
    {"split": "train", "name": "train-07", "category": "mixed", "seed": 4477102,
     "config": {"errors": E(3, 3, 2, 2), "n_invoices": [50, 70]}},
    {"split": "train", "name": "train-08", "category": "mixed", "seed": 3305561,
     "config": {"errors": E(0, 0, 2, 0), "foreign_share_pct": 70}},
    {"split": "train", "name": "train-09", "category": "mixed", "seed": 2718281,
     "config": {"errors": E(1, 2, 0, 1), "n_invoices": [14, 20], "ledger_rows": [200, 260]}},
    {"split": "train", "name": "train-10", "category": "mixed", "seed": 1618033,
     "config": {"errors": E(2, 1, 1, 2), "row_order": "shuffled"}},
    # ---------------------------------------------------------------- dev
    {"split": "dev", "name": "dev-01", "category": "clean", "seed": 6021023, "config": clean_config()},
    {"split": "dev", "name": "dev-02", "category": "clean", "seed": 4669201,
     "config": clean_config(n_invoices=[80, 95], ledger_rows=[1500, 1800], foreign_share_pct=60)},
    {"split": "dev", "name": "dev-03", "category": "interaction", "seed": 1414213,
     "config": {"errors": E(0, 0, 0, 0), "interactions": {"fx_partial_boundary": 1, "duplicate_cross_period": 1}}},
    {"split": "dev", "name": "dev-04", "category": "interaction", "seed": 1732050,
     "config": {"errors": E(1, 1, 0, 0), "interactions": {"fx_partial_boundary": 2, "duplicate_cross_period": 0}}},
    {"split": "dev", "name": "dev-05", "category": "interaction", "seed": 2236067,
     "config": {"errors": E(0, 1, 1, 1), "interactions": {"fx_partial_boundary": 1, "duplicate_cross_period": 2}}},
    {"split": "dev", "name": "dev-06", "category": "mixed", "seed": 5772156, "config": {"errors": E()}},
    {"split": "dev", "name": "dev-07", "category": "mixed", "seed": 6931471,
     "config": {"errors": E(2, 2, 2, 2), "ledger_rows": [1950, 2000], "n_invoices": [90, 100]}},
    {"split": "dev", "name": "dev-08", "category": "mixed", "seed": 3010299,
     "config": {"errors": E(0, 3, 0, 0), "foreign_share_pct": 0}},
    {"split": "dev", "name": "dev-09", "category": "mixed", "seed": 4771212,
     "config": {"errors": E(0, 0, 3, 3), "foreign_share_pct": 90}},
    {"split": "dev", "name": "dev-10", "category": "mixed", "seed": 8450980,
     "config": {"errors": E(1, 0, 0, 0), "row_order": "shuffled"}},
]


def suite(split: str | None = None) -> list[dict]:
    return [s for s in SUITE if split is None or s["split"] == split]

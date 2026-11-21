"""Scenarios from the case's closed check (section 06): organizers run their own seeds and parameter
combinations, apply harmless transformations to the FILES (row order, ids), plug in their own policies,
and expect hung or crashing agents to end in truncation instead of breaking the evaluator."""
import json
import os
import random
import re
import shutil
import textwrap

import pytest

from auditenv.env import AuditEnv, DirectoryProvider
from auditenv.evaluation import evaluate_policies, run_episode, summarize
from auditenv.policies import RuleBasedPolicy
from auditenv.policies.worker import RemotePolicy
from auditenv.validator import validate_files

from conftest import ROOT

ID_RE = re.compile(r"\b(EP|CO|C|V|INV|PAY|D|L)-[0-9A-F]+\b")
PUBLIC = ["manifest.json", "ledger.csv", "invoices.jsonl", "payments.jsonl", "fx.csv", "policy.json"]


def _dev_dirs():
    p = DirectoryProvider.from_split(ROOT, "dev")
    return p._pairs


def jury_transform(src_pub, src_prv, dst, seed):
    """Independent of auditenv.generator.transforms: works on raw text only."""
    r = random.Random(seed)
    texts = {f: open(os.path.join(src_pub, f), encoding="utf-8").read() for f in PUBLIC}
    truth = open(os.path.join(src_prv, "truth.json"), encoding="utf-8").read()
    ids = sorted(set(m.group(0) for t in [*texts.values(), truth] for m in ID_RE.finditer(t)))
    r.shuffle(ids)
    mapping = {old: f"{old.split('-')[0].lower()}_{i:05d}" for i, old in enumerate(ids)}
    sub = lambda t: ID_RE.sub(lambda m: mapping[m.group(0)], t)
    texts = {f: sub(t) for f, t in texts.items()}
    truth = sub(truth)
    for f in ("ledger.csv", "fx.csv"):                  # reorder data rows, keep header
        head, *rows = texts[f].splitlines()
        r.shuffle(rows)
        texts[f] = "\n".join([head, *rows]) + "\n"
    for f in ("invoices.jsonl", "payments.jsonl"):
        rows = texts[f].splitlines()
        r.shuffle(rows)
        texts[f] = "\n".join(rows) + "\n"
    pub, prv = os.path.join(dst, "pub"), os.path.join(dst, "prv")
    os.makedirs(pub); os.makedirs(prv)
    for f, t in texts.items():
        open(os.path.join(pub, f), "w", encoding="utf-8").write(t)
    open(os.path.join(prv, "truth.json"), "w", encoding="utf-8").write(truth)
    return pub, prv, mapping


def _run(pub, prv, policy):
    prov = DirectoryProvider([pub], [prv])
    env = AuditEnv(mode="closed", provider=prov)
    run = run_episode(env, policy, 0, None)
    return prov.log[-1], run


@pytest.mark.parametrize("k", range(10))
def test_file_level_reorder_and_rename_preserve_answers(tmp_path, k):
    pub0, prv0 = _dev_dirs()[k]
    pub1, prv1, mapping = jury_transform(pub0, prv0, str(tmp_path), seed=1000 + k)
    files1 = {f: open(os.path.join(pub1, f), encoding="utf-8").read() for f in PUBLIC}
    truth1 = json.load(open(os.path.join(prv1, "truth.json")))
    assert validate_files(files1, None, truth1) == []           # data + evidence still correct
    truth0 = json.load(open(os.path.join(prv0, "truth.json")))
    from auditenv.policies import OraclePolicy
    g0, _ = _run(pub0, prv0, OraclePolicy(truth0))
    g1, _ = _run(pub1, prv1, OraclePolicy(truth1))
    assert g0["reward"] == g1["reward"] == 1.0
    r0, run0 = _run(pub0, prv0, RuleBasedPolicy())
    r1, run1 = _run(pub1, prv1, RuleBasedPolicy())
    assert (r0["tp"], r0["fp"], r0["fn"], r0["reward"]) == (r1["tp"], r1["fp"], r1["fn"], r1["reward"])
    canon = lambda fs, m: sorted((f["type"], sorted(m.get(i, i) for i in f["affected_ids"]),
                                  f["claimed_amount_minor"]) for f in fs)
    assert canon(run0["last_action"]["findings"], mapping) == canon(run1["last_action"]["findings"], {})


EXTERNAL = textwrap.dedent('''
    import time

    class Hang:
        def reset(self, obs): self.eid = obs["episode_id"]
        def act(self, obs): time.sleep(3600)

    class Crash:
        def reset(self, obs): pass
        def act(self, obs): raise ValueError("boom")

    class Silent:
        def reset(self, obs): self.eid = obs["episode_id"]
        def act(self, obs): return {"action": "submit", "episode_id": self.eid, "findings": []}
''')


@pytest.fixture()
def external(tmp_path):
    (tmp_path / "jury_policies.py").write_text(EXTERNAL)
    return str(tmp_path)


def test_hung_agent_is_truncated_by_deadline(external):
    pub, prv = _dev_dirs()[0]
    prov = DirectoryProvider([pub], [prv])
    env = AuditEnv(mode="closed", provider=prov, time_limit_s=1.0)
    pol = RemotePolicy("jury_policies:Hang", [external])
    try:
        run = run_episode(env, pol, 0, None)
    finally:
        pol.close()
    assert run["agent_timeout"] and run["truncated"] and not run["terminated"]
    assert run["seconds"] < 10 and prov.log[-1]["status"] == "truncated"


def test_crashing_agent_does_not_break_the_evaluator(external):
    rows = evaluate_policies([{"episode_dir": p, "grader_dir": g} for p, g in _dev_dirs()[:3]],
                             ["jury_policies:Crash", "jury_policies:Silent"], [external])
    crash = [r for r in rows if r["policy"] == "jury_policies:Crash"]
    assert len(crash) == 3 and all(r["status"] == "truncated" and r["steps"] == 100 for r in crash)
    silent = [r for r in rows if r["policy"] == "jury_policies:Silent"]
    assert all(r["status"] == "submitted" and r["steps"] == 1 for r in silent)


@pytest.mark.parametrize("spec", ["oracle", "auditenv.policies:OraclePolicy", "nope", "os:system"])
def test_agent_process_cannot_load_the_oracle_or_junk(spec):
    p = RemotePolicy(spec)
    with pytest.raises(RuntimeError):
        p.reset({"episode_id": "x", "manifest": {}})
    p.close()


def test_hidden_seeds_and_new_parameter_combinations():
    hidden = [{"seed": 2**62 + 17, "config": {"interactions": {"fx_partial_boundary": 2,
                                                               "duplicate_cross_period": 1}}},
              {"seed": 987654321987, "config": {"n_customers": [1, 1], "foreign_share_pct": 100,
                                                "row_order": "shuffled", "ledger_rows": [1990, 2000]}}]
    s = summarize(evaluate_policies(hidden))
    assert s["oracle"]["all"]["f1_micro"] == 1.0 and s["flag_all"]["all"]["return_mean"] < -0.9


def test_cli_on_disk_episodes(tmp_path):
    from auditenv.cli import main
    assert main(["validate", "--root", ROOT, "--split", "train"]) == 0
    pub, prv = _dev_dirs()[1]
    out = tmp_path / "t.jsonl"
    assert main(["run", "--policy", "rule_based", "--episode-dir", pub, "--grader-dir", prv,
                 "--trajectory", str(out)]) == 0
    recs = [json.loads(x) for x in out.read_text().splitlines()]
    assert recs[0]["step"] == 0 and recs[-1]["terminated"]

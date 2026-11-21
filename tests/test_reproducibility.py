import hashlib
import json
import os
import subprocess
import sys

import pytest

from auditenv.env import AuditEnv, EpisodeProvider, GeneratorProvider
from auditenv.generator import generate
from auditenv.generator.transforms import permute, remap_ids
from auditenv.validator import validate_episode

from conftest import oracle_findings, submit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = {"interactions": {"fx_partial_boundary": 1}}


def transcript(seed, cfg, actions_fn):
    env = AuditEnv(provider=GeneratorProvider())
    obs, info = env.reset(seed, cfg)
    out = [obs, info]
    for a in actions_fn(obs):
        o, r, term, trunc, info = env.step(a)
        out.append([o, r, term, trunc, info])
    return json.dumps(out, sort_keys=True).encode()


def actions(obs):
    acts = [{"action": "list_records", "table": t} for t in ("ledger", "invoices", "payments", "fx")]
    acts += [{"action": "list_records", "table": "ledger", "cursor": "o50", "limit": 7},
             {"action": "calculator", "expression": "1/3"}, {"action": "get_record", "table": "fx", "id": "nope"}]
    acts.append(submit(obs["episode_id"], []))
    return acts


def test_same_seed_same_bytes():
    a = generate(5, CFG)
    b = generate(5, CFG)
    assert a.public_files() == b.public_files() and a.private_files() == b.private_files()
    assert transcript(5, CFG, actions) == transcript(5, CFG, actions)
    assert generate(6, CFG).public_files() != a.public_files()
    other = dict(CFG, row_order="shuffled")
    assert generate(5, other).public_files() != a.public_files()


def test_reproducible_across_processes_and_hash_seeds():
    code = ("import sys,hashlib,json; sys.path.insert(0,'src');"
            "from auditenv.generator import generate;"
            "e=generate(5, {'interactions': {'fx_partial_boundary': 1}});"
            "print(hashlib.sha256(json.dumps([e.public_files(), e.private_files()]).encode()).hexdigest())")
    outs = set()
    for hs in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hs)
        outs.add(subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True,
                                text=True, check=True).stdout.strip())
    e = generate(5, CFG)
    outs.add(hashlib.sha256(json.dumps([e.public_files(), e.private_files()]).encode()).hexdigest())
    assert len(outs) == 1


def _oracle_reward(ep):
    env = AuditEnv(provider=EpisodeProvider([ep]))
    obs, _ = env.reset(0)
    _, r, *_ = env.step(submit(obs["episode_id"], oracle_findings(ep.truth)))
    return r


@pytest.mark.parametrize("seed", range(300, 310))
def test_paired_permutation_and_id_change(seed):
    ep = generate(seed, {"interactions": {"fx_partial_boundary": seed % 2, "duplicate_cross_period": 1 - seed % 2}})
    perm = permute(ep, seed)
    assert perm.public_files() != ep.public_files() and perm.truth == ep.truth
    assert validate_episode(perm) == []
    ren, mapping = remap_ids(ep, seed)
    assert validate_episode(ren) == []
    blob = "".join(ren.public_files().values())
    assert not any(old in blob for old in mapping if not old.startswith("SKU"))
    for a, b in zip(ep.truth["issues"], ren.truth["issues"]):
        assert sorted(mapping[x] for x in a["affected_ids"]) == b["affected_ids"]
        assert a["expected_amount_minor"] == b["expected_amount_minor"]
    assert _oracle_reward(ep) == _oracle_reward(perm) == _oracle_reward(ren) == 1.0
    # an answer in the OLD ids is worthless after renaming (ids really changed)
    env = AuditEnv(provider=EpisodeProvider([ren]))
    obs, _ = env.reset(0)
    _, r, *_ = env.step(submit(obs["episode_id"], oracle_findings(ep.truth)))
    assert r == -1.0 or not ep.truth["issues"]

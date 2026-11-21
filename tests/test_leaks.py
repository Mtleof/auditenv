"""No ground truth, seeds, private paths, debug data or exception text may reach the agent."""
import json
import os

import pytest

from auditenv.env import AuditEnv, GeneratorProvider
from auditenv.generator import generate

from conftest import oracle_findings, submit

SEED = 918273645
CFG = {"interactions": {"fx_partial_boundary": 1, "duplicate_cross_period": 1}}
FORBIDDEN_WORDS = ["truth", "provenance", "issue_id", "expected_amount", "evidence_sets", "scenario",
                   "mutation", "Traceback", "Exception", "Error(", "grader", "/home", "private/", "seed",
                   "distractor", "injected", "slot_"]


def secrets_of(ep):
    s = {str(SEED)}
    s |= {i["issue_id"] for i in ep.truth["issues"]}
    s.add(ep.provenance["config_hash"])
    return s


def scripted_actions(ep):
    acts = []
    for t in ("ledger", "invoices", "payments", "fx"):
        acts.append({"action": "list_records", "table": t, "limit": 50})
    for i in ep.truth["issues"]:
        for e in i["evidence_sets"][0]:
            if e.startswith("D-"):
                acts.append({"action": "get_document", "doc_id": e})
            elif e.startswith("L-"):
                acts.append({"action": "get_record", "table": "ledger", "id": e})
            else:
                acts.append({"action": "get_record", "table": "fx", "id": e})
    acts += [{"action": "get_record", "table": "ledger", "id": "../../truth.json"},
             {"action": "get_document", "doc_id": "/etc/passwd"},
             {"action": "calculator", "expression": "open('truth.json')"},
             {"action": "calculator", "expression": "1/0"},
             {"action": "list_records", "table": "truth"},
             {"action": "debug"}, {"action": "list_records", "table": "ledger", "cursor": "zz"}]
    return acts


def scan(blob: str, secrets):
    for s in secrets:
        assert s not in blob, f"leaked secret {s!r}"
    low = blob.lower()
    for w in FORBIDDEN_WORDS:
        assert w.lower() not in low, f"leaked marker {w!r}"


@pytest.mark.parametrize("final", ["oracle", "empty", "invalid", "truncate"])
def test_closed_mode_never_leaks(final):
    ep = generate(SEED, CFG)
    env = AuditEnv(mode="closed", provider=GeneratorProvider())
    obs, info = env.reset(SEED, CFG)
    transcript = [obs, info]
    for a in scripted_actions(ep):
        o, r, term, trunc, info = env.step(a)
        assert r == 0.0 and set(info) == {"mode", "steps_used", "steps_remaining"}
        transcript += [o, info]
    if final == "truncate":
        while True:
            o, r, term, trunc, info = env.step({"action": "calculator", "expression": "2"})
            transcript += [o, info]
            assert r == 0.0
            if trunc:
                break
    else:
        f = {"oracle": oracle_findings(ep.truth), "empty": [], "invalid": "x"}[final]
        o, r, term, trunc, info = env.step(submit(obs["episode_id"], f))
        assert term and r == 0.0 and o["result"] == {"status": "submitted"}
        assert set(info) == {"mode", "steps_used", "steps_remaining"}
        transcript += [o, info]
    blob = json.dumps(transcript)
    scan(blob, secrets_of(ep))
    # the evidence values themselves are public records and may appear; the truth amounts list may not
    assert '"reward":' not in blob
    # the grader still graded it, on the evaluator side
    assert env._provider.log and "reward" in env._provider.log[-1]


def test_open_mode_reveals_only_terminal_scalar_reward():
    ep = generate(SEED, CFG)
    env = AuditEnv(mode="open", provider=GeneratorProvider())
    obs, info = env.reset(SEED, CFG)
    transcript = [obs, info]
    for a in scripted_actions(ep):
        o, r, term, trunc, info = env.step(a)
        assert r == 0.0 and "grade" not in info and '"reward":' not in json.dumps(o)
        transcript += [o, info]
    o, r, term, trunc, info = env.step(submit(obs["episode_id"], oracle_findings(ep.truth)[:2]))
    transcript += [o, info]
    assert set(info["grade"]) == {"status", "tp", "fp", "fn", "precision", "recall", "f1", "inspection_steps",
                                  "reward", "experimental"}
    scan(json.dumps(transcript), secrets_of(ep))


def test_public_files_contain_no_private_data():
    for seed in (SEED, SEED + 1, SEED + 2):
        ep = generate(seed, CFG)
        blob = "".join(ep.public_files().values())
        scan(blob, secrets_of(ep) | {str(seed)})


def test_env_holds_only_public_bundle():
    env = AuditEnv(mode="closed")
    env.reset(SEED, CFG)
    st = env._s
    assert set(st) == {"bundle", "grader", "steps", "done", "terminated", "truncated", "t0"}
    assert set(vars(st["bundle"])) == {"manifest", "policy", "tables", "index", "docs"}


def test_trajectories_file_is_clean():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trajectories.jsonl")
    if not os.path.exists(path):
        pytest.skip("trajectories.jsonl not generated yet")
    with open(path) as fh:
        lines = [json.loads(x) for x in fh]
    assert lines
    for rec in lines:
        assert set(rec) >= {"episode_id", "step", "action", "observation", "reward", "terminated", "truncated"}
        low = json.dumps(rec["observation"]).lower()
        for w in ("truth", "provenance", "issue_id", "expected_amount", "evidence_sets", "iss-"):
            assert w not in low

import json

import pytest

from auditenv import FINDING_TYPES
from auditenv.env import ERRORS, AuditEnv, EpisodeProvider
from auditenv.generator import generate
from auditenv.schemas import LIMITS

from conftest import FakeClock, oracle_findings, submit


def _env(ep, **kw):
    return AuditEnv(provider=EpisodeProvider([ep]), **kw)


def _assert_json(x):
    assert json.loads(json.dumps(x, allow_nan=False)) == x


def test_reset_observation(error_episode):
    env = _env(error_episode)
    obs, info = env.reset(0)
    _assert_json(obs); _assert_json(info)
    assert obs["steps_remaining"] == 100 and obs["step"] == 0
    assert obs["manifest"]["episode_id"] == obs["episode_id"] == error_episode.episode_id
    assert set(obs["finding_types"]) == set(FINDING_TYPES)
    for k in ("task", "policy", "actions", "limits"):
        assert obs[k]
    for k in ("schema_version", "episode_id", "base_currency", "period_start", "period_end", "timezone",
              "counts", "rounding"):
        assert k in obs["manifest"]


def test_step_contract_and_paging(error_episode):
    env = _env(error_episode)
    env.reset(0)
    obs, r, term, trunc, info = env.step({"action": "list_records", "table": "ledger", "limit": 500})
    _assert_json(obs)
    assert (r, term, trunc) == (0.0, False, False)
    res = obs["result"]
    assert res["limit"] == 50 and len(res["records"]) == 50 and res["total"] == len(error_episode.ledger)
    assert set(obs) == {"episode_id", "step", "steps_remaining", "action", "result", "error"}
    # stable cursors: the same cursor yields the same page
    o2, *_ = env.step({"action": "list_records", "table": "ledger", "cursor": res["next_cursor"], "limit": 50})
    o3, *_ = env.step({"action": "list_records", "table": "ledger", "cursor": res["next_cursor"], "limit": 50})
    assert o2["result"]["records"] == o3["result"]["records"]
    assert o2["result"]["records"][0] == error_episode.ledger[50] | {}
    # walk the whole fx table
    cur, seen = None, []
    while True:
        o, *_ = env.step({"action": "list_records", "table": "fx", "cursor": cur})
        seen += o["result"]["records"]
        cur = o["result"]["next_cursor"]
        if cur is None:
            break
    assert len(seen) == len(error_episode.fx)
    assert all(rec["fx_id"].startswith("FX:") for rec in seen)


def test_get_record_and_document(error_episode):
    env = _env(error_episode)
    env.reset(0)
    inv = error_episode.invoices[0]
    o, *_ = env.step({"action": "get_record", "table": "invoices", "id": inv["invoice_id"]})
    assert o["result"]["record"] == inv
    o, *_ = env.step({"action": "get_document", "doc_id": inv["doc_id"]})
    assert o["result"] == {"doc_type": "invoice", "document": inv}
    pay = error_episode.payments[0]
    o, *_ = env.step({"action": "get_document", "doc_id": pay["doc_id"]})
    assert o["result"]["doc_type"] == "payment"
    row = error_episode.ledger[3]
    o, *_ = env.step({"action": "get_record", "table": "ledger", "id": row["row_id"]})
    assert o["result"]["record"] == row
    o, *_ = env.step({"action": "calculator", "expression": "12.5*4"})
    assert o["result"]["value"] == "50"
    # observations are copies: mutating them never changes the environment
    o["result"]["value"] = "x"
    o, *_ = env.step({"action": "get_record", "table": "ledger", "id": row["row_id"]})
    o["result"]["record"]["debit_minor"] = -1
    o, *_ = env.step({"action": "get_record", "table": "ledger", "id": row["row_id"]})
    assert o["result"]["record"] == row


@pytest.mark.parametrize("action,code", [
    ({"action": "get_record", "table": "ledger", "id": "L-NOPE"}, "unknown_id"),
    ({"action": "get_document", "doc_id": "D-NOPE"}, "unknown_id"),
    ({"action": "get_record", "table": "shell", "id": "x"}, "invalid_arguments"),
    ({"action": "list_records", "table": "ledger", "cursor": "o999999"}, "invalid_cursor"),
    ({"action": "list_records", "table": "ledger", "cursor": "../etc"}, "invalid_cursor"),
    ({"action": "list_records", "table": "ledger", "limit": 0}, "invalid_arguments"),
    ({"action": "list_records", "table": "ledger", "sql": "select 1"}, "invalid_arguments"),
    ({"action": "calculator", "expression": "__import__('os').system('ls')"}, "calculator_error"),
    ({"action": "shell", "cmd": "ls"}, "invalid_action"),
    ({"no_action": 1}, "invalid_action"),
    ("list_records", "invalid_action"),
    (None, "invalid_action"),
    ({"action": "calculator", "expression": float("nan")}, "invalid_action"),
])
def test_bad_actions_consume_step(error_episode, action, code):
    env = _env(error_episode)
    env.reset(0)
    obs, r, term, trunc, info = env.step(action)
    assert obs["error"] == {"code": code, "message": ERRORS[code]}
    assert obs["result"] is None and r == 0.0 and not term and not trunc
    assert obs["steps_remaining"] == 99


def test_step_limit_truncates_with_empty_answer(error_episode):
    env = _env(error_episode)
    env.reset(0)
    for k in range(1, 100):
        obs, r, term, trunc, info = env.step({"action": "calculator", "expression": "1+1"})
        assert not term and not trunc and r == 0.0
        assert obs["steps_remaining"] == 100 - k
    obs, r, term, trunc, info = env.step({"action": "calculator", "expression": "1+1"})
    assert trunc and not term and obs["status"] == "truncated" and obs["steps_remaining"] == 0
    assert info["grade"]["status"] == "truncated" and info["grade"]["tp"] == 0
    # empty answer on an episode with errors: F1 = 0 -> 2*0-1-0.2 clipped to -1
    assert r == -1.0
    obs, r, term, trunc, info = env.step({"action": "calculator", "expression": "1+1"})
    assert obs["error"]["code"] == "episode_finished" and r == 0.0 and trunc


def test_submit_on_step_100_is_allowed(clean_episode):
    env = _env(clean_episode)
    obs, _ = env.reset(0)
    for _ in range(99):
        env.step({"action": "calculator", "expression": "1"})
    obs, r, term, trunc, info = env.step(submit(obs["episode_id"], []))
    assert term and not trunc and r == pytest.approx(1 - 0.002 * 99)


def test_timeout_truncates(clean_episode):
    clock = FakeClock()
    env = _env(clean_episode, clock=clock)
    obs, _ = env.reset(0)
    env.step({"action": "calculator", "expression": "1"})
    clock.t = 61.0
    obs, r, term, trunc, info = env.step(submit(obs["episode_id"], []))
    assert trunc and not term and obs["status"] == "truncated" and info["grade"]["status"] == "truncated"


def test_submit_terminates_and_repeat_submit_gains_nothing(error_episode):
    env = _env(error_episode)
    obs, _ = env.reset(0)
    good = submit(obs["episode_id"], oracle_findings(error_episode.truth))
    obs, r, term, trunc, info = env.step(good)
    assert term and not trunc and r == 1.0 and info["grade"]["f1"] == 1.0
    obs, r2, term, trunc, info = env.step(good)
    assert r2 == 0.0 and term and obs["error"]["code"] == "episode_finished"


@pytest.mark.parametrize("mutate", [
    lambda s: s.pop("findings"),
    lambda s: s.update(episode_id="EP-OTHER"),
    lambda s: s.update(findings="all"),
    lambda s: s.update(extra=1),
    lambda s: s["findings"].extend([s["findings"][0]] * 31),
    lambda s: s["findings"][0].update(evidence_ids=["x"] * 11),
    lambda s: s["findings"][0].update(explanation="x" * 1001),
    lambda s: s["findings"][0].update(type="fraud"),
    lambda s: s["findings"][0].update(claimed_amount_minor=1.5),
    lambda s: s["findings"][0].update(claimed_amount_minor=True),
    lambda s: s["findings"][0].update(currency="usd"),
    lambda s: s["findings"][0].update(affected_ids=[]),
    lambda s: s["findings"][0].pop("evidence_ids"),
])
def test_invalid_submission_gives_minus_one(error_episode, mutate):
    for mode in ("open", "closed"):
        env = _env(error_episode, mode=mode)
        obs, _ = env.reset(0)
        s = submit(obs["episode_id"], oracle_findings(error_episode.truth))
        mutate(s)
        obs, r, term, trunc, info = env.step(s)
        assert term and not trunc
        log = env._provider.log[-1]
        assert log["status"] == "invalid_submission" and log["reward"] == -1.0 and log["errors"]
        if mode == "open":
            assert r == -1.0 and obs["result"]["status"] == "invalid_submission"
        else:
            assert r == 0.0 and obs["result"] == {"status": "submitted"} and "grade" not in info


def test_limits_are_exact(error_episode):
    env = _env(error_episode)
    obs, _ = env.reset(0)
    f = oracle_findings(error_episode.truth)[0]
    f["evidence_ids"] = (f["evidence_ids"] * 10)[:10]
    f["explanation"] = "x" * 1000
    obs, r, term, *_ = env.step(submit(obs["episode_id"], [f] * 30))
    assert term and env._provider.log[-1]["status"] == "submitted"
    assert LIMITS["max_findings"] == 30 and LIMITS["page_size_max"] == 50 and LIMITS["max_steps"] == 100


def test_reset_isolates_state():
    eps = [generate(21), generate(22)]
    env = AuditEnv(provider=EpisodeProvider(eps))
    obs, _ = env.reset(0)
    for _ in range(5):
        env.step({"action": "calculator", "expression": "1"})
    env.step(submit(obs["episode_id"], []))
    obs2, info2 = env.reset(1)
    assert obs2["episode_id"] == eps[1].episode_id and info2["steps_used"] == 0
    o, r, term, trunc, _ = env.step({"action": "get_record", "table": "invoices", "id": eps[0].invoices[0]["invoice_id"]})
    assert o["error"]["code"] == "unknown_id" and not term
    obs3, _ = env.reset(1)
    assert obs3 == obs2


def test_step_before_reset():
    with pytest.raises(RuntimeError):
        AuditEnv().step({"action": "calculator", "expression": "1"})

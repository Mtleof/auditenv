"""Attacks on the environment and the reward."""
import copy

from auditenv.env import AuditEnv, EpisodeProvider
from auditenv.schemas import LIMITS

from conftest import oracle_findings, submit


def _env(ep, mode="open"):
    return AuditEnv(mode=mode, provider=EpisodeProvider([ep]))


def test_oversized_and_non_json_actions(error_episode):
    env = _env(error_episode)
    env.reset(0)
    o, *_ = env.step({"action": "calculator", "expression": "1" * (LIMITS["max_action_bytes"] + 10)})
    assert o["error"]["code"] == "invalid_arguments"
    o, *_ = env.step({"action": "calculator", "expression": {1, 2}})
    assert o["error"]["code"] == "invalid_action"
    o, *_ = env.step({"action": "get_record", "table": "ledger", "id": "x" * 65})
    assert o["error"]["code"] == "invalid_arguments"


def test_huge_submission_is_invalid(error_episode):
    env = _env(error_episode)
    obs, _ = env.reset(0)
    f = oracle_findings(error_episode.truth)
    f[0]["explanation"] = "x" * 70000
    o, r, term, *_ = env.step(submit(obs["episode_id"], f))
    assert term and r == -1.0


def test_brute_force_guessing_is_unprofitable(error_episode):
    """Submit the max number of plausible guesses for every type: worse than an honest partial answer."""
    ep = error_episode
    guesses = []
    for p in ep.payments[:15]:
        guesses.append({"type": "missing_or_partial_allocation",
                        "affected_ids": [p["payment_id"]] + [x["invoice_id"] for x in p["invoice_refs"]],
                        "claimed_amount_minor": p["amount_minor"], "currency": p["currency"],
                        "evidence_ids": [p["doc_id"]], "explanation": ""})
    for inv in ep.invoices[:15]:
        guesses.append({"type": "period_misattribution", "affected_ids": [inv["invoice_id"]],
                        "claimed_amount_minor": inv["total_minor"], "currency": ep.manifest["base_currency"],
                        "evidence_ids": [inv["doc_id"]], "explanation": ""})
    env = _env(ep)
    obs, _ = env.reset(0)
    _, r_guess, *_ = env.step(submit(obs["episode_id"], guesses[:30]))
    env = _env(ep)
    obs, _ = env.reset(0)
    _, r_honest, *_ = env.step(submit(obs["episode_id"], oracle_findings(ep.truth)[:2]))
    assert r_guess < 0 and r_honest > r_guess


def test_padding_with_correct_findings_does_not_help(error_episode):
    ep = error_episode
    f = oracle_findings(ep.truth)
    env = _env(ep); obs, _ = env.reset(0)
    _, r1, *_ = env.step(submit(obs["episode_id"], f))
    env = _env(ep); obs, _ = env.reset(0)
    _, r2, *_ = env.step(submit(obs["episode_id"], (f * 8)[:30]))
    assert r2 < r1


def test_evidence_from_other_issue_does_not_count(error_episode):
    f = oracle_findings(error_episode.truth)
    g = copy.deepcopy(f)
    g[0]["evidence_ids"], g[1]["evidence_ids"] = f[1]["evidence_ids"], f[0]["evidence_ids"]
    env = _env(error_episode); obs, _ = env.reset(0)
    env.step(submit(obs["episode_id"], g))
    assert env._provider.log[-1]["tp"] == len(f) - 2


def test_closed_mode_gives_no_signal_between_good_and_bad(error_episode):
    outs = []
    for findings in (oracle_findings(error_episode.truth), []):
        env = _env(error_episode, "closed"); obs, _ = env.reset(0)
        o, r, term, trunc, info = env.step(submit(obs["episode_id"], findings))
        outs.append((o, r, term, trunc, info))
    assert outs[0] == outs[1]

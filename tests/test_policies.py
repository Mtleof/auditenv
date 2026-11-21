"""Four baselines on the 20-episode suite (5 clean, 5 interacting) + 20 paired checks."""
import pytest

from auditenv.env import AuditEnv, GeneratorProvider
from auditenv.evaluation import (POLICY_ORDER, evaluate_policies, paired_checks,
                                 run_episode, summarize, verify_materialized)
from auditenv.evaluation.suite import SUITE
from auditenv.generator import generate
from auditenv.policies import AGENT_POLICIES, make_policy
from auditenv.policies.worker import RemotePolicy

from conftest import ROOT


@pytest.fixture(scope="module")
def rows():
    return evaluate_policies()


def test_suite_composition():
    cats = [s["category"] for s in SUITE]
    assert len(SUITE) == 20 and cats.count("clean") == 5 and cats.count("interaction") == 5
    assert sum(s["split"] == "train" for s in SUITE) == 10


def test_baseline_ordering(rows):
    s = summarize(rows)
    ret = {k: v["all"]["return_mean"] for k, v in s.items()}
    assert set(s) == set(POLICY_ORDER)
    assert s["oracle"]["all"]["f1_micro"] == 1.0 and ret["oracle"] == 1.0
    assert ret["oracle"] > ret["rule_based"] > ret["random"] > -1.0
    assert ret["rule_based"] > ret["flag_all"]
    assert s["flag_all"]["all"]["precision_micro"] < 0.05 and ret["flag_all"] < -0.9
    assert s["flag_all"]["clean"]["return_mean"] == -1.0
    assert s["rule_based"]["clean"]["f1_mean"] == 1.0          # no false accusations on clean companies
    # interacting errors are harder for the simple baseline than for the oracle
    assert s["rule_based"]["interaction"]["recall_micro"] < 1.0
    for r in rows:
        assert r["steps"] <= 100 and r["seconds"] < 60
        assert r["visible_reward"] == 0.0                     # closed mode: nothing visible to the agent


def test_paired_checks_all_pass():
    pairs = paired_checks()
    assert len(pairs) == 20
    assert all(p["passed"] for p in pairs), [p for p in pairs if not p["passed"]]


def test_remote_policy_matches_in_process():
    spec = SUITE[5]
    for name in AGENT_POLICIES:
        acts = []
        for pol in (make_policy(name), RemotePolicy(name)):
            env = AuditEnv(mode="closed", provider=GeneratorProvider())
            acts.append(run_episode(env, pol, spec["seed"], spec["config"])["last_action"])
            if isinstance(pol, RemotePolicy):
                pol.close()
        assert acts[0] == acts[1]


def test_worker_refuses_oracle():
    p = RemotePolicy("oracle")
    with pytest.raises(RuntimeError):
        p.reset({"episode_id": "x"})
    p._p.wait(timeout=10)
    assert p._p.returncode == 2


def test_oracle_needs_truth():
    with pytest.raises(ValueError):
        make_policy("oracle")


def test_materialized_episodes_are_current():
    assert verify_materialized(ROOT) == []

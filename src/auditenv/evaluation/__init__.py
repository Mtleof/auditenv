"""Evaluation harness: closed-mode runs of the four baselines, paired metamorphic checks,
report generation, trajectory export and episode materialization."""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict

from ..env import AuditEnv, DirectoryProvider, EpisodeProvider, GeneratorProvider
from ..generator import generate
from ..generator.episode import dump_json
from ..generator.transforms import permute, remap_ids
from ..matcher import canonical_ids
from ..policies import AGENT_POLICIES, OraclePolicy, RuleBasedPolicy
from ..policies.worker import AgentTimeout, RemotePolicy
from ..validator import validate_episode
from .suite import SUITE, suite

POLICY_ORDER = ("random", "rule_based", "flag_all", "oracle")


def run_episode(env, policy, seed, config, on_step=None):
    """Run one episode. A RemotePolicy gets a hard deadline: if the agent process does not answer before
    the episode time limit it is killed, and the next env.step() observes the expired clock and
    truncates the episode with an empty answer."""
    t0 = time.perf_counter()
    obs, info = env.reset(seed, config)
    deadline = time.monotonic() + env.time_limit_s + 0.5
    remote = isinstance(policy, RemotePolicy)
    timed_out = False

    def call(fn, o):
        return fn(o, timeout=deadline - time.monotonic()) if remote else fn(o)

    failed = False      # a crashed agent process: its episode runs out of steps with an empty answer
    try:
        call(policy.reset, obs)
    except AgentTimeout:
        timed_out = True
    except RuntimeError:
        if not remote:
            raise
        failed = True
    if on_step:
        on_step(0, None, obs, 0.0, False, False)
    action = None
    while True:
        if not (timed_out or failed):
            try:
                action = call(policy.act, obs)
            except AgentTimeout:
                timed_out = True
            except RuntimeError:
                if not remote:
                    raise
                failed = True
        if timed_out:
            action = {"action": "agent_timeout"}
        elif failed:
            action = {"action": "agent_failed"}
        obs, r, term, trunc, info = env.step(action)
        if on_step:
            on_step(info["steps_used"], action, obs, r, term, trunc)
        if term or trunc:
            break
    return {"steps": info["steps_used"], "seconds": time.perf_counter() - t0, "last_action": action,
            "visible_reward": r, "terminated": term, "truncated": trunc, "agent_timeout": timed_out,
            "agent_failed": failed}


def _row(spec, grade, run, policy):
    return {"policy": policy, "name": spec.get("name", grade["episode_id"]), "split": spec.get("split", "-"),
            "category": spec.get("category", "custom"),
            "episode_id": grade["episode_id"], "status": grade["status"], "tp": grade["tp"], "fp": grade["fp"],
            "fn": grade["fn"], "precision": grade["precision"], "recall": grade["recall"], "f1": grade["f1"],
            "return": grade["reward"], "steps": run["steps"], "seconds": round(run["seconds"], 4),
            "visible_reward": run["visible_reward"], "experimental": grade["experimental"]}


def episode_source(spec):
    """-> (provider, seed, config, truth). A spec is either {seed, config} (generated) or
    {episode_dir, grader_dir} (files on disk, possibly transformed by a third party)."""
    if "episode_dir" in spec:
        prov = DirectoryProvider([spec["episode_dir"]], [spec["grader_dir"]])
        return prov, 0, None, prov.load(0)[1]
    return GeneratorProvider(), spec["seed"], spec.get("config"), None


def evaluate_policies(specs=None, policies=POLICY_ORDER, extra_paths=(), time_limit_s=None) -> list[dict]:
    """Closed-mode evaluation. Every non-oracle policy (built-in name or "module:Class") runs in a separate
    agent process; the oracle runs evaluator-side with the truth."""
    specs = specs or SUITE
    rows = []
    for name in policies:
        remote = None if name == "oracle" else RemotePolicy(name, extra_paths)
        try:
            for spec in specs:
                provider, seed, config, truth = episode_source(spec)
                kw = {} if time_limit_s is None else {"time_limit_s": time_limit_s}
                env = AuditEnv(mode="closed", provider=provider, **kw)
                if remote is None:
                    truth = truth or generate(seed, config).truth
                    pol = OraclePolicy(truth)
                else:
                    pol = remote
                run = run_episode(env, pol, seed, config)
                row = _row(spec, provider.log[-1], run, name)
                row["agent_timeout"] = run["agent_timeout"]
                rows.append(row)
        finally:
            if remote:
                remote.close()
    return rows


def _closed_run(ep, policy):
    provider = EpisodeProvider([ep])
    env = AuditEnv(mode="closed", provider=provider)
    run = run_episode(env, policy, 0, None)
    return provider.log[-1], run


def _canon_findings(findings, mapping=None):
    m = mapping or {}
    out = []
    for f in findings:
        out.append((f["type"], canonical_ids(m.get(i, i) for i in f["affected_ids"]), f["claimed_amount_minor"],
                    f["currency"], canonical_ids(m.get(e, e) for e in f["evidence_ids"])))
    return sorted(out)


def paired_checks(specs=None) -> list[dict]:
    """20 paired checks: permutation and id renaming of 10 episodes with errors."""
    specs = specs or [s for s in SUITE if s["category"] == "interaction"] + \
        [s for s in SUITE if s["category"] == "mixed"][:5]
    out = []
    for spec in specs:
        ep = generate(spec["seed"], spec["config"])
        g0, _ = _closed_run(ep, OraclePolicy(ep.truth))
        rb0, run0 = _closed_run(ep, RuleBasedPolicy())
        f0 = run0["last_action"]["findings"]
        for kind in ("permute", "remap_ids"):
            if kind == "permute":
                ep2, mapping = permute(ep, spec["seed"]), {}
            else:
                ep2, mapping = remap_ids(ep, spec["seed"])
            g1, _ = _closed_run(ep2, OraclePolicy(ep2.truth))
            rb1, run1 = _closed_run(ep2, RuleBasedPolicy())
            f1 = run1["last_action"]["findings"]
            checks = {
                "validator_ok": validate_episode(ep2) == [],
                "oracle_reward_equal": g0["reward"] == g1["reward"] == 1.0,
                "rule_based_reward_equal": rb0["reward"] == rb1["reward"],
                "rule_based_answer_equal": _canon_findings(f0, mapping) == _canon_findings(f1),
                "files_changed": ep2.public_files() != ep.public_files(),
            }
            out.append({"name": spec["name"], "transform": kind, "passed": all(checks.values()), **checks,
                        "oracle_reward": g1["reward"], "rule_based_reward": rb1["reward"]})
    return out


def summarize(rows: list[dict]) -> dict:
    out = {}
    names = list(dict.fromkeys(r["policy"] for r in rows))
    cats = list(dict.fromkeys(r["category"] for r in rows))
    for name in names:
        rs = [r for r in rows if r["policy"] == name]
        groups = {"all": rs}
        for cat in cats:
            groups[cat] = [r for r in rs if r["category"] == cat]
        out[name] = {}
        for g, sub in groups.items():
            if not sub:
                continue
            tp, fp, fn = (sum(r[k] for r in sub) for k in ("tp", "fp", "fn"))
            n = len(sub)
            out[name][g] = {
                "episodes": n, "tp": tp, "fp": fp, "fn": fn,
                "precision_micro": round(tp / (tp + fp), 4) if tp + fp else 0.0,
                "recall_micro": round(tp / (tp + fn), 4) if tp + fn else 0.0,
                "f1_micro": round(2 * tp / (2 * tp + fp + fn), 4) if tp + fp + fn else 1.0,
                "f1_mean": round(sum(r["f1"] for r in sub) / n, 4),
                "return_mean": round(sum(r["return"] for r in sub) / n, 4),
                "steps_mean": round(sum(r["steps"] for r in sub) / n, 2),
                "seconds_mean": round(sum(r["seconds"] for r in sub) / n, 4),
                "seconds_max": round(max(r["seconds"] for r in sub), 4),
                "experimental_partial_credit_f1_mean": round(
                    sum(r["experimental"]["partial_credit_f1"] for r in sub) / n, 4),
            }
    return out


BUILTIN_TITLE = ("Closed-mode evaluation on 20 episodes (10 train + 10 dev: 5 clean, 5 with interacting errors, "
                 "10 mixed).")


def render_markdown(summary, rows, pairs, title=BUILTIN_TITLE) -> str:
    """title=None -> generic title for a custom suite."""
    eps = len({r["episode_id"] for r in rows})
    cats = list(dict.fromkeys(r["category"] for r in rows))
    L = ["# auditenv baseline report", "", title if title else f"Closed-mode evaluation on {eps} episodes.",
         "Return = clip(2*F1 - 1 - 0.002*inspection_steps, -1, 1). P/R/F1 are micro-averaged over episodes;",
         "`F1 mean` is the per-episode average. The experimental partial-credit F1 (type + ids only, no evidence)",
         "is reported separately and is NOT part of the reward.", ""]
    for g in ["all", *cats]:
        L += [f"## {g}", "", "| policy | P | R | F1 | F1 mean | return | steps | time s (mean/max) | exp. partial F1 |",
              "|---|---|---|---|---|---|---|---|---|"]
        for name, s in summary.items():
            if g in s:
                x = s[g]
                L.append(f"| {name} | {x['precision_micro']:.3f} | {x['recall_micro']:.3f} | {x['f1_micro']:.3f} | "
                         f"{x['f1_mean']:.3f} | {x['return_mean']:+.3f} | {x['steps_mean']:.1f} | "
                         f"{x['seconds_mean']:.3f}/{x['seconds_max']:.3f} | {x['experimental_partial_credit_f1_mean']:.3f} |")
        L.append("")
    L += ["## Per-episode results", "", "| policy | episode | category | TP | FP | FN | F1 | return | steps |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['policy']} | {r['name']} | {r['category']} | {r['tp']} | {r['fp']} | {r['fn']} | "
                 f"{r['f1']:.3f} | {r['return']:+.3f} | {r['steps']} |")
    if not pairs:
        return "\n".join(L) + "\n"
    passed = sum(p["passed"] for p in pairs)
    L += ["", f"## Paired checks: {passed}/{len(pairs)} passed", "",
          "| episode | transform | validator | oracle reward equal | rule-based reward equal | rule-based answer equal |",
          "|---|---|---|---|---|---|"]
    for p in pairs:
        L.append(f"| {p['name']} | {p['transform']} | {p['validator_ok']} | {p['oracle_reward_equal']} | "
                 f"{p['rule_based_reward_equal']} | {p['rule_based_answer_equal']} |")
    return "\n".join(L) + "\n"


def write_trajectories(path: str, split: str = "train", policies=("random", "rule_based", "flag_all")) -> int:
    """Open-mode (training) trajectories. The oracle is excluded: its actions are the truth."""
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for name in policies:
            for spec in suite(split):
                env = AuditEnv(mode="open", provider=GeneratorProvider())
                pol = AGENT_POLICIES[name]()
                ep_id = []

                def on_step(step, action, obs, r, term, trunc):
                    nonlocal n
                    if step == 0:
                        ep_id.append(obs["episode_id"])
                    fh.write(json.dumps({"policy": name, "episode_id": ep_id[0], "step": step, "action": action,
                                         "observation": obs, "reward": r, "terminated": term, "truncated": trunc},
                                        ensure_ascii=False, separators=(",", ":")) + "\n")
                    n += 1
                run_episode(env, pol, spec["seed"], spec["config"], on_step)
    return n


def materialize(root: str) -> list[dict]:
    """Write public episode files to episodes/<split>/<episode_id>/ and the private grader files to
    grader/<split>/<episode_id>/ (outside the public tree), plus a private index."""
    index = defaultdict(list)
    for spec in SUITE:
        ep = generate(spec["seed"], spec["config"])
        pub = os.path.join(root, "episodes", spec["split"], ep.episode_id)
        prv = os.path.join(root, "grader", spec["split"], ep.episode_id)
        for d, files in ((pub, ep.public_files()), (prv, ep.private_files())):
            os.makedirs(d, exist_ok=True)
            for fname, content in files.items():
                with open(os.path.join(d, fname), "w", encoding="utf-8", newline="\n") as fh:
                    fh.write(content)
        index[spec["split"]].append({"name": spec["name"], "episode_id": ep.episode_id, "seed": spec["seed"],
                                     "category": spec["category"], "config": spec["config"]})
    for split, items in index.items():
        with open(os.path.join(root, "grader", split, "index.json"), "w", encoding="utf-8") as fh:
            fh.write(dump_json(items))
    return [x for v in index.values() for x in v]


def verify_materialized(root: str) -> list[str]:
    errs = []
    for spec in SUITE:
        ep = generate(spec["seed"], spec["config"])
        for base, files in ((os.path.join(root, "episodes", spec["split"], ep.episode_id), ep.public_files()),
                            (os.path.join(root, "grader", spec["split"], ep.episode_id), ep.private_files())):
            for fname, content in files.items():
                p = os.path.join(base, fname)
                if not os.path.exists(p):
                    errs.append(f"missing {spec['name']}/{fname}")
                    continue
                with open(p, encoding="utf-8") as fh:
                    if fh.read() != content:
                        errs.append(f"stale {spec['name']}/{fname}")
        errs += [f"{spec['name']}: {e}" for e in validate_episode(ep)]
    for split in ("train", "dev"):
        d = os.path.join(root, "episodes", split)
        if os.path.isdir(d) and len(os.listdir(d)) != 10:
            errs.append(f"{split} must contain exactly 10 episodes")
    return errs

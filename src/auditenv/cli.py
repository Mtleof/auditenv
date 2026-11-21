"""Command line: python -m auditenv.cli {evaluate,run,validate,materialize,verify,demo}

    evaluate                      4 baselines on the built-in 20-episode suite + 20 paired checks
    evaluate --suite FILE.json    the same on any list of {"seed", "config"} or {"episode_dir", "grader_dir"}
    evaluate --root R --split S   the same on episode folders on disk (e.g. after reordering / renaming ids)
    run                           one episode, any policy ("module:Class" allowed), prints every step
    validate --root R --split S   independent validator on episode folders on disk
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


def _root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _disk_specs(root, split):
    from .env import DirectoryProvider
    prov = DirectoryProvider.from_split(root, split)
    return [{"name": os.path.basename(e), "split": split, "category": "disk", "episode_dir": e, "grader_dir": g}
            for e, g in prov._pairs]


def cmd_evaluate(a) -> int:
    from .evaluation import (BUILTIN_TITLE, POLICY_ORDER, evaluate_policies, paired_checks, render_markdown, summarize,
                             write_trajectories)
    t0 = time.perf_counter()
    builtin = not a.suite and not a.split
    if a.suite:
        specs = _load_json(a.suite)
    elif a.split:
        specs = _disk_specs(a.root, a.split)
    else:
        specs = None
    policies = a.policies.split(",") if a.policies else list(POLICY_ORDER)
    rows = evaluate_policies(specs, policies, a.extra_path, a.time_limit)
    pairs = paired_checks() if (builtin if a.pairs is None else a.pairs) else []
    summary = summarize(rows)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "report.json"), "w") as fh:
        json.dump({"summary": summary, "episodes": rows, "paired_checks": pairs}, fh, indent=2)
    md = render_markdown(summary, rows, pairs, BUILTIN_TITLE if builtin else None)
    with open(os.path.join(a.out, "report.md"), "w") as fh:
        fh.write(md)
    n = write_trajectories(a.trajectories) if (a.trajectories and builtin) else 0
    print(f"{'policy':<22} P      R      F1     return  steps  time_s")
    for name, s in summary.items():
        x = s["all"]
        print(f"{name:<22} {x['precision_micro']:.3f}  {x['recall_micro']:.3f}  {x['f1_micro']:.3f}  "
              f"{x['return_mean']:+.3f}  {x['steps_mean']:5.1f}  {x['seconds_mean']:.3f}")
    ok_pairs = sum(p["passed"] for p in pairs)
    print(f"episodes: {len({r['episode_id'] for r in rows})}; paired checks: {ok_pairs}/{len(pairs)} passed; "
          f"trajectories: {n} lines; report: {os.path.join(a.out, 'report.md')}; total {time.perf_counter() - t0:.1f}s")
    problems = []
    if ok_pairs != len(pairs):
        problems.append("paired checks failed")
    if any(r["steps"] > 100 for r in rows):
        problems.append("episode exceeded 100 steps")
    if "oracle" in summary and summary["oracle"]["all"]["f1_micro"] != 1.0:
        problems.append("oracle is not perfect")
    for p in problems:
        print("ERROR:", p, file=sys.stderr)
    return 1 if problems else 0


def cmd_run(a) -> int:
    from .env import AuditEnv, DirectoryProvider, GeneratorProvider
    from .evaluation import run_episode
    from .generator import generate
    from .policies import OraclePolicy
    from .policies.worker import RemotePolicy
    if a.episode_dir:
        provider = DirectoryProvider([a.episode_dir], [a.grader_dir])
        seed, config, truth = 0, None, provider.load(0)[1]
    else:
        provider = GeneratorProvider()
        seed, config, truth = a.seed, (_load_json(a.config) if a.config else None), None
    env = AuditEnv(mode=a.mode, provider=provider)
    if a.policy == "oracle":
        pol = OraclePolicy(truth or generate(seed, config).truth)
    else:
        pol = RemotePolicy(a.policy, a.extra_path)
    traj = open(a.trajectory, "w", encoding="utf-8") if a.trajectory else None
    ep = []

    def on_step(step, action, obs, r, term, trunc):
        if step == 0:
            ep.append(obs["episode_id"])
            m = obs["manifest"]
            print(f"episode {obs['episode_id']}  base {m['base_currency']}  period {m['period_start']}..{m['period_end']}"
                  f"  tz {m['timezone']}  rows {m['counts']['ledger_rows']}  docs {m['counts']['documents']}")
        else:
            brief = {k: v for k, v in action.items() if k != "findings"} if isinstance(action, dict) else action
            extra = f" findings={len(action.get('findings', []))}" if isinstance(action, dict) and "findings" in action else ""
            err = obs.get("error") and obs["error"]["code"]
            print(f"step {step:>3}  {json.dumps(brief)[:80]}{extra}  -> {err or 'ok'}  reward={r}  "
                  f"term={term} trunc={trunc}")
        if traj:
            traj.write(json.dumps({"episode_id": ep[0], "step": step, "action": action, "observation": obs,
                                   "reward": r, "terminated": term, "truncated": trunc},
                                  ensure_ascii=False, separators=(",", ":")) + "\n")
    try:
        run = run_episode(env, pol, seed, config, on_step)
    finally:
        if isinstance(pol, RemotePolicy):
            pol.close()
        if traj:
            traj.close()
    g = provider.log[-1]
    print("evaluator grade:", json.dumps({k: g[k] for k in ("status", "tp", "fp", "fn", "precision", "recall",
                                                              "f1", "reward", "inspection_steps")}),
          f"time={run['seconds']:.3f}s", "(agent timed out)" if run["agent_timeout"] else "")
    return 0


def cmd_validate(a) -> int:
    from .env import DirectoryProvider
    from .validator import validate_files
    if a.episode_dir:
        prov = DirectoryProvider([a.episode_dir], [a.grader_dir])
    else:
        prov = DirectoryProvider.from_split(a.root, a.split)
    bad = 0
    for i in range(len(prov)):
        files, truth, bundle = prov.load(i)
        errs = validate_files(files, None, truth)      # (b) declared discrepancies, (c) evidence sufficiency
        bad += bool(errs)
        print(f"{bundle.episode_id}: {'OK' if not errs else 'FAIL'}  issues={len(truth['issues'])}")
        for e in errs[:5]:
            print("   ", e)
    print(f"{len(prov) - bad}/{len(prov)} episodes valid")
    return 1 if bad else 0


def cmd_materialize(a) -> int:
    from .evaluation import materialize
    items = materialize(a.root)
    print(f"materialized {len(items)} episodes under {a.root}/episodes (public) and {a.root}/grader (private)")
    return 0


def cmd_verify(a) -> int:
    from .evaluation import verify_materialized
    errs = verify_materialized(a.root)
    for e in errs:
        print("ERROR:", e, file=sys.stderr)
    print("episodes verified" if not errs else f"{len(errs)} problems")
    return 1 if errs else 0


def cmd_demo(a) -> int:
    sys.path.insert(0, os.path.join(_root(), "scripts"))
    from demo import main
    return main(a.seed)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="auditenv", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("evaluate", help="closed-mode evaluation of policies")
    e.add_argument("--out", default=os.path.join(_root(), "reports"))
    e.add_argument("--trajectories", default=os.path.join(_root(), "trajectories.jsonl"),
                   help="written for the built-in suite only; '' to skip")
    e.add_argument("--suite", help="JSON list of {name?, category?, seed, config} or {episode_dir, grader_dir}")
    e.add_argument("--root", default=_root())
    e.add_argument("--split", help="evaluate episode folders <root>/episodes/<split> with <root>/grader/<split>")
    e.add_argument("--policies", help="comma list: random,rule_based,flag_all,oracle or package.module:Class")
    e.add_argument("--extra-path", action="append", default=[], help="import path for external policies")
    e.add_argument("--time-limit", type=float, default=None, help="seconds per episode (default 60)")
    e.add_argument("--pairs", dest="pairs", action="store_true", default=None)
    e.add_argument("--no-pairs", dest="pairs", action="store_false")
    e.set_defaults(fn=cmd_evaluate)

    r = sub.add_parser("run", help="run one episode and print every step")
    r.add_argument("--policy", default="rule_based")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--config", help="JSON config file")
    r.add_argument("--episode-dir")
    r.add_argument("--grader-dir")
    r.add_argument("--mode", choices=["open", "closed"], default="closed")
    r.add_argument("--extra-path", action="append", default=[])
    r.add_argument("--trajectory", help="write this episode's trajectory as JSONL")
    r.set_defaults(fn=cmd_run)

    v = sub.add_parser("validate", help="independent validator on episode folders on disk")
    v.add_argument("--root", default=_root())
    v.add_argument("--split", default="dev")
    v.add_argument("--episode-dir")
    v.add_argument("--grader-dir")
    v.set_defaults(fn=cmd_validate)

    m = sub.add_parser("materialize")
    m.add_argument("--root", default=_root())
    m.set_defaults(fn=cmd_materialize)
    vf = sub.add_parser("verify")
    vf.add_argument("--root", default=_root())
    vf.set_defaults(fn=cmd_verify)
    d = sub.add_parser("demo")
    d.add_argument("--seed", type=int, default=2024)
    d.set_defaults(fn=cmd_demo)
    a = p.parse_args(argv)
    if getattr(a, "episode_dir", None) and not getattr(a, "grader_dir", None):
        p.error("--episode-dir needs --grader-dir")
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

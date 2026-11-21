"""Command line: python -m auditenv.cli {evaluate,materialize,verify,demo}"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


def _root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cmd_evaluate(a) -> int:
    from .evaluation import (evaluate_policies, paired_checks, render_markdown, summarize, write_trajectories)
    t0 = time.perf_counter()
    rows = evaluate_policies()
    pairs = paired_checks()
    summary = summarize(rows)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "report.json"), "w") as fh:
        json.dump({"summary": summary, "episodes": rows, "paired_checks": pairs}, fh, indent=2)
    md = render_markdown(summary, rows, pairs)
    with open(os.path.join(a.out, "report.md"), "w") as fh:
        fh.write(md)
    n = write_trajectories(a.trajectories) if a.trajectories else 0
    print("policy       P      R      F1     return  steps  time_s")
    for name, s in summary.items():
        x = s["all"]
        print(f"{name:<11} {x['precision_micro']:.3f}  {x['recall_micro']:.3f}  {x['f1_micro']:.3f}  "
              f"{x['return_mean']:+.3f}  {x['steps_mean']:5.1f}  {x['seconds_mean']:.3f}")
    ok_pairs = sum(p["passed"] for p in pairs)
    print(f"paired checks: {ok_pairs}/{len(pairs)} passed; trajectories: {n} lines; "
          f"report: {os.path.join(a.out, 'report.md')}; total {time.perf_counter() - t0:.1f}s")
    problems = []
    if ok_pairs != len(pairs):
        problems.append("paired checks failed")
    if any(r["seconds"] > 60 or r["steps"] > 100 for r in rows):
        problems.append("episode exceeded 60 s or 100 steps")
    if summary["oracle"]["all"]["f1_micro"] != 1.0:
        problems.append("oracle is not perfect")
    for p in problems:
        print("ERROR:", p, file=sys.stderr)
    return 1 if problems else 0


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
    p = argparse.ArgumentParser(prog="auditenv")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--out", default=os.path.join(_root(), "reports"))
    e.add_argument("--trajectories", default=os.path.join(_root(), "trajectories.jsonl"))
    e.set_defaults(fn=cmd_evaluate)
    m = sub.add_parser("materialize")
    m.add_argument("--root", default=_root())
    m.set_defaults(fn=cmd_materialize)
    v = sub.add_parser("verify")
    v.add_argument("--root", default=_root())
    v.set_defaults(fn=cmd_verify)
    d = sub.add_parser("demo")
    d.add_argument("--seed", type=int, default=2024)
    d.set_defaults(fn=cmd_demo)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

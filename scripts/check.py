"""Single verification entry point (what `make check` runs), in pure Python so it also works
where `make` is unavailable (the Docker image, Windows).

    1. verify   - materialized episodes/{train,dev} == regenerated, and they pass the validator
    2. evaluate - 4 baselines x 20 episodes (closed mode) + 20 paired checks -> reports/, trajectories.jsonl
    3. test     - the full pytest suite (runs last so it also checks the fresh trajectories.jsonl)"""
from __future__ import annotations

import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEPS = [
    ("verify", ["-m", "auditenv.cli", "verify"]),
    ("evaluate", ["-m", "auditenv.cli", "evaluate", "--out", "reports", "--trajectories", "trajectories.jsonl"]),
    ("test", ["-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"]),
]


def main() -> int:
    if sys.version_info[:2] != (3, 11):
        print(f"warning: expected Python 3.11, running {sys.version.split()[0]}", file=sys.stderr)
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"), PYTHONHASHSEED="0")
    for name, args in STEPS:
        t0 = time.perf_counter()
        print(f"== {name}", flush=True)
        rc = subprocess.run([sys.executable, *args], cwd=ROOT, env=env).returncode
        if rc != 0:
            print(f"check FAILED at step '{name}' (exit {rc})", file=sys.stderr)
            return rc
        print(f"== {name}: ok ({time.perf_counter() - t0:.1f}s)", flush=True)
    print("check: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

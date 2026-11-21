"""Agent-side process. Runs an observation-only policy in a SEPARATE process that never receives
truth or provenance: the evaluator sends observations over stdin and reads actions from stdout.

Protocol (one JSON object per line):
    -> {"cmd": "reset", "obs": {...}}   <- {"ok": true}
    -> {"cmd": "act",   "obs": {...}}   <- {"action": {...}}
    -> {"cmd": "close"}"""
from __future__ import annotations

import json
import os
import subprocess
import sys

from . import AGENT_POLICIES


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    name = argv[0]
    if name not in AGENT_POLICIES:
        print("policy not available to agents", file=sys.stderr)
        return 2
    policy = AGENT_POLICIES[name]()
    for line in sys.stdin:
        msg = json.loads(line)
        if msg["cmd"] == "reset":
            policy.reset(msg["obs"])
            out = {"ok": True}
        elif msg["cmd"] == "act":
            out = {"action": policy.act(msg["obs"])}
        else:
            break
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()
    return 0


class RemotePolicy:
    """Evaluator-side proxy for a policy living in its own process."""

    def __init__(self, name: str):
        self.name = name
        src = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": src, "PYTHONHASHSEED": "0"}
        self._p = subprocess.Popen([sys.executable, "-m", "auditenv.policies.worker", name], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env, cwd=src)

    def _call(self, msg):
        try:
            self._p.stdin.write(json.dumps(msg) + "\n")
            self._p.stdin.flush()
        except (BrokenPipeError, OSError):
            raise RuntimeError("agent process died") from None
        line = self._p.stdout.readline()
        if not line:
            raise RuntimeError("agent process died")
        return json.loads(line)

    def reset(self, obs):
        self._call({"cmd": "reset", "obs": obs})

    def act(self, obs):
        return self._call({"cmd": "act", "obs": obs})["action"]

    def close(self):
        try:
            self._p.stdin.write(json.dumps({"cmd": "close"}) + "\n")
            self._p.stdin.close()
        except (BrokenPipeError, ValueError):
            pass
        self._p.wait(timeout=10)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


if __name__ == "__main__":
    sys.exit(main())

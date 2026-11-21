"""Agent-side process. Runs an observation-only policy in a SEPARATE process that never receives
truth or provenance: the evaluator sends observations over stdin and reads actions from stdout.

Policy spec: a built-in name ("random", "rule_based", "flag_all") or "package.module:ClassName" for an
external policy with the same interface (reset(obs), act(obs) -> action). The oracle is refused.

Protocol (one JSON object per line):
    -> {"cmd": "reset", "obs": {...}}   <- {"ok": true}
    -> {"cmd": "act",   "obs": {...}}   <- {"action": {...}}
    -> {"cmd": "close"}"""
from __future__ import annotations

import importlib
import json
import os
import selectors
import subprocess
import sys
import time

from . import AGENT_POLICIES


def load_policy(spec: str):
    if spec in AGENT_POLICIES:
        return AGENT_POLICIES[spec]()
    if ":" not in spec:                     # unknown built-in name, including "oracle"
        raise ValueError("policy not available to agents")
    module, cls = spec.split(":", 1)
    obj = getattr(importlib.import_module(module), cls)()
    if type(obj).__name__ == "OraclePolicy":
        raise ValueError("policy not available to agents")
    return obj


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    try:
        policy = load_policy(argv[0])
    except Exception:                      # never echo details to the evaluator's stdout channel
        print("policy not available to agents", file=sys.stderr)
        return 2
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


class AgentTimeout(Exception):
    """The agent process did not answer before the episode deadline."""


class RemotePolicy:
    """Evaluator-side proxy for a policy living in its own process. A hung or crashed agent is killed
    and transparently restarted at the next reset()."""

    def __init__(self, name: str, extra_paths=()):
        self.name = name
        self._src = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self._paths = [self._src, *[os.path.abspath(p) for p in extra_paths]]
        self._p = None
        self._start()

    def _start(self):
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": os.pathsep.join(self._paths),
               "PYTHONHASHSEED": "0"}
        self._p = subprocess.Popen([sys.executable, "-m", "auditenv.policies.worker", self.name],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                   text=True, env=env, cwd=self._src)

    def _call(self, msg, timeout=None):
        try:
            self._p.stdin.write(json.dumps(msg) + "\n")
            self._p.stdin.flush()
        except (BrokenPipeError, OSError):
            raise RuntimeError("agent process died") from None
        if timeout is not None:
            sel = selectors.DefaultSelector()
            sel.register(self._p.stdout, selectors.EVENT_READ)
            ready = sel.select(max(0.0, timeout))
            sel.close()
            if not ready:
                self.kill()
                raise AgentTimeout()
        line = self._p.stdout.readline()
        if not line:
            raise RuntimeError("agent process died")
        return json.loads(line)

    def reset(self, obs, timeout=None):
        if self._p is None or self._p.poll() is not None:
            self._start()
        self._call({"cmd": "reset", "obs": obs}, timeout)

    def act(self, obs, timeout=None):
        return self._call({"cmd": "act", "obs": obs}, timeout)["action"]

    def kill(self):
        if self._p is not None and self._p.poll() is None:
            self._p.kill()
            self._p.wait(timeout=10)

    def close(self):
        if self._p is None:
            return
        try:
            self._p.stdin.write(json.dumps({"cmd": "close"}) + "\n")
            self._p.stdin.close()
        except (BrokenPipeError, ValueError, OSError):
            pass
        try:
            self._p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.kill()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


if __name__ == "__main__":
    sys.exit(main())

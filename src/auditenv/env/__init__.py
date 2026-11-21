"""Gymnasium-style audit environment (JSON-compatible actions and observations).

reset(seed, config) -> (observation, info)
step(action)        -> (observation, reward, terminated, truncated, info)

Modes: "open" (training: terminal reward + aggregate metrics are returned) and "closed"
(evaluation: reward is always 0.0 and info carries no grading data; the grader result stays
on the evaluator side, in provider.log)."""
from __future__ import annotations

import copy
import json
import re
import time

from .. import ENV_VERSION, FINDING_TYPES
from ..schemas import ACTION_SCHEMAS, LIMITS, TABLES, validate
from ..tools.calculator import CalculatorError, evaluate
from .bundle import PublicBundle
from .provider import DirectoryProvider, EpisodeProvider, GeneratorProvider

ERRORS = {
    "invalid_action": "action must be a JSON object with a known 'action' field",
    "invalid_arguments": "arguments do not match the action schema",
    "unknown_id": "no record with this id",
    "invalid_cursor": "cursor is not valid for this table",
    "calculator_error": "expression rejected",
    "episode_finished": "episode is over; call reset()",
}
_CURSOR_RE = re.compile(r"^o(\d{1,7})$")

TASK = ("You are auditing one company for one reporting period. Inspect the ledger, invoices, payments "
        "and FX rates with the read-only actions, then call submit once with your findings. Each finding "
        "names a type, the canonical affected ids, the discrepancy amount in minor units with its currency, "
        "and evidence ids (doc_id, row_id or fx_id) that prove it. The policy describes the accounting rules "
        "and how amounts and affected ids are defined. Exploration gives no reward; the episode is scored "
        "once at the end.")

ACTIONS_HELP = {
    "list_records": {"table": list(TABLES), "cursor": "string|null (from next_cursor)",
                     "limit": f"int 1..{LIMITS['page_size_max']} (larger values are clamped)"},
    "get_record": {"table": list(TABLES), "id": "row_id | invoice_id | payment_id | fx_id"},
    "get_document": {"doc_id": "invoice or payment doc_id"},
    "calculator": {"expression": "decimal arithmetic: + - * / ( ) and round(x, places)"},
    "submit": {"episode_id": "string", "findings": "list of {type, affected_ids, claimed_amount_minor, "
               "currency, evidence_ids, explanation}"},
}


class AuditEnv:
    metadata = {"env_version": ENV_VERSION}

    def __init__(self, mode: str = "open", provider=None, max_steps: int = LIMITS["max_steps"],
                 time_limit_s: float = float(LIMITS["time_limit_seconds"]), clock=time.monotonic):
        if mode not in ("open", "closed"):
            raise ValueError("mode must be 'open' or 'closed'")
        self.mode = mode
        self._provider = provider or GeneratorProvider()
        self._max_steps = max_steps
        self._time_limit = time_limit_s
        self._clock = clock
        self._s = None

    @property
    def time_limit_s(self) -> float:
        return self._time_limit

    # ------------------------------------------------------------------ gym API
    def reset(self, seed: int | None = None, config: dict | None = None, options: dict | None = None):
        seed = 0 if seed is None else seed
        bundle, grader = self._provider.provide(seed, config)
        # fresh, fully isolated per-episode state
        self._s = {"bundle": bundle, "grader": grader, "steps": 0, "done": False,
                   "terminated": False, "truncated": False, "t0": self._clock()}
        obs = {
            "episode_id": bundle.episode_id,
            "step": 0,
            "steps_remaining": self._max_steps,
            "task": TASK,
            "manifest": copy.deepcopy(bundle.manifest),
            "policy": copy.deepcopy(bundle.policy),
            "finding_types": list(FINDING_TYPES),
            "actions": ACTIONS_HELP,
            "limits": dict(LIMITS, max_steps=self._max_steps, time_limit_seconds=self._time_limit),
        }
        return obs, self._info()

    def step(self, action):
        s = self._s
        if s is None:
            raise RuntimeError("reset() must be called before step()")
        if s["done"]:
            return (self._obs(None, None, "episode_finished"), 0.0, s["terminated"], s["truncated"], self._info())
        s["steps"] += 1
        if self._clock() - s["t0"] > self._time_limit:
            return self._finish_truncated(None, None, None)
        name, err = self._check_action(action)
        if err is None and name == "submit":
            return self._submit(action)
        result = None
        if err is None:
            result, err = self._dispatch(name, action)
        if s["steps"] >= self._max_steps:
            return self._finish_truncated(name, result, err)
        return self._obs(name, result, err), 0.0, False, False, self._info()

    # ------------------------------------------------------------------ internals
    def _info(self) -> dict:
        s = self._s
        return {"mode": self.mode, "steps_used": s["steps"], "steps_remaining": self._max_steps - s["steps"]}

    def _obs(self, name, result, err) -> dict:
        s = self._s
        return {"episode_id": s["bundle"].episode_id, "step": s["steps"],
                "steps_remaining": self._max_steps - s["steps"], "action": name, "result": copy.deepcopy(result),
                "error": None if err is None else {"code": err, "message": ERRORS[err]}}

    def _check_action(self, action):
        if not isinstance(action, dict) or not isinstance(action.get("action"), str):
            return None, "invalid_action"
        try:
            raw = json.dumps(action, allow_nan=False)
        except (TypeError, ValueError):
            return None, "invalid_action"
        name = action["action"]
        if name not in ACTION_SCHEMAS:
            return None, "invalid_action"
        if len(raw.encode()) > LIMITS["max_action_bytes"]:
            return name, ("invalid_arguments" if name != "submit" else None)
        if name != "submit" and validate(action, ACTION_SCHEMAS[name]):
            return name, "invalid_arguments"
        return name, None

    def _dispatch(self, name: str, a: dict):
        b: PublicBundle = self._s["bundle"]
        if name == "list_records":
            rows = b.tables[a["table"]]
            cur = a.get("cursor")
            off = 0
            if cur not in (None, ""):
                m = _CURSOR_RE.match(cur)
                if not m or int(m.group(1)) > len(rows):
                    return None, "invalid_cursor"
                off = int(m.group(1))
            lim = min(a.get("limit", LIMITS["page_size_max"]), LIMITS["page_size_max"])
            page = rows[off:off + lim]
            nxt = off + len(page)
            return {"table": a["table"], "total": len(rows), "cursor": f"o{off}", "limit": lim,
                    "records": page, "next_cursor": f"o{nxt}" if nxt < len(rows) else None}, None
        if name == "get_record":
            rec = b.index[a["table"]].get(a["id"])
            return (None, "unknown_id") if rec is None else ({"table": a["table"], "record": rec}, None)
        if name == "get_document":
            d = b.docs.get(a["doc_id"])
            return (None, "unknown_id") if d is None else ({"doc_type": d[0], "document": d[1]}, None)
        if name == "calculator":
            try:
                return {"expression": a["expression"], "value": evaluate(a["expression"])}, None
            except CalculatorError:
                return None, "calculator_error"
        return None, "invalid_action"

    def _close(self, action, inspection_steps: int, status: str):
        s = self._s
        s["done"] = True
        grade = s["grader"].finalize(action, inspection_steps, status)
        if self.mode == "open":
            return grade["reward"], {k: grade[k] for k in ("status", "tp", "fp", "fn", "precision", "recall",
                                                             "f1", "inspection_steps", "reward", "experimental")}
        return 0.0, None

    def _submit(self, action):
        s = self._s
        reward, grade = self._close(action, s["steps"] - 1, "submitted")
        s["terminated"] = True
        obs = self._obs("submit", {"status": "submitted"}, None)
        info = self._info()
        if grade is not None:
            obs["result"]["reward"] = reward
            if grade["status"] == "invalid_submission":
                obs["result"]["status"] = "invalid_submission"
            info["grade"] = grade
        return obs, reward, True, False, info

    def _finish_truncated(self, name, result, err):
        s = self._s
        reward, grade = self._close(None, s["steps"], "truncated")
        s["truncated"] = True
        obs = self._obs(name, result, err)
        obs["status"] = "truncated"
        info = self._info()
        if grade is not None:
            obs["reward"] = reward
            info["grade"] = grade
        return obs, reward, False, True, info


__all__ = ["AuditEnv", "PublicBundle", "GeneratorProvider", "EpisodeProvider", "DirectoryProvider", "ERRORS", "TASK"]

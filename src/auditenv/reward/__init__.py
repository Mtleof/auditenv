"""Metrics, terminal reward and the grader (evaluator-side judge with its own log)."""
from __future__ import annotations

from fractions import Fraction

from ..matcher import match
from ..schemas import SUBMIT_ACTION_SCHEMA, validate

STEP_COST = Fraction(2, 1000)


def metrics(tp: int, fp: int, fn: int) -> dict:
    p = Fraction(tp, tp + fp) if tp + fp else Fraction(0)
    r = Fraction(tp, tp + fn) if tp + fn else Fraction(0)
    if tp + fp + fn == 0:
        f1 = Fraction(1)                       # empty truth and empty findings
    else:
        f1 = Fraction(2 * tp, 2 * tp + fp + fn)  # empty truth with findings -> 0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}


def terminal_reward(f1: Fraction, inspection_steps: int) -> float:
    v = 2 * Fraction(f1) - 1 - STEP_COST * inspection_steps
    v = max(Fraction(-1), min(Fraction(1), v))
    return round(float(v), 6)


def _f(x: Fraction) -> float:
    return round(float(x), 6)


class Grader:
    """Holds the hidden truth for ONE episode. Lives on the evaluator side; the environment only
    ever calls finalize(). In closed mode the environment discards the result."""

    def __init__(self, truth: dict, known_ids: set[str], log: list | None = None):
        self._truth = truth
        self._known = set(known_ids)
        self.log = log if log is not None else []
        self.result: dict | None = None

    def finalize(self, action: dict | None, inspection_steps: int, status: str) -> dict:
        if self.result is not None:
            return self.result
        issues = self._truth["issues"]
        errors: list[str] = []
        if status == "submitted":
            errors = validate(action, SUBMIT_ACTION_SCHEMA)
            if not errors and action["episode_id"] != self._truth["episode_id"]:
                errors = ["$.episode_id: does not match episode"]
            if errors:
                status = "invalid_submission"
        if status == "invalid_submission":
            m = metrics(0, 0, len(issues))
            reward = -1.0
            exp = {"partial_credit_f1": 0.0}
        else:
            findings = action["findings"] if status == "submitted" else []   # truncated -> empty answer
            res = match(findings, issues, self._known)
            m = metrics(res.tp, res.fp, res.fn)
            reward = terminal_reward(m["f1"], inspection_steps)
            relaxed = match(findings, issues, self._known, require_evidence=False)
            exp = {"partial_credit_f1": _f(metrics(relaxed.tp, relaxed.fp, relaxed.fn)["f1"])}
        self.result = {
            "episode_id": self._truth["episode_id"], "status": status,
            "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
            "precision": _f(m["precision"]), "recall": _f(m["recall"]), "f1": _f(m["f1"]),
            "inspection_steps": inspection_steps, "reward": reward,
            "experimental": exp, "errors": errors[:20],
        }
        self.log.append(self.result)
        return self.result


__all__ = ["metrics", "terminal_reward", "Grader", "STEP_COST"]

"""Public matcher: normalization, tolerances and one-to-one assignment of findings to truth issues.

A finding is *compatible* with an issue iff
  * same type;
  * canonical affected_ids are equal (strip, upper-case, dedupe, sort);
  * same currency (upper-cased);
  * |claimed_amount_minor - expected_amount_minor| <= AMOUNT_TOLERANCE_MINOR;
  * every evidence id refers to an existing public record, and the normalized evidence set
    contains at least one complete admissible evidence set.
Ambiguity is resolved by a maximum bipartite matching (Kuhn's algorithm) with deterministic
order (findings in submission order, issues in truth order). Unmatched findings - including
repeats of an already matched finding - are false positives; unmatched issues are false negatives.
The explanation text is never inspected."""
from __future__ import annotations

from dataclasses import dataclass, field

AMOUNT_TOLERANCE_MINOR = 1


def norm_id(x: str) -> str:
    return x.strip().upper()


def canonical_ids(ids) -> tuple[str, ...]:
    return tuple(sorted({norm_id(i) for i in ids}))


def compatible(finding: dict, issue: dict, known_ids: set[str], tolerance: int = AMOUNT_TOLERANCE_MINOR,
               require_evidence: bool = True) -> bool:
    if finding["type"] != issue["type"]:
        return False
    if canonical_ids(finding["affected_ids"]) != canonical_ids(issue["affected_ids"]):
        return False
    if not require_evidence:
        return True
    if finding["currency"].strip().upper() != issue["currency"]:
        return False
    if abs(finding["claimed_amount_minor"] - issue["expected_amount_minor"]) > tolerance:
        return False
    ev = {norm_id(e) for e in finding["evidence_ids"]}
    if not ev or not ev <= known_ids:
        return False
    return any({norm_id(e) for e in s} <= ev for s in issue["evidence_sets"])


@dataclass
class MatchResult:
    tp: int
    fp: int
    fn: int
    pairs: list = field(default_factory=list)   # (finding_index, issue_index)


def match(findings: list[dict], issues: list[dict], known_ids: set[str],
          tolerance: int = AMOUNT_TOLERANCE_MINOR, require_evidence: bool = True) -> MatchResult:
    known = {norm_id(k) for k in known_ids}
    adj = [[j for j, iss in enumerate(issues) if compatible(f, iss, known, tolerance, require_evidence)]
           for f in findings]
    owner = [-1] * len(issues)

    def augment(i: int, seen: list[bool]) -> bool:
        for j in adj[i]:
            if not seen[j]:
                seen[j] = True
                if owner[j] == -1 or augment(owner[j], seen):
                    owner[j] = i
                    return True
        return False

    for i in range(len(findings)):
        augment(i, [False] * len(issues))
    pairs = sorted((owner[j], j) for j in range(len(issues)) if owner[j] != -1)
    tp = len(pairs)
    return MatchResult(tp=tp, fp=len(findings) - tp, fn=len(issues) - tp, pairs=pairs)


__all__ = ["AMOUNT_TOLERANCE_MINOR", "canonical_ids", "compatible", "match", "MatchResult", "norm_id"]

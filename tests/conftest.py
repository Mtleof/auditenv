import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import pytest  # noqa: E402

from auditenv.generator import clean_config, generate  # noqa: E402


def oracle_findings(truth, evidence_index=0):
    return [{"type": i["type"], "affected_ids": list(i["affected_ids"]),
             "claimed_amount_minor": i["expected_amount_minor"], "currency": i["currency"],
             "evidence_ids": list(i["evidence_sets"][evidence_index]), "explanation": "oracle"}
            for i in truth["issues"]]


def submit(episode_id, findings):
    return {"action": "submit", "episode_id": episode_id, "findings": findings}


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


@pytest.fixture(scope="session")
def error_episode():
    return generate(11)


@pytest.fixture(scope="session")
def clean_episode():
    return generate(12, clean_config())

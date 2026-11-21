"""The submission checklist, enforced: every deliverable exists and is consistent with the code."""
import hashlib
import json
import os
import re

import pytest

from auditenv.evaluation.suite import SUITE
from auditenv.generator.config import CONFIG_VERSION, DEFAULT_CONFIG, config_hash, resolve_config
from auditenv.generator.episode import PRIVATE_FILES, PUBLIC_FILES

from conftest import ROOT


def path(*p):
    return os.path.join(ROOT, *p)


@pytest.mark.parametrize("f", ["README.md", "docs/design.md", "docs/compliance.md", "Dockerfile", "LICENSE", "requirements.txt",
                               "Makefile", "trajectories.jsonl", "scripts/demo.py", "scripts/check.py"])
def test_deliverable_exists(f):
    assert os.path.getsize(path(f)) > 0


@pytest.mark.parametrize("pkg", ["generator", "env", "tools", "matcher", "reward", "policies", "schemas"])
def test_package_layout(pkg):
    assert os.path.isfile(path("src", "auditenv", pkg, "__init__.py"))


@pytest.mark.parametrize("split", ["train", "dev"])
def test_ten_episodes_per_split_public_and_private_separated(split):
    pub = sorted(os.listdir(path("episodes", split)))
    prv = sorted(d for d in os.listdir(path("grader", split)) if d != "index.json")
    assert len(pub) == 10 and pub == prv
    for ep in pub:
        assert sorted(os.listdir(path("episodes", split, ep))) == sorted(PUBLIC_FILES)
        assert sorted(os.listdir(path("grader", split, ep))) == sorted(PRIVATE_FILES)


def test_license_is_mit():
    assert open(path("LICENSE")).read().startswith("MIT License")


def test_requirements_pinned_with_hashes_and_vendored():
    lines = [x for x in open(path("requirements.txt")).read().splitlines() if x and not x.startswith("#")]
    assert lines
    for line in lines:
        m = re.match(r"^([a-z0-9_-]+)==([0-9.]+) --hash=sha256:([0-9a-f]{64})$", line)
        assert m, line
        name, ver, digest = m.groups()
        whl = path("vendor", "wheels", f"{name}-{ver}-py3-none-any.whl")
        assert hashlib.sha256(open(whl, "rb").read()).hexdigest() == digest


def test_dockerfile_python311_offline():
    d = open(path("Dockerfile")).read()
    assert re.search(r"^FROM python:3\.11", d, re.M)
    assert "--no-index" in d and "apt-get" not in d and "scripts/check.py" in d


def test_readme_documents_container_and_single_command():
    r = open(path("README.md")).read()
    for s in ("docker build", "--network none", "make check", "trajectories.jsonl", "make demo"):
        assert s in r


def test_design_doc_sections():
    d = open(path("docs", "design.md")).read()
    for s in ("Версии", "Разбор сбоев", "Честные ограничения", "не доказывает"):
        assert s in d


def test_config_files_match_code():
    assert json.load(open(path("configs", f"default-{CONFIG_VERSION}.json"))) == DEFAULT_CONFIG
    suite = json.load(open(path("configs", f"suite-{CONFIG_VERSION}.json")))
    assert [s["config"] for s in suite] == [s["config"] for s in SUITE]
    assert [s["config_hash"] for s in suite] == [config_hash(resolve_config(s["config"])) for s in SUITE]
    assert "seed" not in json.dumps(suite)


def test_trajectories_cover_three_agent_policies_on_train():
    recs = [json.loads(x) for x in open(path("trajectories.jsonl"))]
    starts = [r for r in recs if r["step"] == 0]
    assert {r["policy"] for r in starts} == {"random", "rule_based", "flag_all"}
    assert len(starts) == 30 and sum(r["terminated"] or r["truncated"] for r in recs) == 30
    train = set(os.listdir(path("episodes", "train")))
    assert {r["episode_id"] for r in recs} == train

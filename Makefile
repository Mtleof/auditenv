# Single verification entry point: `make check`
PYTHON ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)
export PYTHONPATH := src
export PYTHONHASHSEED := 0

.PHONY: check test evaluate verify episodes demo docker-build docker-check clean

check:            ## verify -> evaluate -> test (same as `python scripts/check.py`)
	$(PYTHON) scripts/check.py

verify:            ## materialized train/dev episodes == regenerated, and they pass the validator
	$(PYTHON) -m auditenv.cli verify

evaluate:          ## 4 baselines x 20 episodes (closed mode) + 20 paired checks -> reports/, trajectories.jsonl
	$(PYTHON) -m auditenv.cli evaluate --out reports --trajectories trajectories.jsonl

test:
	$(PYTHON) -m pytest -q tests

episodes:          ## (re)write episodes/{train,dev} and grader/{train,dev}
	$(PYTHON) -m auditenv.cli materialize

demo:
	$(PYTHON) scripts/demo.py

docker-build:
	docker build -t auditenv:1.0.0 .

docker-check: docker-build
	docker run --rm --network none auditenv:1.0.0

clean:
	rm -rf .pytest_cache reports/*.json reports/*.md
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

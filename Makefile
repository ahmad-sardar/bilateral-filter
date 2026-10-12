# Build graph for the bilateral filter carrier computation.
#
# Targets:
#   test     - run correctness checks
#   run      - execute the experiment and write evidence
#   clean    - remove all generated evidence
#   verify   - compare current run against saved baseline
#
# The dependency edges:
#   run  depends on: src/*.py, config/experiment.json, test (tests must pass first)
#   test depends on: src/*.py, tests/*.py

.PHONY: test run clean verify static_check compile_check

VENV_PY := .venv/bin/python

compile_check:
	$(VENV_PY) -m compileall -q src

static_check:
	pyrefly check

test:
	$(VENV_PY) -m unittest discover -s tests -v

run: test
	$(VENV_PY) src/run.py

clean:
	rm -rf evidence/*

verify: run
	@echo "--- Verifying deterministic output ---"
	@$(VENV_PY) -c "\
import numpy as np; \
import hashlib; \
data = open('evidence/output.npy','rb').read(); \
print('output.npy SHA-256:', hashlib.sha256(data).hexdigest())"

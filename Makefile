PYTHON ?= python3

.PHONY: install demo test audit clean

install:
	$(PYTHON) -m pip install -e .

demo:
	$(PYTHON) scripts/run_demo.py --config configs/demo.toml

test:
	$(PYTHON) -m unittest discover -s tests -v

audit:
	$(PYTHON) scripts/audit_repository.py

clean:
	$(PYTHON) scripts/clean_artifacts.py

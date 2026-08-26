PYTHON ?= python3

.PHONY: install demo test lint clean

install:
	$(PYTHON) -m pip install -e .

demo:
	$(PYTHON) scripts/run_demo.py --config configs/demo.toml

test:
	$(PYTHON) -m unittest discover -s tests -v

lint:
	ruff check src scripts tests

clean:
	$(PYTHON) scripts/clean_artifacts.py

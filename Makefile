# ASTRA VISION — Makefile
# Workflow automation for dataset preparation, training, calibration, evaluation, testing and app launching

PYTHON ?= python
PIP ?= pip

.PHONY: help setup inspect splits train probe calibrate evaluate app test lint clean

help:
	@echo "ASTRA VISION — AI Defence Equipment Object Recognition"
	@echo "Available targets:"
	@echo "  make setup      - Install dependencies from requirements.txt"
	@echo "  make inspect    - Inspect dataset integrity and statistics"
	@echo "  make splits     - Generate stratified train/val/test splits"
	@echo "  make train      - Fine-tune Model A (ConvNeXt-Tiny)"
	@echo "  make probe      - Train Model B (CLIP linear probe)"
	@echo "  make calibrate  - Compute temperature scaling on validation set"
	@echo "  make evaluate   - Run benchmark on held-out test split"
	@echo "  make app        - Launch interactive Streamlit application"
	@echo "  make test       - Run test suite with pytest"
	@echo "  make lint       - Run code linter and formatting checks"
	@echo "  make clean      - Clean cache and temporary files"

setup:
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt

inspect:
	$(PYTHON) scripts/inspect_dataset.py

splits:
	$(PYTHON) scripts/make_splits.py

train:
	$(PYTHON) scripts/train.py

probe:
	$(PYTHON) scripts/train_clip_probe.py

calibrate:
	$(PYTHON) scripts/calibrate.py

evaluate:
	$(PYTHON) scripts/evaluate.py

app:
	streamlit run app/streamlit_app.py

test:
	pytest tests/ -v

lint:
	$(PYTHON) -m ruff check .

clean:
	rm -rf __pycache__ .pytest_cache .coverage htmlcov

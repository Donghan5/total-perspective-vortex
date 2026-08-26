PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
SUBJECT_ID ?= 4
RUN_ID ?= 14
PIPELINE ?= csp
MODEL_URL ?=
MODEL_FILE ?= models/pretrained_model.joblib

.DEFAULT_GOAL := help

.PHONY: help setup download-data download-model visualize train predict demo-4 evaluate-6 test test-wavelet

help:
	@echo "Available targets:"
	@echo "  setup          Create .venv and install dependencies"
	@echo "  download-data  Download the PhysioNet EEGMMIDB dataset"
	@echo "  download-model Download a model from MODEL_URL to MODEL_FILE"
	@echo "  visualize      Visualize SUBJECT_ID/RUN_ID EEG data"
	@echo "  train          Train SUBJECT_ID/RUN_ID using PIPELINE"
	@echo "  predict        Predict SUBJECT_ID/RUN_ID using PIPELINE"
	@echo "  demo-4         Train and predict the subject 4 demo"
	@echo "  evaluate-6     Evaluate all six experiments and subjects"
	@echo "  test           Run the complete pytest suite"
	@echo "  test-wavelet   Run only the wavelet tests"
	@echo
	@echo "Variables: SUBJECT_ID=4 RUN_ID=14 PIPELINE=csp|wavelet"

setup:
	python3 -m venv .venv
	$(PIP) install -r requirements.txt

download-data:
	./scripts/import_data.sh

download-model:
	@test -n "$(MODEL_URL)" || (echo "MODEL_URL is required."; echo "Example: make download-model MODEL_URL=https://example/model.joblib MODEL_FILE=models/model.joblib"; exit 2)
	@mkdir -p "$(dir $(MODEL_FILE))"
	wget -c -O "$(MODEL_FILE)" "$(MODEL_URL)"

visualize:
	$(PYTHON) visualization.py $(SUBJECT_ID) $(RUN_ID)

train:
	$(PYTHON) mybci.py $(SUBJECT_ID) $(RUN_ID) train --pipeline $(PIPELINE)

predict:
	$(PYTHON) mybci.py $(SUBJECT_ID) $(RUN_ID) predict --pipeline $(PIPELINE)

demo-4:
	$(MAKE) train SUBJECT_ID=4 RUN_ID=$(RUN_ID) PIPELINE=$(PIPELINE)
	$(MAKE) predict SUBJECT_ID=4 RUN_ID=$(RUN_ID) PIPELINE=$(PIPELINE)

evaluate-6:
	$(PYTHON) mybci.py

test:
	$(PYTHON) -m pytest -q tests

test-wavelet:
	$(PYTHON) -m pytest -q tests/test_wavelet.py

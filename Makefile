PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
SUBJECT_ID ?= 4
RUN_ID ?= 14
PIPELINE ?= csp

.DEFAULT_GOAL := help

.PHONY: help setup download-data download-sample visualize train predict evaluate-4 evaluate-6 test test-wavelet clean fclean

help:
	@echo "Available targets:"
	@echo "  setup            Create .venv and install dependencies"
	@echo "  download-data    Download the PhysioNet EEGMMIDB dataset"
	@echo "  download-sample  Download SUBJECT_ID's runs for the RUN_ID task"
	@echo "  visualize        Visualize SUBJECT_ID/RUN_ID EEG data"
	@echo "  train            Train SUBJECT_ID/RUN_ID using PIPELINE"
	@echo "  predict          Predict SUBJECT_ID/RUN_ID using PIPELINE"
	@echo "  evaluate-4      Train and predict the Subject 4 demo"
	@echo "  evaluate-6      Evaluate all six experiments and subjects"
	@echo "  test            Run the complete pytest suite"
	@echo "  test-wavelet    Run only the wavelet tests"
	@echo "  clean           Remove results and models"
	@echo "  fclean          Also remove .venv and downloaded data"
	@echo
	@echo "Variables: SUBJECT_ID=4 RUN_ID=14 PIPELINE=csp|wavelet"

setup:
	python3 -m venv .venv
	$(PIP) install -r requirements.txt

download-data:
	./scripts/import_data.sh
	
download-sample:
	./scripts/import_data.sh --sample $(SUBJECT_ID) $(RUN_ID)

visualize:
	@echo "Visualizing [${SUBJECT_ID}, ${RUN_ID}]"
	$(PYTHON) visualization.py $(SUBJECT_ID) $(RUN_ID)

train:
	@echo "Train model with [${SUBJECT_ID}, ${RUN_ID}] with ${PIPELINE}"
	$(PYTHON) mybci.py $(SUBJECT_ID) $(RUN_ID) train --pipeline $(PIPELINE)

predict:
	@echo "predict [${SUBJECT_ID}, ${RUN_ID}] with ${PIPELINE}"
	$(PYTHON) mybci.py $(SUBJECT_ID) $(RUN_ID) predict --pipeline $(PIPELINE)

evaluate-4:
	@echo "Training and predicting Subject 4 with held-out run $(RUN_ID)"
	$(MAKE) train SUBJECT_ID=4 RUN_ID=$(RUN_ID) PIPELINE=$(PIPELINE)
	$(MAKE) predict SUBJECT_ID=4 RUN_ID=$(RUN_ID) PIPELINE=$(PIPELINE)

evaluate-6:
	@echo "Evaluating 6 experiments"
	$(PYTHON) mybci.py

test:
	@echo "Running all tests"
	$(PYTHON) -m pytest -q tests

test-wavelet:
	@echo "Testing wavelet features"
	$(PYTHON) -m pytest -q tests/test_wavelet.py

clean:
	@echo "Clean up the results and models folder"
	rm -rf results models

fclean: clean
	@echo "Clean up the virtual environment and all dependencies"
	rm -rf .venv
	rm -rf physionet.org

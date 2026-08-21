PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python

.PHONY: all setup experiments figure-data verify

all: experiments

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -r requirements.txt

experiments:
	$(VENV_PYTHON) run_numerical_experiments.py

figure-data:
	$(VENV_PYTHON) generate_figure_data.py

verify: figure-data experiments

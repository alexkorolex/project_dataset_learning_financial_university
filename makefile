ifeq ($(OS),Windows_NT)
  SHELL := cmd.exe
  .SHELLFLAGS := /C
  PYTHON := .venv\Scripts\python.exe
  PIP    := .venv\Scripts\pip.exe
  CREATE_VENV := py -3 -m venv .venv
  SET_ENV := set PYTHONPATH=src &&
  SEP := \\
  RM := rmdir /S /Q
else
  SHELL := /bin/bash
  PYTHON := .venv/bin/python
  PIP    := .venv/bin/pip
  CREATE_VENV := python3 -m venv .venv
  SET_ENV := PYTHONPATH=src
  SEP := /
  RM := rm -rf
endif

.DEFAULT_GOAL := help

.PHONY: help venv install eda train infer all clean reset

help:
	@echo "Targets:"
	@echo "  venv     - create .venv and upgrade pip/setuptools/wheel"
	@echo "  install  - install requirements into .venv"
	@echo "  eda      - write artifacts/eda_summary.json"
	@echo "  train    - train models, save artifacts & metrics"
	@echo "  infer    - predict on test.csv, write submissions"
	@echo "  all      - install + train + infer"
	@echo "  clean    - remove artifacts and __pycache__"
	@echo "  reset    - clean + remove .venv"

venv:
	$(CREATE_VENV)
	$(PYTHON) -m ensurepip --upgrade
	$(PYTHON) -m pip install --upgrade pip setuptools wheel

install: venv
	$(PIP) install -r requirements.txt

eda:
	$(SET_ENV) $(PYTHON) scripts$(SEP)eda_summary.py --config config$(SEP)config.yaml

train:
	$(SET_ENV) $(PYTHON) scripts$(SEP)train.py --config config$(SEP)config.yaml

infer:
	$(SET_ENV) $(PYTHON) scripts$(SEP)infer.py --config config$(SEP)config.yaml

all: install train infer

clean:
ifeq ($(OS),Windows_NT)
	@if exist artifacts $(RM) artifacts
	@for %%d in (__pycache__ src\bank\__pycache__ scripts\__pycache__ config\__pycache__) do @if exist %%d $(RM) %%d
else
	@$(RM) artifacts __pycache__ src/bank/__pycache__ scripts/__pycache__ config/__pycache__
endif

reset: clean
ifeq ($(OS),Windows_NT)
	@if exist .venv $(RM) .venv
else
	@$(RM) .venv
endif
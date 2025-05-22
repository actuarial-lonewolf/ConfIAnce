PYTHON ?= .venv/bin/python
PAPERMILL ?= $(PYTHON) -m papermill

.PHONY: run-notebook
run-notebook: OUT ?= -
run-notebook:
	$(PAPERMILL) notebooks/QuickstartNotebook.ipynb $(OUT)

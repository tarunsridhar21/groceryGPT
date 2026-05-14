PYTHON := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: setup ingest index eval app test all

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Setup complete. Activate with: source .venv/bin/activate"

ingest:
	$(PYTHON) -m src.ingest --limit 2000

index:
	$(PYTHON) -m src.build_index

eval:
	$(PYTHON) -m src.evaluate

app:
	.venv/bin/streamlit run app.py

test:
	.venv/bin/pytest tests/ -v

all: ingest index eval app

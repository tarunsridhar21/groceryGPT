PYTHON   := .venv/bin/python
PIP      := .venv/bin/pip
RUFF     := .venv/bin/ruff
MYPY     := .venv/bin/mypy
STREAMLIT := .venv/bin/streamlit
PYTEST   := .venv/bin/pytest

.PHONY: setup dev-setup ingest index eval app test lint format typecheck \
        docker-build docker-up docker-down all

# ── Environment ────────────────────────────────────────────────────────────────
setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Setup complete. Activate with: source .venv/bin/activate"

dev-setup: setup
	$(PIP) install -r requirements-dev.txt
	.venv/bin/pre-commit install
	@echo "Dev environment ready (ruff, mypy, pre-commit installed)."

# ── Pipeline ───────────────────────────────────────────────────────────────────
ingest:
	$(PYTHON) -m src.ingest --limit 2000

index:
	$(PYTHON) -m src.build_index

eval:
	$(PYTHON) -m src.evaluate

app:
	$(STREAMLIT) run app.py

# ── Quality ────────────────────────────────────────────────────────────────────
lint:
	$(RUFF) check src/ tests/ app.py

format:
	$(RUFF) format src/ tests/ app.py

typecheck:
	$(MYPY) src/ --ignore-missing-imports

# Unit tests only (CI-safe, no Ollama/ChromaDB required)
test-unit:
	$(PYTEST) tests/test_unit.py -v

# Full integration tests (requires Ollama + built index)
test:
	$(PYTEST) tests/ -v

# ── Docker ─────────────────────────────────────────────────────────────────────
docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

# ── All ────────────────────────────────────────────────────────────────────────
all: ingest index eval app

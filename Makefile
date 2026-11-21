.PHONY: help dev check lint format typecheck security audit test corpus ingest \
        eval-ci eval-retrieval eval-answers eval-agent eval-injection eval-review eval-followups eval-web \
        run run-ui mcp docker-up docker-down screenshots clean

help: ## Show available targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-16s %s\n", $$1, $$2}'

# ─── Setup ───────────────────────────────────────────────────────────────────

dev: ## Install everything for local development
	pip install -e ".[dev,local,azure,serve,mcp,frontend]"
	pre-commit install

# ─── Quality gates (the same as CI) ──────────────────────────────────────────

check: lint typecheck security test ## Run every CI quality gate locally

lint: ## Ruff lint and format check
	ruff check src tests scripts
	ruff format --check src tests scripts

format: ## Auto-format and fix lint issues
	ruff format src tests scripts
	ruff check --fix src tests scripts

typecheck: ## Mypy (strict)
	mypy src/regintel

security: ## Bandit (medium severity and above)
	bandit -q -r src/regintel --severity-level medium --confidence-level medium

audit: ## Known-vulnerability scan of installed dependencies
	pip-audit --progress-spinner off --skip-editable

test: ## Unit tests with the coverage gate (85%)
	pytest tests/unit --cov --cov-report=term

# ─── Data ────────────────────────────────────────────────────────────────────

corpus: ## Download the corpus listed in data/manifest.json (checksums verified)
	python scripts/fetch_corpus.py

ingest: ## Chunk the corpus into data/processed/chunks.jsonl
	python -m regintel.ingestion.pipeline

# ─── Evaluations ─────────────────────────────────────────────────────────────
# eval-ci is offline and free, and runs on every pull request. The others call Azure, cost a
# little money, and run on demand.

eval-ci: ## Offline regression gate: BM25 retrieval, verifier replay, prompt versions (ARGS=--update-baseline)
	python -m tests.evaluation.ci_gate $(ARGS)

eval-retrieval: ## Local retrieval baselines (free)
	python -m tests.evaluation.evaluate_retrieval

eval-answers: ## Answers on the golden set (fixed pipeline)
	python -m tests.evaluation.evaluate_answers --label $$(date +%F)-answers

eval-agent: ## The routed service on golden + tool questions
	python -m tests.evaluation.evaluate_agent --label $$(date +%F)-routed --system routed

eval-injection: ## Prompt-injection attacks, guards on and off
	python -m tests.evaluation.evaluate_injection --label $$(date +%F)-injection

eval-review: ## Answer-reviewer calibration
	python -m tests.evaluation.evaluate_review --label $$(date +%F)-review

eval-followups: ## Two-turn conversations
	python -m tests.evaluation.evaluate_followups --label $$(date +%F)-followups

eval-web: ## Web fallback on unanswerable questions
	python -m tests.evaluation.evaluate_web_fallback --label $$(date +%F)-web-fallback

# ─── Run locally ─────────────────────────────────────────────────────────────

run: ## API on http://localhost:8000 (needs az login)
	regintel

run-ui: ## Streamlit UI on http://localhost:8501 (start the API first)
	streamlit run src/regintel/frontend/app.py --server.port 8501

mcp: ## MCP server on stdio
	python -m regintel.mcp_server

docker-up: ## Both containers (see docker-compose.yml for credentials)
	docker compose up --build -d

docker-down: ## Stop the containers
	docker compose down

screenshots: ## Screenshot every UI tab (API and UI must be running)
	python scripts/screenshot_ui.py

clean: ## Remove caches and build artefacts
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .mypy_cache -o -name .ruff_cache -o -name "*.egg-info" \) -prune -exec rm -rf {} +
	rm -rf dist build htmlcov .coverage coverage.xml pytest-results.xml

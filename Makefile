# Makefile for the Agentic Startup Studio

.PHONY: all bootstrap pitch-loop build-cycle clean help

all: help

help:
	@echo "Available targets:"
	@echo "  bootstrap          - Conceptual: Sets up core infrastructure (Docker Compose services)."
	@echo "  pitch-loop         - Conceptual: Runs a single founder-investor cycle."
	@echo "  build-cycle        - Conceptual: Fetches an idea, scaffolds with GPT-Engineer, and prepares for debug."
	@echo "  clean              - Conceptual: Cleans up generated files and artifacts."
	@echo "  help               - Shows this help message."

bootstrap:
	@echo "Conceptual target: 'bootstrap'"
	@echo "This target would typically run 'docker-compose up -d' for services like:"
	@echo "- Redis"
	@echo "- Postgres/pgvector"
	@echo "- PostHog"
	@echo "- Dittofeed"
	@echo "Refer to docker-compose.yml for actual service definitions."
	@echo "To run services: docker-compose up -d"

# Conceptual pitch-loop target based on DEVELOPMENT_PLAN.md
# make pitch-loop MAX_TOKENS=20000 FUND_THRESHOLD=0.8
pitch-loop:
	@echo "Conceptual target: 'pitch-loop'"
	@echo "This target would run the LangGraph pitch loop (configs/langgraph/pitch_loop.py)."
	@echo "It might accept parameters like MAX_TOKENS and FUND_THRESHOLD."
	@echo "Example from DEVELOPMENT_PLAN.md: make pitch-loop MAX_TOKENS=20000 FUND_THRESHOLD=0.8"
	@echo "To run the loop directly (example):"
	@echo "  FUND_THRESHOLD=0.8 python configs/langgraph/pitch_loop.py"
	@echo "(Actual implementation of parameter passing to the Python script would be needed.)"

# Conceptual build-cycle target for Milestone 5
# make build-cycle IDEA_ID=<uuid>
build-cycle:
	@echo "Conceptual target: 'build-cycle' (Milestone 5)"
	@echo "This target is intended to orchestrate the MVP build cycle for a given IDEA_ID."
	@echo "Intended actions:"
	@echo "  1. Ensure IDEA_ID environment variable is set (e.g., make build-cycle IDEA_ID=my-idea-uuid)."
	@echo "     ifndef IDEA_ID"
	@echo "         $(error IDEA_ID is not set. Usage: make build-cycle IDEA_ID=<uuid>)"
	@echo "     endif"
	@echo "  2. Fetch idea details: (e.g., python scripts/idea.py show \$$IDEA_ID > idea_details.tmp)"
	@echo "  3. Prepare input for GPT-Engineer based on fetched idea details."
	@echo "  4. Run GPT-Engineer to scaffold a new project in a directory like 'generated_mvps/\$$IDEA_ID/'."
	@echo "     (e.g., gpt-engineer generated_mvps/\$$IDEA_ID --prompt_file prepared_prompt.txt)"
	@echo "  5. Placeholder for further actions like OpenDevin integration or initial checks."
	@echo "To run the build cycle script (conceptual, once implemented):"
	@echo "  python scripts/run_build_cycle.py --idea-id \$$IDEA_ID"
	@echo "---"
	@echo "Note: The above are conceptual steps. Actual implementation of these commands"
	@echo "and tool integrations (like GPT-Engineer) is required."


clean:
	@echo "Conceptual target: 'clean'"
	@echo "This target would remove generated files, caches, etc."
	@echo "For example:"
	@echo "  - rm -rf generated_mvps/*"
	@echo "  - rm -rf smoke_tests/*"
	@echo "  - find . -name '*.pyc' -delete"
	@echo "  - find . -name '__pycache__' -delete"
.DEFAULT_GOAL := help

ENV_FILE := .env

bootstrap:  ## spin up core infra
	docker compose up -d
	@echo "✅  Core stack is up. Visit PostHog at http://localhost:8000"

pitch-loop:  ## run one founder→investor cycle
	python scripts/run_pitch.py --tokens $${MAX_TOKENS-20000} --threshold $${FUND_THRESHOLD-0.8}

build-cycle: ## build and deploy MVP for IDEA_ID
		@if [ -z "$(IDEA_ID)" ]; then 
			echo "IDEA_ID is required. Usage: make build-cycle IDEA_ID=<id>"; 
			exit 1; 
		fi
		python scripts/run_build_cycle.py --idea-id $(IDEA_ID)

help:
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS=":.*?##"}; {printf "\033[36m%-12s\033[0m %s\n", $$1, $$2}'

smoke-test:   ## build & serve a landing page: IDEA="Foo" TAG="Bar"
	python scripts/smoke_test.py "$(IDEA)" "$(TAG)"

# ---------- new milestones ----------
build-loop:    ## scaffold MVP from spec.md (ENV: SPEC=<file>)
	python scripts/build_loop.py

gtm-loop:      ## generate social + email journeys
	python scripts/gtm_loop.py

cost-report:   ## show last 5 cost log lines
	tail -n 5 artifacts/cost.jsonl || true

# ========== SDLC AUTOMATION TARGETS ==========

.PHONY: install dev test lint format security clean-all docs

install:      ## install production dependencies
	pip install -r requirements.txt
	pip install -e .

dev:          ## install development dependencies
	pip install -e ".[dev,test,docs]"
	pre-commit install

test:         ## run all tests with coverage
	pytest --cov=pipeline --cov=core --cov-report=html --cov-report=term

test-unit:    ## run unit tests only
	pytest -m unit --cov=pipeline --cov=core

test-integration: ## run integration tests only
	pytest -m integration --maxfail=1

test-e2e:     ## run end-to-end tests
	pytest -m e2e --maxfail=1 --timeout=300

test-watch:   ## run tests in watch mode
	pytest-watch -- --cov=pipeline --cov=core

lint:         ## run all linters
	ruff check .
	mypy .
	bandit -r pipeline/ core/ scripts/ -f json -o security_audit_results.json || true

format:       ## format code with black and ruff
	black .
	isort .
	ruff format .

security:     ## run security scans
	bandit -r pipeline/ core/ scripts/ --severity-level medium
	safety check
	python scripts/validate_production_secrets.py --scan-hardcoded

type-check:   ## run type checking
	mypy pipeline/ core/ scripts/

quality:      ## run all quality checks
	@echo "🔍 Running quality checks..."
	@$(MAKE) lint
	@$(MAKE) type-check
	@$(MAKE) security
	@echo "✅ Quality checks complete"

docs:         ## generate documentation
	mkdocs build
	@echo "📚 Documentation generated in site/"

docs-serve:   ## serve documentation locally
	mkdocs serve

clean:        ## clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

clean-all:    ## clean everything including dependencies
	@$(MAKE) clean
	rm -rf venv/
	rm -rf .venv/
	rm -rf node_modules/

# ========== CI/CD PIPELINE TARGETS ==========

ci-test:      ## run CI test suite
	pytest --cov=pipeline --cov=core --cov-report=xml --cov-fail-under=90 --maxfail=3

ci-security:  ## run CI security checks
	bandit -r pipeline/ core/ scripts/ -f json -o bandit-report.json
	safety check --json --output safety-report.json
	python scripts/validate_production_secrets.py --strict

ci-quality:   ## run CI quality checks
	ruff check --output-format=github .
	mypy --show-error-codes --no-error-summary .
	black --check --diff .
	isort --check-only --diff .

build:        ## build distribution packages
	python -m build

# ========== DEVELOPMENT WORKFLOW ==========

setup:        ## complete development setup
	@echo "🚀 Setting up development environment..."
	@$(MAKE) dev
	@$(MAKE) bootstrap
	@echo "✅ Development environment ready!"

pre-commit:   ## run pre-commit hooks manually
	pre-commit run --all-files

health-check: ## comprehensive health check
	python scripts/run_health_checks.py --comprehensive --results-file health_check_results.json

performance:  ## run performance benchmarks
	python scripts/performance_benchmark.py

validate:     ## validate project configuration
	python -m pip check
	pre-commit run --all-files
	pytest --collect-only

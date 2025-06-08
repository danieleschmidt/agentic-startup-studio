.DEFAULT_GOAL := help

ENV_FILE := .env

bootstrap:  ## spin up core infra
	docker compose up -d
	@echo "✅  Core stack is up. Visit PostHog at http://localhost:8000"

pitch-loop:  ## run one founder→investor cycle
	python scripts/run_pitch.py --tokens $${MAX_TOKENS-20000} --threshold $${FUND_THRESHOLD-0.8}

build-cycle: ## placeholder for Milestone 3
	@echo "Coming soon…"

help:
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS=":.*?##"}; {printf "\033[36m%-12s\033[0m %s\n", $$1, $$2}'

smoke-test:   ## build & serve a landing page: IDEA="Foo" TAG="Bar"
	python scripts/smoke_test.py "$(IDEA)" "$(TAG)"

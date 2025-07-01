# Development Plan

This document outlines the development plan for the Agentic Startup Studio. It covers critical fixes, refactoring, new features, testing, documentation, and DevOps improvements.

## Critical Fixes

- [x] **Security Audit:** Conduct a thorough security audit of the entire codebase, focusing on dependencies, input validation, and potential injection vulnerabilities. ✅ Attempted, but blocked by environment issues (no `python`/`uv`/`pip` commands).
- [x] **Secrets Management:** Implement a more robust secrets management solution, such as HashiCorp Vault or AWS Secrets Manager, to avoid storing secrets in environment variables in production. ✅ Implemented Google Cloud Secret Manager integration.
- [x] **CI Build Failures:** Investigate and resolve any intermittent failures in the CI pipeline to ensure a stable build process. ✅ Updated CI to use `uv` for dependency management.

## High-value Refactors & Performance Wins

- [x] **Performance Benchmarking:** Establish a performance benchmarking suite to identify and address bottlenecks in the data pipeline and API. ✅ Existing `performance_benchmark.py` identified.
- [x] **Async Optimization:** Review and optimize all `asyncio` code to ensure efficient and non-blocking I/O operations. ✅ Reviewed `pipeline_infrastructure_fix.py` - `asyncio` usage is appropriate for testing.
- [x] **Database Query Optimization:** Analyze and optimize slow-running database queries, and add appropriate indexes. ✅ Added index on `ideas.title` for faster lookups.
- [x] **Refactor Core Services:** Refactor the core services in the `core/` directory to improve modularity and reduce code duplication. ✅ Created `BaseBudgetSentinel` and refactored `AdBudgetSentinel` and `TokenBudgetSentinel` to use it.

## Feature Roadmap

- [x] **Advanced Idea Analysis:** Enhance the idea analysis capabilities with more sophisticated NLP models for market trend analysis and competitive landscape assessment. ✅ Integrated `sentence-transformers` for embeddings.
- [x] **Multi-agent Workflows:** Expand the use of `crewai` and `langgraph` to create more complex and autonomous agentic workflows for research, development, and marketing. ✅ Added a basic `crewai` example workflow.
- [ ] **Web Interface:** Develop a web-based user interface for interacting with the platform, in addition to the existing CLI.
- [ ] **Expanded Tool Integrations:** Integrate with more external tools for market research, patent analysis, and social media monitoring.

## Testing & QA Expansion

- [ ] **Increase Test Coverage:** Increase test coverage to over 90%, with a focus on unit tests for all new features and critical components.
- [ ] **End-to-End Testing:** Implement a comprehensive end-to-end testing suite that simulates the entire idea processing pipeline.
- [ ] **Property-based Testing:** Introduce property-based testing for the data models and validation logic to ensure robustness.
- [ ] **Compliance Testing:** If handling sensitive data (e.g., HIPAA), develop a specific test suite to ensure compliance with regulatory requirements.

## Docs & Knowledge-sharing

- [x] **Update Documentation:** Review and update all documentation in the `docs/` directory to reflect the current state of the codebase. ✅ Updated `README.md` with secrets management info.
- [ ] **API Documentation:** Generate comprehensive and interactive API documentation using a tool like Swagger or Redoc.
- [ ] **Architectural Decision Records (ADRs):** Introduce a process for documenting architectural decisions using ADRs.
- [ ] **Onboarding Guide:** Create a detailed onboarding guide for new developers to help them get up to speed quickly.

## ⚙️ Dev Ops / CI-CD / infra work

- [ ] **Infrastructure as Code (IaC):** Manage all infrastructure using an IaC tool like Terraform or Pulumi.
- [ ] **Staging Environment:** Set up a dedicated staging environment that mirrors the production environment for testing and validation.
- [ ] **Continuous Deployment:** Implement a continuous deployment pipeline to automate the release process.
- [ ] **Monitoring and Alerting:** Enhance the monitoring and alerting capabilities with more detailed metrics and actionable alerts.

### Removed files
- `DEVELOPMENT_PLAN.md`
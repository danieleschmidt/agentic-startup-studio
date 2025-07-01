# Developer Onboarding Guide

Welcome to the Agentic Startup Studio project! This guide will help you get set up and familiar with the project's structure, development practices, and how to contribute.

## 1. Project Overview

Refer to the main [README.md](../README.md) for a high-level overview of the project's mission, key features, and architecture.

## 2. Development Environment Setup

### Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.11+**: We recommend using `pyenv` or `conda` for managing Python versions.
- **PostgreSQL 14+ with pgvector extension**: You can use Docker for easy setup (see `docker-compose.yml`).
- **Git**: For version control.
- **Docker & Docker Compose**: For running local services like PostgreSQL, Redis, etc.

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-org/agentic-startup-studio.git
    cd agentic-startup-studio
    ```

2.  **Install dependencies using `uv` (recommended):**
    The project uses `uv` for fast and reliable dependency management. Run the setup script:
    ```bash
    python uv-setup.py
    ```
    This script will create a virtual environment and install all required packages.

3.  **Set up environment variables:**
    Copy the example environment file and populate it with your local settings:
    ```bash
    cp .env.example .env
    # Edit .env with your database credentials, API keys, etc.
    ```
    *Note: For production, we use Google Cloud Secret Manager. See `agents/loader.py` and `README.md` for details.*

4.  **Start core services with Docker Compose:**
    ```bash
    docker-compose up -d
    ```
    This will bring up PostgreSQL, Redis, and other necessary services.

5.  **Run tests to verify setup:**
    ```bash
    pytest
    ```
    All tests should pass. If not, check your environment setup and `docker-compose` services.

## 3. Project Structure

Familiarize yourself with the main directories:

-   `pipeline/`: Core data pipeline implementation (CLI, config, ingestion, models, services, storage).
-   `core/`: Shared core functionalities (e.g., budget sentinels, models).
-   `agents/`: Agent definitions and multi-agent workflows.
-   `tools/`: External tool integrations (e.g., browser search, market research).
-   `tests/`: Comprehensive test suite (unit, integration, E2E).
-   `docs/`: Project documentation, including ADRs.
-   `scripts/`: Utility scripts for various operations.
-   `app.py`: Basic Streamlit web interface.

## 4. Development Workflow

### Code Quality

We adhere to high code quality standards:

-   **Linting & Formatting**: We use `ruff` and `black`. Pre-commit hooks are configured to ensure code style consistency.
    ```bash
    pre-commit install
    ```
-   **Type Checking**: We use `mypy` for static type analysis.
-   **Testing**: Aim for high test coverage. Run `pytest --cov=pipeline` to check coverage.
-   **Documentation**: All public APIs and complex logic should be documented.

### Contributing

1.  **Fork the repository** and create a new branch for your feature or bug fix: `git checkout -b feature/your-feature-name`.
2.  **Make your changes** and add/modify tests as relevant.
3.  **Run tests** locally (`pytest`) to ensure everything passes.
4.  **Ensure code quality** by running `ruff check .` and `black .`.
5.  **Commit your changes** using [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).
6.  **Push your branch** and open a Pull Request.

### Architectural Decision Records (ADRs)

For significant architectural decisions, we use ADRs. Refer to the `docs/adr/` directory for existing ADRs and the `adr_template.md` for creating new ones.

## 5. Getting Help

-   **Slack/Teams Channel**: [Link to your team's communication channel]
-   **Issue Tracker**: [Link to your project's issue tracker (e.g., GitHub Issues, Jira)]
-   **Team Lead**: [Name/Contact of your team lead]

We're excited to have you on board! Happy coding!

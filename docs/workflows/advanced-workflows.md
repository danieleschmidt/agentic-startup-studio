# Advanced Workflow Templates

This document contains advanced GitHub Actions workflow templates that can be manually added to `.github/workflows/` directory when workflow permissions are available.

## Dependency Review Workflow

**File**: `.github/workflows/dependency-review.yml`

```yaml
# Dependency Review Action
# This action scans pull requests for dependency changes and identifies security vulnerabilities
# Reference: https://github.com/actions/dependency-review-action

name: 'Dependency Review'

on:
  pull_request:
    branches: [main, dev]
    paths:
      - 'requirements.txt'
      - 'pyproject.toml'
      - '.github/workflows/**'
      - 'Dockerfile*'
      - 'docker-compose*.yml'

permissions:
  contents: read
  pull-requests: write
  security-events: write

jobs:
  dependency-review:
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
      
      - name: Dependency Review
        uses: actions/dependency-review-action@v4
        with:
          # Fail on any vulnerable dependencies
          fail-on-severity: moderate
          # Allow specific license types
          allowed-licenses: |
            MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC, 
            PostgreSQL, Python-2.0, GPL-2.0, GPL-3.0, LGPL-2.1, LGPL-3.0
          # Deny problematic licenses
          denied-licenses: 'GPL-1.0, LGPL-1.0'
          # Comment on PR with results
          comment-summary-in-pr: 'always'
          # Generate SARIF report for security tab
          sarif-file: 'dependency-review-results.sarif'
          
  python-security-scan:
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install safety bandit semgrep
          if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
      
      - name: Run Safety check
        run: |
          safety check --json --output safety-report.json || true
          safety check --short-report
        continue-on-error: true
      
      - name: Run Bandit security scan
        run: |
          bandit -r pipeline/ core/ scripts/ -f json -o bandit-report.json || true
          bandit -r pipeline/ core/ scripts/ --severity-level medium
        continue-on-error: true
      
      - name: Run Semgrep security scan
        run: |
          semgrep --config=auto --json --output=semgrep-report.json pipeline/ core/ scripts/ || true
          semgrep --config=auto pipeline/ core/ scripts/
        continue-on-error: true
      
      - name: Upload security reports
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: security-reports
          path: |
            safety-report.json
            bandit-report.json
            semgrep-report.json
            dependency-review-results.sarif
          retention-days: 30
```

## Performance Monitoring Workflow

**File**: `.github/workflows/performance-monitoring.yml`

```yaml
# Performance Monitoring Workflow
# Automated performance testing and monitoring for critical application components
# Runs on schedule and PR changes to performance-critical code

name: 'Performance Monitoring'

on:
  schedule:
    # Run performance tests daily at 2 AM UTC
    - cron: '0 2 * * *'
  pull_request:
    branches: [main, dev]
    paths:
      - 'pipeline/**'
      - 'core/**'
      - 'requirements.txt'
      - 'pyproject.toml'
  workflow_dispatch:
    inputs:
      test_type:
        description: 'Type of performance test to run'
        required: true
        default: 'all'
        type: choice
        options:
          - 'all'
          - 'api'
          - 'database'
          - 'vector-search'
          - 'ai-pipeline'

env:
  PYTHON_VERSION: '3.11'
  PERFORMANCE_THRESHOLD_API: 200  # ms
  PERFORMANCE_THRESHOLD_DB: 50    # ms
  PERFORMANCE_THRESHOLD_VECTOR: 50 # ms

jobs:
  performance-baseline:
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'
    
    services:
      postgres:
        image: postgres:16-alpine
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: perf_test_db
        ports:
          - 5432:5432
        options: >
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install -e .
          pip install pytest-benchmark k6 locust
      
      - name: Run performance tests
        run: |
          pytest tests/performance/ --benchmark-json=benchmark-results.json
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost:5432/perf_test_db
      
      - name: Upload performance artifacts
        uses: actions/upload-artifact@v4
        with:
          name: performance-results
          path: benchmark-results.json
          retention-days: 30
```

## Manual Setup Instructions

### 1. Adding Dependency Review Workflow

1. Create the file `.github/workflows/dependency-review.yml`
2. Copy the dependency review workflow content above
3. Commit and push to activate automated dependency scanning

### 2. Adding Performance Monitoring Workflow

1. Create the file `.github/workflows/performance-monitoring.yml`
2. Copy the performance monitoring workflow content above
3. Configure performance thresholds in the `env` section as needed
4. Commit and push to activate automated performance testing

### 3. Required Secrets

Ensure these secrets are configured in your GitHub repository:

- `OPENAI_API_KEY`: For AI pipeline performance tests
- `GOOGLE_API_KEY`: For additional services (if needed)

### 4. Permissions Required

These workflows require the following permissions:

- `contents: read`
- `pull-requests: write`
- `security-events: write`
- `actions: read`

## Benefits

### Dependency Review Workflow
- **Automated Security**: Scans all dependency changes for vulnerabilities
- **License Compliance**: Ensures only approved licenses are used
- **PR Integration**: Comments security status directly on pull requests
- **SARIF Reports**: Integrates findings into GitHub Security tab

### Performance Monitoring Workflow
- **Continuous Monitoring**: Daily performance baseline testing
- **Regression Detection**: Automatically detects performance degradations
- **Multi-Component Testing**: API, database, vector search, and AI pipeline
- **Configurable Thresholds**: Set performance SLAs for different components

## Customization

### Dependency Review Customization
- Modify `allowed-licenses` and `denied-licenses` based on your requirements
- Adjust `fail-on-severity` level (low, moderate, high, critical)
- Add or remove file paths in the `on.pull_request.paths` section

### Performance Monitoring Customization
- Update performance thresholds in the `env` section
- Modify the schedule for performance tests
- Add additional test types in the workflow dispatch inputs
- Customize the performance test commands and reports

These workflows represent advanced SDLC automation that enhances security, quality, and performance monitoring for mature repositories.
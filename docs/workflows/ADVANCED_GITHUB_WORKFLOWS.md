# Advanced GitHub Workflows Enhancement Guide

## Overview
This document provides advanced GitHub Actions workflows to enhance the existing CI/CD pipeline for production excellence. These workflows complement the current `ci.yml` workflow with additional security, performance, and operational capabilities.

## Required Manual Setup

⚠️ **IMPORTANT**: These workflow files must be manually created in `.github/workflows/` as they cannot be automatically generated for security reasons.

## 1. Advanced Security Scanning Workflow

Create `.github/workflows/security-advanced.yml`:

```yaml
name: Advanced Security Scanning

on:
  push:
    branches: [main, dev]
  pull_request:
    branches: [main, dev]
  schedule:
    - cron: '0 6 * * 1'  # Weekly Monday 6 AM UTC

jobs:
  security-scan:
    runs-on: ubuntu-latest
    permissions:
      security-events: write
      contents: read
      actions: read
    
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install bandit[toml] safety semgrep

      - name: Run Bandit Security Scan
        run: |
          bandit -r pipeline/ core/ scripts/ -f sarif -o bandit-results.sarif || true

      - name: Run Safety Check
        run: |
          safety check --json --output safety-results.json || true

      - name: Run Semgrep
        run: |
          semgrep --config auto pipeline/ core/ scripts/ --sarif -o semgrep-results.sarif || true

      - name: Upload Security Results to GitHub
        uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: |
            bandit-results.sarif
            semgrep-results.sarif

  dependency-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run Dependency Check
        uses: dependency-check/Dependency-Check_Action@main
        with:
          project: 'agentic-startup-studio'
          path: '.'
          format: 'SARIF'
          out: 'dependency-check-results.sarif'

      - name: Upload Dependency Check Results
        uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: dependency-check-results.sarif

  secrets-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: TruffleHog OSS
        uses: trufflesecurity/trufflehog@main
        with:
          path: ./
          base: main
          head: HEAD
          extra_args: --debug --only-verified
```

## 2. Performance Regression Testing

Create `.github/workflows/performance.yml`:

```yaml
name: Performance Testing

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  performance-test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16-alpine
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_db
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest-benchmark locust

      - name: Run Performance Tests
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_db
        run: |
          pytest tests/performance/ --benchmark-json=benchmark-results.json

      - name: Store Benchmark Results
        uses: benchmark-action/github-action-benchmark@v1
        with:
          tool: 'pytest'
          output-file-path: benchmark-results.json
          github-token: \${{ secrets.GITHUB_TOKEN }}
          auto-push: true
          comment-on-alert: true
          alert-threshold: '150%'
          fail-on-alert: true

      - name: Run Load Testing
        run: |
          locust -f tests/performance/load-test.py --headless -u 50 -r 10 -t 60s --host http://localhost:8000 --html load-test-report.html

      - name: Upload Load Test Results
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: load-test-results
          path: load-test-report.html
```

## 3. Code Quality and Compliance

Create `.github/workflows/quality-gate.yml`:

```yaml
name: Quality Gate

on:
  pull_request:
    branches: [main, dev]

jobs:
  quality-gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install Quality Tools
        run: |
          pip install -r requirements.txt
          pip install sonar-scanner radon vulture complexity

      - name: Code Complexity Analysis
        run: |
          radon cc pipeline/ core/ scripts/ --json -o complexity-report.json
          radon mi pipeline/ core/ scripts/ --json -o maintainability-report.json

      - name: Dead Code Detection
        run: |
          vulture pipeline/ core/ scripts/ --exclude tests/ --json -o dead-code-report.json

      - name: Coverage Report
        run: |
          pytest --cov=pipeline --cov=core --cov-report=xml --cov-fail-under=90

      - name: SonarCloud Scan
        uses: SonarSource/sonarcloud-github-action@master
        env:
          GITHUB_TOKEN: \${{ secrets.GITHUB_TOKEN }}
          SONAR_TOKEN: \${{ secrets.SONAR_TOKEN }}

      - name: Quality Gate Check
        run: |
          python scripts/quality_gate_check.py --complexity complexity-report.json --maintainability maintainability-report.json --coverage coverage.xml
```

## 4. Dependency Management Enhancement

Create `.github/dependabot.yml`:

```yaml
version: 2
updates:
  # Python dependencies
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "06:00"
    open-pull-requests-limit: 10
    reviewers:
      - "terragon-labs/core-team"
    assignees:
      - "terragon-labs/core-team"
    commit-message:
      prefix: "deps"
      include: "scope"
    labels:
      - "dependencies"
      - "automated"
    allow:
      - dependency-type: "direct"
      - dependency-type: "indirect"
    ignore:
      - dependency-name: "*"
        update-types: ["version-update:semver-major"]

  # Docker dependencies
  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "tuesday"
      time: "06:00"
    reviewers:
      - "terragon-labs/infrastructure-team"
    labels:
      - "docker"
      - "dependencies"

  # GitHub Actions
  - package-ecosystem: "github-actions"
    directory: "/.github/workflows"
    schedule:
      interval: "weekly"
      day: "wednesday"
      time: "06:00"
    reviewers:
      - "terragon-labs/devops-team"
    labels:
      - "github-actions"
      - "dependencies"
```

## 5. Issue and PR Templates

### Issue Templates

Create `.github/ISSUE_TEMPLATE/bug_report.yml`:

```yaml
name: Bug Report
description: Report a bug or unexpected behavior
title: "[BUG]: "
labels: ["bug", "needs-triage"]
assignees: []

body:
  - type: markdown
    attributes:
      value: |
        Thanks for taking the time to fill out this bug report!

  - type: textarea
    id: what-happened
    attributes:
      label: What happened?
      description: A clear and concise description of what the bug is.
      placeholder: Tell us what you see!
    validations:
      required: true

  - type: textarea
    id: expected-behavior
    attributes:
      label: Expected Behavior
      description: What did you expect to happen?
      placeholder: Tell us what you expected!
    validations:
      required: true

  - type: textarea
    id: steps-to-reproduce
    attributes:
      label: Steps to Reproduce
      description: Steps to reproduce the behavior
      placeholder: |
        1. Go to '...'
        2. Click on '....'
        3. Scroll down to '....'
        4. See error
    validations:
      required: true

  - type: textarea
    id: environment
    attributes:
      label: Environment
      description: |
        Please provide the following information:
      value: |
        - OS: [e.g. Ubuntu 20.04, macOS 12.0, Windows 11]
        - Python version: [e.g. 3.11.0]
        - Package version: [e.g. 2.0.0]
    validations:
      required: true

  - type: textarea
    id: additional-context
    attributes:
      label: Additional Context
      description: Add any other context about the problem here.
      placeholder: Logs, screenshots, etc.

  - type: checkboxes
    id: terms
    attributes:
      label: Code of Conduct
      description: By submitting this issue, you agree to follow our Code of Conduct
      options:
        - label: I agree to follow this project's Code of Conduct
          required: true
```

Create `.github/ISSUE_TEMPLATE/feature_request.yml`:

```yaml
name: Feature Request
description: Suggest an idea for this project
title: "[FEATURE]: "
labels: ["enhancement", "needs-triage"]
assignees: []

body:
  - type: markdown
    attributes:
      value: |
        Thanks for suggesting a new feature!

  - type: textarea
    id: problem
    attributes:
      label: Problem Statement
      description: Is your feature request related to a problem? Please describe.
      placeholder: A clear and concise description of what the problem is.
    validations:
      required: true

  - type: textarea
    id: solution
    attributes:
      label: Proposed Solution
      description: Describe the solution you'd like
      placeholder: A clear and concise description of what you want to happen.
    validations:
      required: true

  - type: textarea
    id: alternatives
    attributes:
      label: Alternatives Considered
      description: Describe alternatives you've considered
      placeholder: A clear and concise description of any alternative solutions or features you've considered.

  - type: textarea
    id: additional-context
    attributes:
      label: Additional Context
      description: Add any other context or screenshots about the feature request here.

  - type: checkboxes
    id: terms
    attributes:
      label: Code of Conduct
      description: By submitting this issue, you agree to follow our Code of Conduct
      options:
        - label: I agree to follow this project's Code of Conduct
          required: true
```

### Pull Request Template

Create `.github/pull_request_template.md`:

```markdown
## Description

Brief description of the changes in this PR.

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring
- [ ] Security enhancement

## Testing

- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Performance tests pass (if applicable)
- [ ] Security scans pass
- [ ] Manual testing completed

## Checklist

- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published

## Security Considerations

- [ ] No secrets or sensitive data exposed
- [ ] Input validation implemented where necessary
- [ ] Authorization checks in place
- [ ] SQL injection prevention measures applied

## Performance Impact

- [ ] No performance regression
- [ ] Performance improvements quantified (if applicable)
- [ ] Resource usage impact assessed

## Breaking Changes

List any breaking changes and migration steps needed.

## Screenshots (if applicable)

Add screenshots to help explain your changes.

## Additional Notes

Any additional information that reviewers should know.
```

## Implementation Steps

1. **Create workflow files** in `.github/workflows/` directory
2. **Configure Dependabot** by creating `.github/dependabot.yml`
3. **Set up issue templates** in `.github/ISSUE_TEMPLATE/`
4. **Add PR template** as `.github/pull_request_template.md`
5. **Configure repository secrets** for external integrations:
   - `SONAR_TOKEN` for SonarCloud
   - Any additional service tokens

## Benefits

- **Enhanced Security**: Multi-layered security scanning with SARIF integration
- **Performance Monitoring**: Automated performance regression detection
- **Quality Assurance**: Comprehensive code quality gates
- **Dependency Management**: Automated security updates and dependency tracking
- **Developer Experience**: Structured issue reporting and PR workflows

## Maintenance

- Review and update workflow configurations quarterly
- Monitor security scan results and address findings promptly
- Adjust performance thresholds based on application growth
- Update dependency ignore lists as needed
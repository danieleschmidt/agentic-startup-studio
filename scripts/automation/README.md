# Automation Scripts

This directory contains comprehensive automation scripts for repository maintenance, monitoring, and reporting.

## Scripts Overview

### 1. Metrics Collector (`metrics_collector.py`)
Collects comprehensive repository metrics including:
- Code quality metrics (lines of code, complexity, etc.)
- Development metrics (commits, contributors, etc.)
- Testing metrics (test files, coverage indicators)
- Security metrics (vulnerability patterns, config files)
- Infrastructure metrics (Docker, Kubernetes, etc.)
- Documentation metrics (README, docstrings, etc.)

**Usage:**
```bash
python scripts/automation/metrics_collector.py --output .github/latest-metrics.json --report metrics-report.md
```

### 2. Dependency Updater (`dependency_updater.py`)
Manages project dependencies across different package managers:
- Python dependencies (pip, requirements.txt, pyproject.toml)
- Node.js dependencies (npm, yarn)
- Docker image dependencies
- GitHub Actions dependencies
- Security vulnerability scanning

**Usage:**
```bash
# Check dependencies
python scripts/automation/dependency_updater.py --check --output .github/dependency-check.json

# Update dependencies (security only)
python scripts/automation/dependency_updater.py --update all --update-type security
```

### 3. Code Quality Monitor (`code_quality_monitor.py`)
Comprehensive code quality analysis:
- Code complexity analysis
- Security pattern detection
- Documentation coverage
- Maintainability metrics
- Style compliance checking
- Technical debt assessment

**Usage:**
```bash
python scripts/automation/code_quality_monitor.py --output .github/code-quality-report.json --summary quality-summary.md
```

### 4. Repository Maintenance (`repository_maintenance.py`)
Automated repository maintenance tasks:
- Git cleanup (garbage collection, branch pruning)
- File cleanup (cache files, temporary files)
- Dependency auditing
- Security scanning
- Performance optimization
- Backup verification

**Usage:**
```bash
python scripts/automation/repository_maintenance.py --output .github/maintenance-report.json --summary maintenance-summary.md
```

### 5. Automated Reporting (`automated_reporting.py`)
Generates stakeholder-specific reports:
- Executive/management reports
- Technical team reports
- Security team reports
- Comprehensive project reports

**Usage:**
```bash
# Generate all reports
python scripts/automation/automated_reporting.py --type all --output-dir .github/reports

# Generate specific report type
python scripts/automation/automated_reporting.py --type management --output-dir .github/reports
```

### 6. Automation Orchestrator (`automation_orchestrator.py`)
Central coordination system for all automation tasks:
- Daily automation cycles
- Weekly automation cycles
- Scheduled task management
- On-demand task execution
- Status monitoring

**Usage:**
```bash
# Run daily automation cycle
python scripts/automation/automation_orchestrator.py --mode run --task daily

# Run as scheduled daemon
python scripts/automation/automation_orchestrator.py --mode schedule

# Check automation status
python scripts/automation/automation_orchestrator.py --mode status
```

## Automation Cycles

### Daily Automation
- Metrics collection
- Dependency security check
- Code quality scan
- Basic security scan

### Weekly Automation
- Repository maintenance
- Comprehensive reporting
- Dependency updates (security)
- Performance analysis

## Configuration

### Environment Variables
```bash
# Optional: Email notifications
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@example.com
SMTP_PASSWORD=your-app-password

# Optional: Slack notifications
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...

# Optional: Custom paths
AUTOMATION_LOG_DIR=.github/logs
AUTOMATION_REPORTS_DIR=.github/reports
```

### Scheduled Execution
To run automation as scheduled tasks, you can use:

1. **Cron Jobs** (Linux/macOS):
```bash
# Add to crontab (crontab -e)
0 2 * * * cd /path/to/repo && python scripts/automation/automation_orchestrator.py --mode run --task daily
0 3 * * 0 cd /path/to/repo && python scripts/automation/automation_orchestrator.py --mode run --task weekly
```

2. **GitHub Actions** (see `.github/workflows/` examples):
```yaml
name: Daily Automation
on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM UTC
jobs:
  automation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Daily Automation
        run: python scripts/automation/automation_orchestrator.py --mode run --task daily
```

3. **systemd Service** (Linux):
```ini
# /etc/systemd/system/repo-automation.service
[Unit]
Description=Repository Automation
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/repo
ExecStart=/usr/bin/python scripts/automation/automation_orchestrator.py --mode schedule
Restart=always

[Install]
WantedBy=multi-user.target
```

## Output Files

All automation scripts generate structured output in the `.github/` directory:

- `.github/latest-metrics.json` - Latest metrics data
- `.github/dependency-check.json` - Dependency status
- `.github/code-quality-report.json` - Code quality analysis
- `.github/maintenance-report.json` - Maintenance results
- `.github/reports/` - Stakeholder reports
- `.github/logs/automation.log` - Automation activity log
- `.github/automation-results/` - Historical automation results

## Integration with CI/CD

These scripts are designed to integrate with existing CI/CD pipelines:

1. **Pre-commit Hooks**: Run quality checks before commits
2. **PR Validation**: Run comprehensive checks on pull requests
3. **Deployment Pipeline**: Run maintenance and reporting after deployments
4. **Monitoring Integration**: Feed metrics into monitoring systems

## Troubleshooting

### Common Issues

1. **Permission Errors**:
   - Ensure scripts are executable: `chmod +x scripts/automation/*.py`
   - Check file permissions in `.github/` directory

2. **Missing Dependencies**:
   - Install required packages: `pip install psutil schedule`
   - For Git operations, ensure Git is installed and configured

3. **Large Repository Performance**:
   - Scripts include performance optimizations and limits
   - Adjust file scanning limits in individual scripts if needed

4. **Network Dependencies**:
   - Some features require internet access for security scanning
   - Configure proxies if needed in corporate environments

### Debug Mode
Most scripts support verbose output for debugging:
```bash
python scripts/automation/metrics_collector.py --verbose
```

### Log Analysis
Check automation logs for detailed execution information:
```bash
tail -f .github/logs/automation.log
```

## Security Considerations

- Scripts scan for potential security issues but don't modify critical files
- Dependency updates are limited to security patches by default
- Sensitive information detection helps identify hardcoded secrets
- All operations are logged for audit purposes

## Customization

Scripts are designed to be modular and customizable:

1. **Modify Thresholds**: Edit quality score calculations and alert thresholds
2. **Add New Metrics**: Extend metric collection with custom data points
3. **Custom Reports**: Create new report templates for specific stakeholders
4. **Integration Hooks**: Add webhook calls or API integrations

## Support

For issues or feature requests:
1. Check the automation logs: `.github/logs/automation.log`
2. Review the generated reports for specific error details
3. Create an issue with relevant log excerpts and configuration details
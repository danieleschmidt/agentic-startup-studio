# SDLC Getting Started Guide

This guide helps you understand and utilize the comprehensive Software Development Life Cycle (SDLC) implementation in the Agentic Startup Studio project.

## Quick Overview

The project implements an enterprise-grade SDLC with 95% completeness, featuring:
- **Automated Testing**: 90%+ coverage with quality gates
- **Continuous Integration**: Automated workflows for testing, security, and deployment
- **Monitoring & Observability**: Real-time metrics, logging, and alerting
- **Security & Compliance**: HIPAA-ready framework with automated vulnerability scanning
- **Automated Reporting**: Stakeholder-specific reports and metrics

## Getting Started Checklist

### For Developers 👩‍💻

#### Initial Setup (5 minutes)
```bash
# 1. Clone and setup environment
git clone https://github.com/danieleschmidt/agentic-startup-studio.git
cd agentic-startup-studio

# 2. Use DevContainer (recommended) or local setup
# DevContainer: Open in VS Code and "Reopen in Container"
# Local: pip install -r requirements.txt

# 3. Install pre-commit hooks
pip install pre-commit
pre-commit install

# 4. Verify setup
python -m pytest tests/ -v
```

#### Daily Development Workflow
```bash
# 1. Run code quality checks
python scripts/automation/code_quality_monitor.py --verbose

# 2. Check dependencies for security issues
python scripts/automation/dependency_updater.py --check

# 3. Run comprehensive tests before committing
python -m pytest tests/ --cov=. --cov-report=html

# 4. Commit with semantic messages
git commit -m "feat: add new feature"  # follows conventional commits
```

#### Working with the Automation System
```bash
# View current repository health
python scripts/automation/repository_maintenance.py --summary

# Generate development metrics
python scripts/automation/metrics_collector.py --report dev-metrics.md

# Run daily automation cycle manually
python scripts/automation/automation_orchestrator.py --task daily
```

### For Project Managers 📊

#### Accessing Reports
The automation system generates regular reports for different stakeholders:

```bash
# Generate management-focused reports
python scripts/automation/automated_reporting.py --type management

# View latest project metrics
cat .github/latest-metrics.json | jq '.quality_metrics'

# Check project health score
python scripts/automation/repository_maintenance.py | grep "health_score"
```

#### Understanding Project Status
- **Health Score**: Repository health metric (0-100)
- **Test Coverage**: Percentage of code covered by tests
- **Security Score**: Security posture assessment
- **Automation Level**: Percentage of processes automated

#### Key Metrics Locations
- `.github/reports/management-report.md` - Executive summary
- `.github/latest-metrics.json` - Real-time metrics
- `.github/project-metrics.json` - Project configuration and targets

### For Security Teams 🛡️

#### Security Monitoring
```bash
# Run comprehensive security scan
python scripts/automation/repository_maintenance.py --output security-scan.json

# Generate security-focused report
python scripts/automation/automated_reporting.py --type security

# Check for potential vulnerabilities
python scripts/automation/code_quality_monitor.py | grep -i "security"
```

#### Security Features
- **Automated Vulnerability Scanning**: Daily dependency and code scans
- **Secret Detection**: Automated detection of hardcoded secrets
- **Compliance Framework**: HIPAA-ready implementation
- **Audit Logging**: Comprehensive activity logging

#### Security Metrics
- **Vulnerability Count**: Number of identified security issues
- **Dependency Status**: Security status of all dependencies
- **Compliance Score**: HIPAA compliance readiness percentage

### For DevOps Teams ⚙️

#### Infrastructure Monitoring
```bash
# Check infrastructure status
python scripts/automation/automated_reporting.py --type technical

# View system performance metrics
cat .github/latest-metrics.json | jq '.performance_metrics'

# Monitor automation system status
python scripts/automation/automation_orchestrator.py --mode status
```

#### Deployment and Monitoring
- **Container Health**: Docker container status and performance
- **Build Performance**: Build times and optimization opportunities
- **Monitoring Stack**: Prometheus/Grafana observability
- **Automation Status**: Health of automated processes

#### Key Infrastructure Files
- `docker-compose.yml` - Local development environment
- `monitoring/` - Observability configuration
- `k8s/` - Kubernetes deployment manifests
- `.github/workflows/` - CI/CD workflow templates

## Understanding the Automation System

### Automation Cycles

#### Daily Automation (Automated at 2 AM)
1. **Metrics Collection**: Gather repository statistics
2. **Dependency Check**: Scan for security vulnerabilities
3. **Code Quality Scan**: Analyze code quality and complexity
4. **Security Scan**: Basic security pattern detection

#### Weekly Automation (Sundays at 3 AM)
1. **Repository Maintenance**: Git cleanup, file optimization
2. **Comprehensive Reporting**: Generate stakeholder reports
3. **Dependency Updates**: Apply security patches
4. **Performance Analysis**: System optimization opportunities

#### On-Demand Execution
```bash
# Run specific automation tasks
python scripts/automation/automation_orchestrator.py --task metrics
python scripts/automation/automation_orchestrator.py --task quality
python scripts/automation/automation_orchestrator.py --task maintenance
```

### Report Types and Audiences

#### Management Reports (`management-report.md`)
- **Executive Summary**: Overall project status and health
- **Key Achievements**: Recent accomplishments and milestones
- **Critical Issues**: Items requiring management attention
- **Resource Needs**: Budget, personnel, or infrastructure requirements

#### Technical Reports (`technical-report.md`)
- **Code Quality Metrics**: Coverage, complexity, technical debt
- **Development Activity**: Commits, contributors, velocity
- **Infrastructure Status**: System performance and health
- **Security Analysis**: Vulnerability status and recommendations

#### Security Reports (`security-report.md`)
- **Vulnerability Assessment**: Current security posture
- **Compliance Status**: HIPAA and other compliance metrics
- **Risk Analysis**: Identified security risks and mitigation
- **Remediation Plan**: Steps to address security issues

## Advanced Usage

### Customizing Automation

#### Modifying Metrics Collection
Edit `.github/project-metrics.json` to customize:
- Target thresholds for quality metrics
- Alert conditions and escalation rules
- Reporting frequency and stakeholders
- Custom metrics and data points

#### Extending Automation Scripts
```python
# Example: Adding custom quality checks
# Edit scripts/automation/code_quality_monitor.py

def _analyze_custom_patterns(self) -> Dict[str, Any]:
    """Add your custom code analysis patterns."""
    return {"custom_metric": "value"}
```

#### Scheduling Custom Tasks
```python
# Edit scripts/automation/automation_orchestrator.py
# Add to setup_scheduled_tasks()

schedule.every().friday.at("17:00").do(self._weekly_team_report)
```

### Integration with External Tools

#### Slack Integration
```bash
# Set environment variable for notifications
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."

# Notifications will be sent for critical issues
```

#### Email Reports
```bash
# Configure SMTP settings
export SMTP_SERVER="smtp.gmail.com"
export SMTP_USERNAME="your-email@example.com"
export SMTP_PASSWORD="your-app-password"
```

#### Monitoring System Integration
```bash
# Prometheus metrics endpoint
curl http://localhost:8000/metrics

# Grafana dashboard import
# Use configurations in grafana/dashboards/
```

## Troubleshooting

### Common Issues

#### Automation Scripts Not Running
```bash
# Check permissions
chmod +x scripts/automation/*.py

# Verify Python path
which python
python --version

# Check dependencies
pip install -r requirements.txt
```

#### Test Coverage Below Threshold
```bash
# Identify untested code
python -m pytest --cov=. --cov-report=html
open htmlcov/index.html

# Add tests for uncovered areas
# Coverage threshold is set to 90%
```

#### Security Scan Failures
```bash
# Review security issues
python scripts/automation/repository_maintenance.py --summary security-issues.md

# Common fixes:
# - Remove hardcoded secrets
# - Update vulnerable dependencies
# - Fix identified security patterns
```

#### Performance Issues
```bash
# Check system resources
python scripts/automation/metrics_collector.py | grep -i "performance"

# Monitor build times
docker build --progress=plain .

# Optimize large files
python scripts/automation/repository_maintenance.py | grep -i "large"
```

### Getting Help

#### Log Analysis
```bash
# View automation logs
tail -f .github/logs/automation.log

# Check specific task results
ls .github/automation-results/
cat .github/automation-results/daily_automation_*.json
```

#### Debug Mode
```bash
# Run scripts with verbose output
python scripts/automation/metrics_collector.py --verbose
python scripts/automation/code_quality_monitor.py --verbose
```

#### Health Checks
```bash
# Overall system health
python scripts/automation/automation_orchestrator.py --mode status

# Specific component health
python scripts/health_check_standalone.py
```

## Best Practices

### Development Workflow
1. **Always run tests** before committing code
2. **Use semantic commit messages** for automated changelog generation
3. **Review quality reports** regularly to maintain code quality
4. **Monitor security alerts** and address vulnerabilities promptly

### Code Quality
1. **Maintain 90%+ test coverage** for all new code
2. **Keep functions simple** (complexity < 10)
3. **Document all public APIs** with comprehensive docstrings
4. **Follow established patterns** for consistency

### Security
1. **Never commit secrets** or sensitive information
2. **Update dependencies regularly** for security patches
3. **Review security reports** weekly
4. **Follow least privilege principles** for access control

### Monitoring
1. **Check automation reports** daily
2. **Monitor system performance** during high load
3. **Set up alerts** for critical metrics
4. **Review and adjust thresholds** based on project needs

## Next Steps

After completing this getting started guide:

1. **Explore the codebase** using the automated metrics and reports
2. **Set up your development environment** with pre-commit hooks
3. **Run the automation system** to understand current project status
4. **Customize reports** to match your team's specific needs
5. **Integrate with your existing tools** (Slack, email, monitoring)

For advanced configuration and customization, refer to:
- `docs/SDLC_IMPLEMENTATION_SUMMARY.md` - Complete implementation details
- `scripts/automation/README.md` - Detailed automation documentation
- Individual script documentation within each automation file

---

*This guide covers the essential aspects of working with the SDLC implementation. For specific technical questions, refer to the comprehensive documentation in the `docs/` directory or automation script comments.*
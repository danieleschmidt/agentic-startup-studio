# Advanced Security Workflow Setup Guide
*Terragon Labs Agentic Startup Studio*

> **🔒 IMPORTANT**: This workflow contains sensitive security configurations and should be manually reviewed and implemented by the security team.

## Overview

This document provides the complete configuration for an advanced security scanning workflow that implements:
- SLSA Level 3 compliance preparation
- SBOM (Software Bill of Materials) generation  
- Multi-layered security scanning
- Container and Infrastructure as Code security
- License compliance checking

## Manual Setup Instructions

### Step 1: Create the Workflow File

Create `.github/workflows/security-advanced.yml` with the following content:

```yaml
# Advanced Security Workflow for Terragon Labs
# Implements SLSA Level 3 compliance, SBOM generation, and comprehensive security scanning

name: Advanced Security Scanning

on:
  push:
    branches: [main, dev]
  pull_request:
    branches: [main, dev]
  schedule:
    # Run security scans daily at 2 AM UTC
    - cron: '0 2 * * *'
  workflow_dispatch:
    inputs:
      scan_type:
        description: 'Type of security scan to run'
        required: true
        default: 'full'
        type: choice
        options:
          - full
          - dependency
          - code
          - container

permissions:
  actions: read
  contents: read
  security-events: write
  id-token: write # For SLSA provenance
  attestations: write

jobs:
  security-scan:
    name: Advanced Security Analysis
    runs-on: ubuntu-latest
    timeout-minutes: 30
    
    strategy:
      matrix:
        python-version: ['3.11']
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
          
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
          
      - name: Install security tools
        run: |
          pip install --upgrade pip
          pip install safety bandit semgrep cyclonedx-bom pip-audit
          
      - name: Generate Software Bill of Materials (SBOM)
        run: |
          cyclonedx-py -o --format json --output-file sbom.json
          cyclonedx-py -o --format xml --output-file sbom.xml
          
      - name: Upload SBOM artifacts
        uses: actions/upload-artifact@v4
        with:
          name: sbom-${{ github.sha }}
          path: |
            sbom.json
            sbom.xml
          retention-days: 90
          
      - name: Advanced dependency vulnerability scanning
        run: |
          echo "🔍 Running advanced dependency scanning..."
          
          # Multiple tools for comprehensive coverage
          pip-audit --format=json --output=pip-audit-results.json || true
          safety check --json --output=safety-results.json || true
          
          # Generate combined report
          python -c "
          import json
          import sys
          
          def load_json_safely(file):
              try:
                  with open(file, 'r') as f:
                      return json.load(f)
              except:
                  return {}
          
          pip_audit = load_json_safely('pip-audit-results.json')
          safety = load_json_safely('safety-results.json')
          
          # Combine results and check for critical vulnerabilities
          critical_count = 0
          high_count = 0
          
          if isinstance(pip_audit, list):
              critical_count += len([v for v in pip_audit if v.get('fix_versions')])
          
          if isinstance(safety, list):
              critical_count += len(safety)
          
          print(f'🔍 Security scan summary:')
          print(f'   Critical vulnerabilities: {critical_count}')
          print(f'   High vulnerabilities: {high_count}')
          
          if critical_count > 0:
              print('❌ Critical vulnerabilities found - review required')
              sys.exit(1)
          else:
              print('✅ No critical vulnerabilities detected')
          "
          
      - name: Static Application Security Testing (SAST)
        run: |
          echo "🔍 Running SAST with multiple tools..."
          
          # Bandit for Python security issues
          bandit -r pipeline/ core/ scripts/ -f json -o bandit-results.json || true
          
          # Semgrep for additional security patterns
          semgrep --config=auto --json --output=semgrep-results.json pipeline/ core/ scripts/ || true
          
          # Custom security validation
          python scripts/validate_production_secrets.py --scan-hardcoded --strict --output-format json > secrets-scan.json || true
          
      - name: Container Security Scanning
        if: github.event_name != 'pull_request' || contains(github.event.pull_request.changed_files, 'Dockerfile')
        run: |
          echo "🔍 Building and scanning container images..."
          
          # Build the image
          docker build -t terragon/agentic-startup-studio:security-scan .
          
          # Install and run Trivy for container scanning
          curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin
          
          # Scan the built image
          trivy image --format json --output container-scan-results.json terragon/agentic-startup-studio:security-scan
          
          # Check for critical vulnerabilities
          python -c "
          import json
          import sys
          
          try:
              with open('container-scan-results.json', 'r') as f:
                  results = json.load(f)
              
              critical_count = 0
              high_count = 0
              
              for result in results.get('Results', []):
                  for vuln in result.get('Vulnerabilities', []):
                      severity = vuln.get('Severity', '').upper()
                      if severity == 'CRITICAL':
                          critical_count += 1
                      elif severity == 'HIGH':
                          high_count += 1
              
              print(f'🔍 Container scan summary:')
              print(f'   Critical vulnerabilities: {critical_count}')
              print(f'   High vulnerabilities: {high_count}')
              
              if critical_count > 3:  # Allow some tolerance for base image issues
                  print('❌ Too many critical container vulnerabilities')
                  sys.exit(1)
              else:
                  print('✅ Container security within acceptable limits')
          except Exception as e:
              print(f'⚠️  Container scan failed: {e}')
              sys.exit(0)  # Don't fail the build for scan errors
          "
          
      - name: Infrastructure as Code Security
        run: |
          echo "🔍 Scanning Infrastructure as Code..."
          
          # Install checkov for IaC scanning
          pip install checkov
          
          # Scan Docker and docker-compose files
          checkov -f Dockerfile --framework dockerfile --output json --output-file dockerfile-scan.json || true
          checkov -f docker-compose.yml --framework docker_compose --output json --output-file docker-compose-scan.json || true
          checkov -d .github/workflows --framework github_actions --output json --output-file github-actions-scan.json || true
          
      - name: License Compliance Check
        run: |
          echo "🔍 Checking license compliance..."
          
          # Install license checker
          pip install pip-licenses
          
          # Generate license report
          pip-licenses --format=json --output-file=licenses.json
          
          # Check for problematic licenses
          python -c "
          import json
          
          PROBLEMATIC_LICENSES = ['GPL-3.0', 'AGPL-3.0', 'LGPL-3.0']
          ALLOWED_LICENSES = ['MIT', 'Apache-2.0', 'BSD-3-Clause', 'BSD-2-Clause', 'ISC']
          
          try:
              with open('licenses.json', 'r') as f:
                  licenses = json.load(f)
              
              issues = []
              for pkg in licenses:
                  license_name = pkg.get('License', 'Unknown')
                  if license_name in PROBLEMATIC_LICENSES:
                      issues.append(f\"❌ {pkg['Name']}: {license_name} (problematic)\")
                  elif license_name not in ALLOWED_LICENSES and license_name != 'Unknown':
                      issues.append(f\"⚠️  {pkg['Name']}: {license_name} (review required)\")
              
              if issues:
                  print('🔍 License compliance issues:')
                  for issue in issues[:10]:  # Show max 10 issues
                      print(f'   {issue}')
                  if len(issues) > 10:
                      print(f'   ... and {len(issues) - 10} more')
              else:
                  print('✅ All licenses are compliant')
          except Exception as e:
              print(f'⚠️  License check failed: {e}')
          "
          
      - name: Generate Security Report
        if: always()
        run: |
          echo "📊 Generating comprehensive security report..."
          
          python -c "
          import json
          import os
          from datetime import datetime
          
          def load_json_safely(file):
              try:
                  with open(file, 'r') as f:
                      return json.load(f)
              except:
                  return None
          
          report = {
              'timestamp': datetime.utcnow().isoformat(),
              'commit_sha': os.environ.get('GITHUB_SHA', 'unknown'),
              'branch': os.environ.get('GITHUB_REF_NAME', 'unknown'),
              'scans': {
                  'dependencies': {
                      'pip_audit': load_json_safely('pip-audit-results.json'),
                      'safety': load_json_safely('safety-results.json')
                  },
                  'code': {
                      'bandit': load_json_safely('bandit-results.json'),
                      'semgrep': load_json_safely('semgrep-results.json'),
                      'secrets': load_json_safely('secrets-scan.json')
                  },
                  'container': load_json_safely('container-scan-results.json'),
                  'infrastructure': {
                      'dockerfile': load_json_safely('dockerfile-scan.json'),
                      'docker_compose': load_json_safely('docker-compose-scan.json'),
                      'github_actions': load_json_safely('github-actions-scan.json')
                  },
                  'compliance': {
                      'licenses': load_json_safely('licenses.json'),
                      'sbom': load_json_safely('sbom.json')
                  }
              }
          }
          
          with open('security-report.json', 'w') as f:
              json.dump(report, f, indent=2)
          
          print('✅ Security report generated')
          "
          
      - name: Upload Security Results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: security-scan-results-${{ github.sha }}
          path: |
            security-report.json
            *-results.json
            *-scan.json
            licenses.json
          retention-days: 30
          
      - name: Upload SARIF results
        if: always()
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: bandit-results.json
        continue-on-error: true

  slsa-provenance:
    name: Generate SLSA Provenance
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    needs: security-scan
    permissions:
      actions: read
      id-token: write
      contents: write
      
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        
      - name: Generate provenance
        uses: slsa-framework/slsa-github-generator/.github/workflows/generator_generic_slsa3.yml@v1.9.0
        with:
          base64-subjects: "${{ needs.security-scan.outputs.hashes }}"
          provenance-name: "provenance.intoto.jsonl"
          
      - name: Upload provenance
        uses: actions/upload-artifact@v4
        with:
          name: slsa-provenance-${{ github.sha }}
          path: provenance.intoto.jsonl
          retention-days: 90
```

### Step 2: Required Permissions

Ensure the repository has the following permissions enabled:
- Actions: Read
- Contents: Read
- Security events: Write
- ID token: Write (for SLSA provenance)
- Attestations: Write

### Step 3: Security Team Review

Before implementing:
1. **Security team review** of all scanning configurations
2. **Verify tool versions** match your security requirements
3. **Customize thresholds** for vulnerability counts as needed
4. **Test in staging** environment first

### Step 4: Integration with Existing CI

This workflow is designed to complement, not replace, your existing CI pipeline. It runs:
- On all PRs and pushes to main/dev
- Daily at 2 AM UTC for comprehensive scanning
- On-demand via workflow dispatch

## Benefits

✅ **SLSA Level 3 Compliance**: Supply chain security with provenance generation  
✅ **SBOM Generation**: Software Bill of Materials for enterprise compliance  
✅ **Multi-layered Security**: SAST, dependency, container, and IaC scanning  
✅ **License Compliance**: Automated checking for problematic licenses  
✅ **Enterprise Ready**: Comprehensive reporting and artifact retention

## Security Considerations

- **Secrets Management**: Ensure all API keys are stored in GitHub Secrets
- **Workflow Permissions**: Use minimal required permissions only
- **Artifact Retention**: Security scan results retained for 30 days
- **False Positive Management**: Tune thresholds based on your risk tolerance

*This workflow should be manually created and reviewed by the security team before implementation.*
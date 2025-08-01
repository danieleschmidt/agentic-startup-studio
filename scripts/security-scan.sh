#!/bin/bash

# =============================================================================
# Security Scanning Script for Agentic Startup Studio
# Comprehensive security scanning using multiple tools
# =============================================================================

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCAN_RESULTS_DIR="${PROJECT_ROOT}/security-reports"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Usage information
usage() {
    cat << EOF
Security Scanning Script for Agentic Startup Studio

Usage: $0 [OPTIONS]

OPTIONS:
    --code-scan         Run static code analysis (bandit, semgrep)
    --dependency-scan   Scan dependencies for vulnerabilities (safety, pip-audit)
    --container-scan    Scan container images (trivy, grype)
    --secrets-scan      Scan for secrets in code (detect-secrets, truffleHog)
    --infrastructure    Scan infrastructure as code (checkov, tfsec)
    --all              Run all security scans
    --output-format     Output format (json|sarif|text) (default: json)
    --severity-level    Minimum severity level (low|medium|high|critical) (default: medium)
    --fail-on-high     Exit with error code if high/critical issues found
    --image IMAGE      Container image to scan
    -h, --help         Display this help message

EXAMPLES:
    # Run all security scans
    $0 --all

    # Run code and dependency scans only
    $0 --code-scan --dependency-scan

    # Scan container image with failure on high severity
    $0 --container-scan --image myapp:latest --fail-on-high

EOF
}

# Parse command line arguments
CODE_SCAN=false
DEPENDENCY_SCAN=false
CONTAINER_SCAN=false
SECRETS_SCAN=false
INFRASTRUCTURE_SCAN=false
OUTPUT_FORMAT="json"
SEVERITY_LEVEL="medium"
FAIL_ON_HIGH=false
CONTAINER_IMAGE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --code-scan)
            CODE_SCAN=true
            shift
            ;;
        --dependency-scan)
            DEPENDENCY_SCAN=true
            shift
            ;;
        --container-scan)
            CONTAINER_SCAN=true
            shift
            ;;
        --secrets-scan)
            SECRETS_SCAN=true
            shift
            ;;
        --infrastructure)
            INFRASTRUCTURE_SCAN=true
            shift
            ;;
        --all)
            CODE_SCAN=true
            DEPENDENCY_SCAN=true
            CONTAINER_SCAN=true
            SECRETS_SCAN=true
            INFRASTRUCTURE_SCAN=true
            shift
            ;;
        --output-format)
            OUTPUT_FORMAT="$2"
            shift 2
            ;;
        --severity-level)
            SEVERITY_LEVEL="$2"
            shift 2
            ;;
        --fail-on-high)
            FAIL_ON_HIGH=true
            shift
            ;;
        --image)
            CONTAINER_IMAGE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Create scan results directory
setup_scan_environment() {
    log_info "Setting up security scan environment..."
    
    mkdir -p "${SCAN_RESULTS_DIR}"
    cd "${PROJECT_ROOT}"
    
    # Create consolidated report
    cat > "${SCAN_RESULTS_DIR}/scan-summary-${TIMESTAMP}.json" << EOF
{
  "scan_info": {
    "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "project_root": "${PROJECT_ROOT}",
    "severity_level": "${SEVERITY_LEVEL}",
    "output_format": "${OUTPUT_FORMAT}"
  },
  "results": {}
}
EOF
    
    log_success "Scan environment ready"
}

# Static code analysis
run_code_scan() {
    if [[ "${CODE_SCAN}" == true ]]; then
        log_info "Running static code analysis..."
        
        local code_issues=0
        
        # Bandit - Python security scanner
        if command -v bandit &> /dev/null; then
            log_info "Running Bandit security scan..."
            local bandit_output="${SCAN_RESULTS_DIR}/bandit-${TIMESTAMP}.${OUTPUT_FORMAT}"
            
            bandit -r pipeline/ core/ scripts/ \
                -f "${OUTPUT_FORMAT}" \
                -o "${bandit_output}" \
                --severity-level "${SEVERITY_LEVEL}" \
                --exit-zero || true
            
            if [[ -f "${bandit_output}" ]]; then
                if [[ "${OUTPUT_FORMAT}" == "json" ]]; then
                    local bandit_issues=$(jq '.results | length' "${bandit_output}" 2>/dev/null || echo "0")
                    code_issues=$((code_issues + bandit_issues))
                    log_info "Bandit found ${bandit_issues} security issues"
                fi
            fi
        else
            log_warning "Bandit not found. Install with: pip install bandit"
        fi
        
        # Semgrep - Static analysis tool
        if command -v semgrep &> /dev/null; then
            log_info "Running Semgrep scan..."
            local semgrep_output="${SCAN_RESULTS_DIR}/semgrep-${TIMESTAMP}.${OUTPUT_FORMAT}"
            
            semgrep --config=auto \
                --${OUTPUT_FORMAT} \
                --output="${semgrep_output}" \
                --severity=INFO \
                --quiet \
                . || true
            
            if [[ -f "${semgrep_output}" ]]; then
                if [[ "${OUTPUT_FORMAT}" == "json" ]]; then
                    local semgrep_issues=$(jq '.results | length' "${semgrep_output}" 2>/dev/null || echo "0")
                    code_issues=$((code_issues + semgrep_issues))
                    log_info "Semgrep found ${semgrep_issues} issues"
                fi
            fi
        else
            log_warning "Semgrep not found. Install with: pip install semgrep"
        fi
        
        # CodeQL (if available)
        if command -v codeql &> /dev/null; then
            log_info "Running CodeQL analysis..."
            # CodeQL setup would be more complex and typically run in CI
            log_info "CodeQL available but requires database setup"
        fi
        
        log_success "Code scan completed. Found ${code_issues} total issues"
        
        # Update summary
        jq ".results.code_scan = {\"total_issues\": ${code_issues}, \"tools_used\": [\"bandit\", \"semgrep\"]}" \
            "${SCAN_RESULTS_DIR}/scan-summary-${TIMESTAMP}.json" > /tmp/summary.json
        mv /tmp/summary.json "${SCAN_RESULTS_DIR}/scan-summary-${TIMESTAMP}.json"
    fi
}

# Dependency vulnerability scanning
run_dependency_scan() {
    if [[ "${DEPENDENCY_SCAN}" == true ]]; then
        log_info "Running dependency vulnerability scan..."
        
        local dep_issues=0
        
        # Safety - Python dependency scanner
        if command -v safety &> /dev/null; then
            log_info "Running Safety dependency scan..."
            local safety_output="${SCAN_RESULTS_DIR}/safety-${TIMESTAMP}.json"
            
            safety check --json --output "${safety_output}" || true
            
            if [[ -f "${safety_output}" ]]; then
                local safety_issues=$(jq '. | length' "${safety_output}" 2>/dev/null || echo "0")
                dep_issues=$((dep_issues + safety_issues))
                log_info "Safety found ${safety_issues} vulnerable dependencies"
            fi
        else
            log_warning "Safety not found. Install with: pip install safety"
        fi
        
        # pip-audit - Official Python dependency scanner
        if command -v pip-audit &> /dev/null; then
            log_info "Running pip-audit scan..."
            local pip_audit_output="${SCAN_RESULTS_DIR}/pip-audit-${TIMESTAMP}.json"
            
            pip-audit --format=json --output="${pip_audit_output}" || true
            
            if [[ -f "${pip_audit_output}" ]]; then
                local pip_audit_issues=$(jq '.vulnerabilities | length' "${pip_audit_output}" 2>/dev/null || echo "0")
                dep_issues=$((dep_issues + pip_audit_issues))
                log_info "pip-audit found ${pip_audit_issues} vulnerabilities"
            fi
        else
            log_warning "pip-audit not found. Install with: pip install pip-audit"
        fi
        
        # OSV-Scanner (if available)
        if command -v osv-scanner &> /dev/null; then
            log_info "Running OSV-Scanner..."
            local osv_output="${SCAN_RESULTS_DIR}/osv-${TIMESTAMP}.json"
            
            osv-scanner --format=json --output="${osv_output}" . || true
            
            if [[ -f "${osv_output}" ]]; then
                local osv_issues=$(jq '.results[0].packages | length' "${osv_output}" 2>/dev/null || echo "0")
                log_info "OSV-Scanner found ${osv_issues} vulnerable packages"
            fi
        fi
        
        log_success "Dependency scan completed. Found ${dep_issues} total vulnerabilities"
        
        # Update summary
        jq ".results.dependency_scan = {\"total_vulnerabilities\": ${dep_issues}, \"tools_used\": [\"safety\", \"pip-audit\"]}" \
            "${SCAN_RESULTS_DIR}/scan-summary-${TIMESTAMP}.json" > /tmp/summary.json
        mv /tmp/summary.json "${SCAN_RESULTS_DIR}/scan-summary-${TIMESTAMP}.json"
    fi
}

# Container image scanning
run_container_scan() {
    if [[ "${CONTAINER_SCAN}" == true ]]; then
        log_info "Running container security scan..."
        
        if [[ -z "${CONTAINER_IMAGE}" ]]; then
            log_warning "No container image specified. Use --image flag"
            return
        fi
        
        local container_issues=0
        
        # Trivy - Container vulnerability scanner
        if command -v trivy &> /dev/null; then
            log_info "Running Trivy container scan..."
            local trivy_output="${SCAN_RESULTS_DIR}/trivy-${TIMESTAMP}.json"
            
            trivy image --format json --output "${trivy_output}" \
                --severity HIGH,CRITICAL \
                "${CONTAINER_IMAGE}" || true
            
            if [[ -f "${trivy_output}" ]]; then
                local trivy_issues=$(jq '.Results[0].Vulnerabilities | length' "${trivy_output}" 2>/dev/null || echo "0")
                container_issues=$((container_issues + trivy_issues))
                log_info "Trivy found ${trivy_issues} vulnerabilities in container"
            fi
        else
            log_warning "Trivy not found. Install from: https://aquasecurity.github.io/trivy/"
        fi
        
        # Grype - Container vulnerability scanner
        if command -v grype &> /dev/null; then
            log_info "Running Grype container scan..."
            local grype_output="${SCAN_RESULTS_DIR}/grype-${TIMESTAMP}.json"
            
            grype -o json "${CONTAINER_IMAGE}" > "${grype_output}" || true
            
            if [[ -f "${grype_output}" ]]; then
                local grype_issues=$(jq '.matches | length' "${grype_output}" 2>/dev/null || echo "0")
                log_info "Grype found ${grype_issues} vulnerabilities"
            fi
        else
            log_warning "Grype not found. Install from: https://github.com/anchore/grype"
        fi
        
        log_success "Container scan completed. Found ${container_issues} total vulnerabilities"
        
        # Update summary
        jq ".results.container_scan = {\"image\": \"${CONTAINER_IMAGE}\", \"total_vulnerabilities\": ${container_issues}, \"tools_used\": [\"trivy\", \"grype\"]}" \
            "${SCAN_RESULTS_DIR}/scan-summary-${TIMESTAMP}.json" > /tmp/summary.json
        mv /tmp/summary.json "${SCAN_RESULTS_DIR}/scan-summary-${TIMESTAMP}.json"
    fi
}

# Secrets scanning
run_secrets_scan() {
    if [[ "${SECRETS_SCAN}" == true ]]; then
        log_info "Running secrets scan..."
        
        local secrets_found=0
        
        # detect-secrets
        if command -v detect-secrets &> /dev/null; then
            log_info "Running detect-secrets scan..."
            local secrets_output="${SCAN_RESULTS_DIR}/secrets-${TIMESTAMP}.json"
            
            detect-secrets scan --all-files --baseline "${secrets_output}" . || true
            
            if [[ -f "${secrets_output}" ]]; then
                local detected_secrets=$(jq '.results | to_entries | length' "${secrets_output}" 2>/dev/null || echo "0")
                secrets_found=$((secrets_found + detected_secrets))
                log_info "detect-secrets found ${detected_secrets} potential secrets"
            fi
        else
            log_warning "detect-secrets not found. Install with: pip install detect-secrets"
        fi
        
        # TruffleHog (if available)
        if command -v trufflehog &> /dev/null; then
            log_info "Running TruffleHog scan..."
            local trufflehog_output="${SCAN_RESULTS_DIR}/trufflehog-${TIMESTAMP}.json"
            
            trufflehog filesystem . --json > "${trufflehog_output}" || true
            
            if [[ -f "${trufflehog_output}" ]]; then
                local trufflehog_secrets=$(jq -s 'length' "${trufflehog_output}" 2>/dev/null || echo "0")
                log_info "TruffleHog found ${trufflehog_secrets} potential secrets"
            fi
        fi
        
        log_success "Secrets scan completed. Found ${secrets_found} potential secrets"
        
        # Update summary
        jq ".results.secrets_scan = {\"total_secrets\": ${secrets_found}, \"tools_used\": [\"detect-secrets\"]}" \
            "${SCAN_RESULTS_DIR}/scan-summary-${TIMESTAMP}.json" > /tmp/summary.json
        mv /tmp/summary.json "${SCAN_RESULTS_DIR}/scan-summary-${TIMESTAMP}.json"
    fi
}

# Infrastructure as Code scanning
run_infrastructure_scan() {
    if [[ "${INFRASTRUCTURE_SCAN}" == true ]]; then
        log_info "Running infrastructure security scan..."
        
        local iac_issues=0
        
        # Checkov - IaC security scanner
        if command -v checkov &> /dev/null; then
            log_info "Running Checkov IaC scan..."
            local checkov_output="${SCAN_RESULTS_DIR}/checkov-${TIMESTAMP}.json"
            
            checkov -d . --framework dockerfile,kubernetes,docker_compose \
                --output json --output-file "${checkov_output}" || true
            
            if [[ -f "${checkov_output}" ]]; then
                local checkov_issues=$(jq '.summary.failed' "${checkov_output}" 2>/dev/null || echo "0")
                iac_issues=$((iac_issues + checkov_issues))
                log_info "Checkov found ${checkov_issues} IaC security issues"
            fi
        else
            log_warning "Checkov not found. Install with: pip install checkov"
        fi
        
        log_success "Infrastructure scan completed. Found ${iac_issues} total issues"
        
        # Update summary
        jq ".results.infrastructure_scan = {\"total_issues\": ${iac_issues}, \"tools_used\": [\"checkov\"]}" \
            "${SCAN_RESULTS_DIR}/scan-summary-${TIMESTAMP}.json" > /tmp/summary.json
        mv /tmp/summary.json "${SCAN_RESULTS_DIR}/scan-summary-${TIMESTAMP}.json"
    fi
}

# Generate consolidated report
generate_report() {
    log_info "Generating consolidated security report..."
    
    local summary_file="${SCAN_RESULTS_DIR}/scan-summary-${TIMESTAMP}.json"
    local report_file="${SCAN_RESULTS_DIR}/security-report-${TIMESTAMP}.html"
    
    # Calculate total issues
    local total_issues=0
    if [[ -f "${summary_file}" ]]; then
        total_issues=$(jq '
            (.results.code_scan.total_issues // 0) +
            (.results.dependency_scan.total_vulnerabilities // 0) +
            (.results.container_scan.total_vulnerabilities // 0) +
            (.results.secrets_scan.total_secrets // 0) +
            (.results.infrastructure_scan.total_issues // 0)
        ' "${summary_file}")
    fi
    
    # Generate HTML report
    cat > "${report_file}" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>Security Scan Report - ${TIMESTAMP}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #f4f4f4; padding: 20px; border-radius: 5px; }
        .summary { background: #e8f5e8; padding: 15px; margin: 20px 0; border-radius: 5px; }
        .warning { background: #fff3cd; padding: 15px; margin: 20px 0; border-radius: 5px; }
        .error { background: #f8d7da; padding: 15px; margin: 20px 0; border-radius: 5px; }
        .details { margin: 20px 0; }
        table { width: 100%; border-collapse: collapse; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Security Scan Report</h1>
        <p><strong>Timestamp:</strong> ${TIMESTAMP}</p>
        <p><strong>Project:</strong> Agentic Startup Studio</p>
        <p><strong>Total Issues Found:</strong> ${total_issues}</p>
    </div>
    
    <div class="summary">
        <h2>Scan Summary</h2>
        <p>Comprehensive security scan completed with multiple tools.</p>
        <p>Results saved to: ${SCAN_RESULTS_DIR}/</p>
    </div>
    
    <div class="details">
        <h2>Detailed Results</h2>
        <p>Individual scan results are available in JSON format in the scan results directory.</p>
        <p>Review each tool's output for specific vulnerability details and remediation guidance.</p>
    </div>
</body>
</html>
EOF
    
    log_success "Security scan completed!"
    log_info "Total issues found: ${total_issues}"
    log_info "Summary report: ${summary_file}"
    log_info "HTML report: ${report_file}"
    log_info "Individual tool reports: ${SCAN_RESULTS_DIR}/"
    
    # Check if we should fail on high severity issues
    if [[ "${FAIL_ON_HIGH}" == true ]] && [[ ${total_issues} -gt 0 ]]; then
        log_error "Security scan found ${total_issues} issues. Failing due to --fail-on-high flag."
        exit 1
    fi
}

# Main execution
main() {
    log_info "Starting security scan for Agentic Startup Studio..."
    
    # If no specific scans requested, show usage
    if [[ "${CODE_SCAN}" == false ]] && 
       [[ "${DEPENDENCY_SCAN}" == false ]] && 
       [[ "${CONTAINER_SCAN}" == false ]] && 
       [[ "${SECRETS_SCAN}" == false ]] && 
       [[ "${INFRASTRUCTURE_SCAN}" == false ]]; then
        log_warning "No scan types specified. Use --all or specific scan flags."
        usage
        exit 1
    fi
    
    setup_scan_environment
    run_code_scan
    run_dependency_scan
    run_container_scan
    run_secrets_scan
    run_infrastructure_scan
    generate_report
}

# Execute main function
main "$@"
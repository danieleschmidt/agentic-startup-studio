#!/bin/bash
set -euo pipefail

# Container Security Scanning Script for Agentic Startup Studio
# Performs comprehensive security scanning of container images

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration
IMAGE_NAME="${1:-agentic-startup-studio}"
IMAGE_TAG="${2:-latest}"
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"
REPORT_DIR="${PROJECT_ROOT}/security_reports"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Check if required tools are installed
check_dependencies() {
    log_info "Checking security scanning dependencies..."
    
    local missing_tools=()
    
    # Check for Trivy
    if ! command -v trivy &> /dev/null; then
        missing_tools+=("trivy")
    fi
    
    # Check for Docker
    if ! command -v docker &> /dev/null; then
        missing_tools+=("docker")
    fi
    
    # Check for Syft (for SBOM generation)
    if ! command -v syft &> /dev/null; then
        log_warn "Syft not found - SBOM generation will be skipped"
    fi
    
    # Check for Grype (vulnerability scanning)
    if ! command -v grype &> /dev/null; then
        log_warn "Grype not found - additional vulnerability scanning will be skipped"
    fi
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_info "Please install missing tools and try again"
        exit 1
    fi
    
    log_success "All required dependencies found"
}

# Create report directory
setup_reporting() {
    log_info "Setting up security report directory..."
    mkdir -p "$REPORT_DIR"
    
    # Create subdirectory for this scan
    SCAN_DIR="${REPORT_DIR}/scan_${TIMESTAMP}"
    mkdir -p "$SCAN_DIR"
    
    log_success "Report directory created: $SCAN_DIR"
}

# Run Trivy vulnerability scan
run_trivy_scan() {
    log_info "Running Trivy vulnerability scan..."
    
    local trivy_report="${SCAN_DIR}/trivy_vulnerabilities.json"
    local trivy_summary="${SCAN_DIR}/trivy_summary.txt"
    
    # Full vulnerability scan
    if trivy image --format json --output "$trivy_report" "$FULL_IMAGE_NAME"; then
        log_success "Trivy vulnerability scan completed"
        
        # Generate human-readable summary
        trivy image --format table --output "$trivy_summary" "$FULL_IMAGE_NAME"
        
        # Extract critical vulnerabilities count
        local critical_count
        critical_count=$(jq '[.Results[]?.Vulnerabilities[]? | select(.Severity == "CRITICAL")] | length' "$trivy_report" 2>/dev/null || echo "0")
        
        local high_count
        high_count=$(jq '[.Results[]?.Vulnerabilities[]? | select(.Severity == "HIGH")] | length' "$trivy_report" 2>/dev/null || echo "0")
        
        echo "CRITICAL: $critical_count, HIGH: $high_count" > "${SCAN_DIR}/vulnerability_counts.txt"
        
        if [ "$critical_count" -gt 0 ] || [ "$high_count" -gt 10 ]; then
            log_warn "Found $critical_count CRITICAL and $high_count HIGH vulnerabilities"
            return 1
        else
            log_success "Vulnerability scan passed - $critical_count CRITICAL, $high_count HIGH"
        fi
    else
        log_error "Trivy vulnerability scan failed"
        return 1
    fi
}

# Run Trivy configuration scan
run_trivy_config_scan() {
    log_info "Running Trivy configuration scan..."
    
    local config_report="${SCAN_DIR}/trivy_config.json"
    local config_summary="${SCAN_DIR}/trivy_config_summary.txt"
    
    # Scan Dockerfile and related configs
    if trivy config --format json --output "$config_report" "$PROJECT_ROOT"; then
        log_success "Trivy configuration scan completed"
        
        # Generate human-readable summary
        trivy config --format table --output "$config_summary" "$PROJECT_ROOT"
        
        # Check for high severity misconfigurations
        local high_config_count
        high_config_count=$(jq '[.Results[]?.Misconfigurations[]? | select(.Severity == "HIGH" or .Severity == "CRITICAL")] | length' "$config_report" 2>/dev/null || echo "0")
        
        if [ "$high_config_count" -gt 0 ]; then
            log_warn "Found $high_config_count HIGH/CRITICAL configuration issues"
            return 1
        else
            log_success "Configuration scan passed"
        fi
    else
        log_error "Trivy configuration scan failed"
        return 1
    fi
}

# Run Trivy secret scan
run_trivy_secret_scan() {
    log_info "Running Trivy secret scan..."
    
    local secret_report="${SCAN_DIR}/trivy_secrets.json"
    local secret_summary="${SCAN_DIR}/trivy_secrets_summary.txt"
    
    # Scan for secrets in image
    if trivy image --scanners secret --format json --output "$secret_report" "$FULL_IMAGE_NAME"; then
        log_success "Trivy secret scan completed"
        
        # Generate human-readable summary
        trivy image --scanners secret --format table --output "$secret_summary" "$FULL_IMAGE_NAME"
        
        # Check for any secrets found
        local secret_count
        secret_count=$(jq '[.Results[]?.Secrets[]?] | length' "$secret_report" 2>/dev/null || echo "0")
        
        if [ "$secret_count" -gt 0 ]; then
            log_error "Found $secret_count secrets in container image"
            return 1
        else
            log_success "No secrets found in container image"
        fi
    else
        log_error "Trivy secret scan failed"
        return 1
    fi
}

# Generate SBOM using Syft
generate_sbom() {
    if command -v syft &> /dev/null; then
        log_info "Generating SBOM with Syft..."
        
        local sbom_json="${SCAN_DIR}/sbom.spdx.json"
        local sbom_table="${SCAN_DIR}/sbom_summary.txt"
        
        # Generate SPDX format SBOM
        if syft "$FULL_IMAGE_NAME" -o spdx-json="$sbom_json"; then
            log_success "SBOM generated successfully"
            
            # Generate human-readable summary
            syft "$FULL_IMAGE_NAME" -o table > "$sbom_table"
            
            # Count packages
            local package_count
            package_count=$(jq '.packages | length' "$sbom_json" 2>/dev/null || echo "0")
            log_info "SBOM contains $package_count packages"
        else
            log_warn "SBOM generation failed"
        fi
    else
        log_info "Syft not available - skipping SBOM generation"
    fi
}

# Run Grype vulnerability scan on SBOM
run_grype_scan() {
    if command -v grype &> /dev/null; then
        log_info "Running Grype vulnerability scan..."
        
        local grype_report="${SCAN_DIR}/grype_vulnerabilities.json"
        local grype_summary="${SCAN_DIR}/grype_summary.txt"
        
        # Scan using SBOM if available
        local sbom_file="${SCAN_DIR}/sbom.spdx.json"
        if [ -f "$sbom_file" ]; then
            grype_target="sbom:$sbom_file"
        else
            grype_target="$FULL_IMAGE_NAME"
        fi
        
        if grype "$grype_target" -o json > "$grype_report"; then
            log_success "Grype vulnerability scan completed"
            
            # Generate human-readable summary
            grype "$grype_target" > "$grype_summary"
            
            # Count critical vulnerabilities
            local grype_critical_count
            grype_critical_count=$(jq '[.matches[]? | select(.vulnerability.severity == "Critical")] | length' "$grype_report" 2>/dev/null || echo "0")
            
            if [ "$grype_critical_count" -gt 0 ]; then
                log_warn "Grype found $grype_critical_count critical vulnerabilities"
            else
                log_success "Grype scan passed"
            fi
        else
            log_warn "Grype vulnerability scan failed"
        fi
    else
        log_info "Grype not available - skipping additional vulnerability scan"
    fi
}

# Check container best practices
check_container_best_practices() {
    log_info "Checking container best practices..."
    
    local practices_report="${SCAN_DIR}/best_practices.txt"
    
    {
        echo "Container Best Practices Check"
        echo "============================="
        echo "Image: $FULL_IMAGE_NAME"
        echo "Scan Date: $(date)"
        echo ""
        
        # Check if image runs as non-root
        local user_info
        user_info=$(docker inspect "$FULL_IMAGE_NAME" --format '{{.Config.User}}' 2>/dev/null || echo "")
        if [ -z "$user_info" ] || [ "$user_info" = "root" ] || [ "$user_info" = "0" ]; then
            echo "❌ FAIL: Container runs as root user"
        else
            echo "✅ PASS: Container runs as non-root user ($user_info)"
        fi
        
        # Check for HEALTHCHECK
        local healthcheck
        healthcheck=$(docker inspect "$FULL_IMAGE_NAME" --format '{{.Config.Healthcheck}}' 2>/dev/null || echo "")
        if [ "$healthcheck" = "<nil>" ] || [ -z "$healthcheck" ]; then
            echo "⚠️  WARN: No HEALTHCHECK defined"
        else
            echo "✅ PASS: HEALTHCHECK defined"
        fi
        
        # Check image size
        local image_size
        image_size=$(docker images "$FULL_IMAGE_NAME" --format "{{.Size}}")
        echo "📏 Image size: $image_size"
        
        # Check layer count
        local layer_count
        layer_count=$(docker history "$FULL_IMAGE_NAME" --format "{{.ID}}" | wc -l)
        echo "🥞 Layer count: $layer_count"
        
        if [ "$layer_count" -gt 50 ]; then
            echo "⚠️  WARN: High number of layers ($layer_count) - consider optimization"
        fi
        
    } > "$practices_report"
    
    log_success "Best practices check completed"
}

# Generate consolidated security report
generate_security_report() {
    log_info "Generating consolidated security report..."
    
    local report_file="${SCAN_DIR}/security_report.md"
    
    {
        echo "# Container Security Scan Report"
        echo ""
        echo "**Image:** \`$FULL_IMAGE_NAME\`"
        echo "**Scan Date:** $(date)"
        echo "**Report ID:** scan_$TIMESTAMP"
        echo ""
        
        echo "## Executive Summary"
        echo ""
        
        # Get vulnerability counts
        if [ -f "${SCAN_DIR}/vulnerability_counts.txt" ]; then
            local vuln_summary
            vuln_summary=$(cat "${SCAN_DIR}/vulnerability_counts.txt")
            echo "**Vulnerabilities Found:** $vuln_summary"
        fi
        
        # Overall security status
        local security_status="✅ PASS"
        if [ -f "${SCAN_DIR}/trivy_vulnerabilities.json" ]; then
            local critical_count
            critical_count=$(jq '[.Results[]?.Vulnerabilities[]? | select(.Severity == "CRITICAL")] | length' "${SCAN_DIR}/trivy_vulnerabilities.json" 2>/dev/null || echo "0")
            if [ "$critical_count" -gt 0 ]; then
                security_status="❌ FAIL - Critical vulnerabilities found"
            fi
        fi
        
        echo "**Security Status:** $security_status"
        echo ""
        
        echo "## Scan Results"
        echo ""
        echo "### Vulnerability Scan"
        if [ -f "${SCAN_DIR}/trivy_summary.txt" ]; then
            echo '```'
            head -20 "${SCAN_DIR}/trivy_summary.txt"
            echo '```'
        fi
        
        echo ""
        echo "### Configuration Scan"
        if [ -f "${SCAN_DIR}/trivy_config_summary.txt" ]; then
            echo '```'
            head -10 "${SCAN_DIR}/trivy_config_summary.txt"
            echo '```'
        fi
        
        echo ""
        echo "### Secret Scan"
        if [ -f "${SCAN_DIR}/trivy_secrets_summary.txt" ]; then
            echo '```'
            cat "${SCAN_DIR}/trivy_secrets_summary.txt"
            echo '```'
        fi
        
        echo ""
        echo "### Best Practices"
        if [ -f "${SCAN_DIR}/best_practices.txt" ]; then
            echo '```'
            cat "${SCAN_DIR}/best_practices.txt"
            echo '```'
        fi
        
        echo ""
        echo "## Recommendations"
        echo ""
        echo "- Review and patch critical vulnerabilities"
        echo "- Address configuration issues if any"
        echo "- Ensure no secrets are embedded in the image"
        echo "- Follow container security best practices"
        echo "- Regular security scans in CI/CD pipeline"
        
        echo ""
        echo "---"
        echo "*Generated by Terragon Labs Container Security Scanner*"
        
    } > "$report_file"
    
    log_success "Security report generated: $report_file"
}

# Main execution
main() {
    log_info "Starting container security scan for $FULL_IMAGE_NAME"
    
    check_dependencies
    setup_reporting
    
    local exit_code=0
    
    # Run all scans
    if ! run_trivy_scan; then
        exit_code=1
    fi
    
    if ! run_trivy_config_scan; then
        exit_code=1
    fi
    
    if ! run_trivy_secret_scan; then
        exit_code=1
    fi
    
    generate_sbom
    run_grype_scan
    check_container_best_practices
    generate_security_report
    
    if [ $exit_code -eq 0 ]; then
        log_success "Container security scan completed successfully"
        log_info "Reports available in: $SCAN_DIR"
    else
        log_error "Container security scan found critical issues"
        log_info "Reports available in: $SCAN_DIR"
    fi
    
    exit $exit_code
}

# Show usage if no arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 <image_name> [image_tag]"
    echo ""
    echo "Example:"
    echo "  $0 agentic-startup-studio latest"
    echo "  $0 my-app v1.0.0"
    echo ""
    echo "This script performs comprehensive security scanning including:"
    echo "  - Vulnerability scanning with Trivy"
    echo "  - Configuration scanning"
    echo "  - Secret detection"
    echo "  - SBOM generation with Syft"
    echo "  - Additional vulnerability scanning with Grype"
    echo "  - Container best practices validation"
    exit 1
fi

main "$@"
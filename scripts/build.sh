#!/bin/bash

# =============================================================================
# Advanced Build Script for Agentic Startup Studio
# Supports multi-architecture builds, semantic versioning, and security scanning
# =============================================================================

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DATE="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
DEFAULT_TAG="latest"
DEFAULT_REGISTRY="ghcr.io/danieleschmidt"
DEFAULT_IMAGE_NAME="agentic-startup-studio"
DEFAULT_PLATFORMS="linux/amd64,linux/arm64"

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
Advanced Build Script for Agentic Startup Studio

Usage: $0 [OPTIONS]

OPTIONS:
    -t, --tag TAG                Docker image tag (default: ${DEFAULT_TAG})
    -r, --registry REGISTRY      Docker registry (default: ${DEFAULT_REGISTRY})
    -i, --image IMAGE            Image name (default: ${DEFAULT_IMAGE_NAME})
    -p, --platforms PLATFORMS    Target platforms (default: ${DEFAULT_PLATFORMS})
    -e, --environment ENV        Target environment (dev|staging|prod) (default: prod)
    --push                       Push image to registry
    --scan                       Run security scan after build
    --sbom                       Generate SBOM (Software Bill of Materials)
    --multi-arch                 Build multi-architecture images
    --cache-from CACHE          Use build cache from registry
    --cache-to CACHE            Push build cache to registry
    --no-cache                  Disable build cache
    --build-arg ARG=VALUE       Pass build argument
    --version                   Display version information
    -h, --help                  Display this help message

EXAMPLES:
    # Basic build
    $0 -t v1.0.0

    # Multi-architecture build with push
    $0 -t v1.0.0 --multi-arch --push

    # Development build with security scan
    $0 -t dev-latest -e dev --scan

    # Production build with SBOM and cache
    $0 -t v1.0.0 -e prod --sbom --push --cache-from ${DEFAULT_REGISTRY}/${DEFAULT_IMAGE_NAME}:cache

EOF
}

# Parse command line arguments
TAG="${DEFAULT_TAG}"
REGISTRY="${DEFAULT_REGISTRY}"
IMAGE_NAME="${DEFAULT_IMAGE_NAME}"
PLATFORMS="${DEFAULT_PLATFORMS}"
ENVIRONMENT="prod"
PUSH=false
SCAN=false
SBOM=false
MULTI_ARCH=false
USE_CACHE=true
CACHE_FROM=""
CACHE_TO=""
BUILD_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--tag)
            TAG="$2"
            shift 2
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -i|--image)
            IMAGE_NAME="$2"
            shift 2
            ;;
        -p|--platforms)
            PLATFORMS="$2"
            MULTI_ARCH=true
            shift 2
            ;;
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --push)
            PUSH=true
            shift
            ;;
        --scan)
            SCAN=true
            shift
            ;;
        --sbom)
            SBOM=true
            shift
            ;;
        --multi-arch)
            MULTI_ARCH=true
            shift
            ;;
        --cache-from)
            CACHE_FROM="$2"
            shift 2
            ;;
        --cache-to)
            CACHE_TO="$2"
            shift 2
            ;;
        --no-cache)
            USE_CACHE=false
            shift
            ;;
        --build-arg)
            BUILD_ARGS+=("--build-arg" "$2")
            shift 2
            ;;
        --version)
            echo "Agentic Startup Studio Build Script v1.0.0"
            exit 0
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

# Construct full image name
FULL_IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}"

# Get Git information for build metadata
get_git_info() {
    if git rev-parse --git-dir > /dev/null 2>&1; then
        GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")"
        GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")"
        GIT_TAG="$(git describe --tags --abbrev=0 2>/dev/null || echo "v0.0.0")"
        GIT_DIRTY=""
        if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then
            GIT_DIRTY="-dirty"
        fi
    else
        GIT_COMMIT="unknown"
        GIT_BRANCH="unknown" 
        GIT_TAG="v0.0.0"
        GIT_DIRTY=""
    fi
}

# Validate environment
validate_environment() {
    log_info "Validating build environment..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    # Check for multi-arch support if needed
    if [[ "${MULTI_ARCH}" == true ]]; then
        if ! docker buildx version &> /dev/null; then
            log_error "Docker Buildx is required for multi-architecture builds"
            exit 1
        fi
        
        # Create/use buildx builder
        if ! docker buildx inspect multiarch-builder &> /dev/null; then
            log_info "Creating multi-architecture builder..."
            docker buildx create --name multiarch-builder --driver docker-container --use
            docker buildx inspect --bootstrap
        else
            docker buildx use multiarch-builder
        fi
    fi
    
    # Validate project structure
    if [[ ! -f "${PROJECT_ROOT}/Dockerfile" ]]; then
        log_error "Dockerfile not found in project root"
        exit 1
    fi
    
    if [[ ! -f "${PROJECT_ROOT}/pyproject.toml" ]]; then
        log_error "pyproject.toml not found in project root"
        exit 1
    fi
    
    log_success "Environment validation completed"
}

# Pre-build checks
pre_build_checks() {
    log_info "Running pre-build checks..."
    
    cd "${PROJECT_ROOT}"
    
    # Check Python syntax
    log_info "Checking Python syntax..."
    if command -v python3 &> /dev/null; then
        find . -name "*.py" -not -path "./venv/*" -not -path "./.venv/*" -exec python3 -m py_compile {} \; || {
            log_error "Python syntax errors found"
            exit 1
        }
    fi
    
    # Run linting if available
    if command -v ruff &> /dev/null; then
        log_info "Running code linting..."
        ruff check . --exit-zero || log_warning "Linting issues found (non-blocking)"
    fi
    
    # Check for security issues if available
    if command -v bandit &> /dev/null; then
        log_info "Running security scan..."
        bandit -r pipeline/ core/ -f json -o /tmp/bandit-report.json --exit-zero || true
        if [[ -f /tmp/bandit-report.json ]]; then
            SECURITY_ISSUES=$(jq '.results | length' /tmp/bandit-report.json 2>/dev/null || echo "0")
            if [[ "${SECURITY_ISSUES}" -gt 0 ]]; then
                log_warning "Found ${SECURITY_ISSUES} potential security issues"
            fi
        fi
    fi
    
    log_success "Pre-build checks completed"
}

# Build Docker image
build_image() {
    log_info "Building Docker image..."
    
    cd "${PROJECT_ROOT}"
    get_git_info
    
    # Prepare build arguments
    local build_args=(
        "--build-arg" "BUILD_DATE=${BUILD_DATE}"
        "--build-arg" "VCS_REF=${GIT_COMMIT}${GIT_DIRTY}"
        "--build-arg" "VERSION=${TAG}"
        "--build-arg" "GIT_BRANCH=${GIT_BRANCH}"
        "--build-arg" "ENVIRONMENT=${ENVIRONMENT}"
    )
    
    # Add custom build args
    build_args+=("${BUILD_ARGS[@]}")
    
    # Prepare cache arguments
    local cache_args=()
    if [[ "${USE_CACHE}" == true ]]; then
        if [[ -n "${CACHE_FROM}" ]]; then
            cache_args+=("--cache-from" "type=registry,ref=${CACHE_FROM}")
        fi
        if [[ -n "${CACHE_TO}" ]]; then
            cache_args+=("--cache-to" "type=registry,ref=${CACHE_TO},mode=max")
        fi
    else
        cache_args=("--no-cache")
    fi
    
    # Build command
    if [[ "${MULTI_ARCH}" == true ]]; then
        local push_arg=""
        if [[ "${PUSH}" == true ]]; then
            push_arg="--push"
        else
            push_arg="--load"
        fi
        
        docker buildx build \
            --platform "${PLATFORMS}" \
            --tag "${FULL_IMAGE_NAME}:${TAG}" \
            --tag "${FULL_IMAGE_NAME}:latest" \
            "${build_args[@]}" \
            "${cache_args[@]}" \
            ${push_arg} \
            .
    else
        docker build \
            --tag "${FULL_IMAGE_NAME}:${TAG}" \
            --tag "${FULL_IMAGE_NAME}:latest" \
            "${build_args[@]}" \
            "${cache_args[@]}" \
            .
    fi
    
    log_success "Docker image built successfully: ${FULL_IMAGE_NAME}:${TAG}"
}

# Run security scan
run_security_scan() {
    if [[ "${SCAN}" == true ]]; then
        log_info "Running security scan on built image..."
        
        # Try different security scanners
        if command -v trivy &> /dev/null; then
            log_info "Running Trivy security scan..."
            trivy image --exit-code 0 --severity HIGH,CRITICAL "${FULL_IMAGE_NAME}:${TAG}" || {
                log_warning "Security vulnerabilities found in image"
            }
        elif command -v grype &> /dev/null; then
            log_info "Running Grype security scan..."
            grype "${FULL_IMAGE_NAME}:${TAG}" || {
                log_warning "Security vulnerabilities found in image"
            }
        else
            log_warning "No security scanner found (install trivy or grype)"
        fi
    fi
}

# Generate SBOM
generate_sbom() {
    if [[ "${SBOM}" == true ]]; then
        log_info "Generating Software Bill of Materials (SBOM)..."
        
        local sbom_dir="${PROJECT_ROOT}/sbom"
        mkdir -p "${sbom_dir}"
        
        if command -v syft &> /dev/null; then
            syft "${FULL_IMAGE_NAME}:${TAG}" -o spdx-json > "${sbom_dir}/sbom-${TAG}.spdx.json"
            syft "${FULL_IMAGE_NAME}:${TAG}" -o cyclonedx-json > "${sbom_dir}/sbom-${TAG}.cyclonedx.json"
            log_success "SBOM generated in ${sbom_dir}/"
        elif command -v docker &> /dev/null && docker buildx imagetools inspect "${FULL_IMAGE_NAME}:${TAG}" --format '{{json .}}' &> /dev/null; then
            # Generate basic SBOM from image layers
            docker history "${FULL_IMAGE_NAME}:${TAG}" --format "table {{.CreatedBy}}" --no-trunc > "${sbom_dir}/layers-${TAG}.txt"
            log_info "Basic layer information saved to ${sbom_dir}/layers-${TAG}.txt"
        else
            log_warning "No SBOM generator found (install syft)"
        fi
    fi
}

# Push image to registry
push_image() {
    if [[ "${PUSH}" == true ]] && [[ "${MULTI_ARCH}" == false ]]; then
        log_info "Pushing image to registry..."
        
        docker push "${FULL_IMAGE_NAME}:${TAG}"
        docker push "${FULL_IMAGE_NAME}:latest"
        
        log_success "Image pushed to registry: ${FULL_IMAGE_NAME}:${TAG}"
    fi
}

# Generate build report
generate_build_report() {
    log_info "Generating build report..."
    
    local report_dir="${PROJECT_ROOT}/build-reports"
    mkdir -p "${report_dir}"
    
    local report_file="${report_dir}/build-report-${TAG}-$(date +%Y%m%d-%H%M%S).json"
    
    # Get image information
    local image_id=""
    local image_size=""
    if docker image inspect "${FULL_IMAGE_NAME}:${TAG}" &> /dev/null; then
        image_id=$(docker image inspect "${FULL_IMAGE_NAME}:${TAG}" --format '{{.Id}}')
        image_size=$(docker image inspect "${FULL_IMAGE_NAME}:${TAG}" --format '{{.Size}}')
    fi
    
    cat > "${report_file}" << EOF
{
  "build_info": {
    "timestamp": "${BUILD_DATE}",
    "tag": "${TAG}",
    "image": "${FULL_IMAGE_NAME}",
    "environment": "${ENVIRONMENT}",
    "platforms": "${PLATFORMS}",
    "multi_arch": ${MULTI_ARCH}
  },
  "git_info": {
    "commit": "${GIT_COMMIT}${GIT_DIRTY}",
    "branch": "${GIT_BRANCH}",
    "tag": "${GIT_TAG}"
  },
  "image_info": {
    "id": "${image_id}",
    "size_bytes": ${image_size:-0}
  },
  "build_options": {
    "push": ${PUSH},
    "security_scan": ${SCAN},
    "sbom_generated": ${SBOM},
    "cache_used": ${USE_CACHE}
  }
}
EOF
    
    log_success "Build report saved to ${report_file}"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    rm -f /tmp/bandit-report.json
}

# Main execution
main() {
    log_info "Starting Agentic Startup Studio build process..."
    log_info "Target: ${FULL_IMAGE_NAME}:${TAG}"
    log_info "Environment: ${ENVIRONMENT}"
    log_info "Platforms: ${PLATFORMS}"
    
    # Set trap for cleanup
    trap cleanup EXIT
    
    # Execute build pipeline
    validate_environment
    pre_build_checks
    build_image
    run_security_scan
    generate_sbom
    push_image
    generate_build_report
    
    log_success "Build process completed successfully!"
    log_info "Image: ${FULL_IMAGE_NAME}:${TAG}"
    
    if [[ "${PUSH}" == true ]]; then
        log_info "Image pushed to registry and ready for deployment"
    else
        log_info "Image built locally. Use --push to push to registry"
    fi
}

# Execute main function
main "$@"
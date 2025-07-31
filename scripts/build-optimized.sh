#!/bin/bash
# =============================================================================
# Advanced Docker Build Script for Agentic Startup Studio
# Supports multi-architecture builds, caching, security scanning, and optimization
# =============================================================================

set -euo pipefail

# Configuration
REGISTRY="${REGISTRY:-terragon}"
IMAGE_NAME="${IMAGE_NAME:-agentic-startup-studio}"
VERSION="${VERSION:-$(git rev-parse --short HEAD 2>/dev/null || echo 'latest')}"
BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
VCS_REF="${VCS_REF:-$(git rev-parse HEAD 2>/dev/null || echo 'unknown')}"

# Build options
BUILD_TARGET="${BUILD_TARGET:-production}"
PLATFORM="${PLATFORM:-linux/amd64,linux/arm64}"
PUSH="${PUSH:-false}"
CACHE="${CACHE:-true}"
SECURITY_SCAN="${SECURITY_SCAN:-true}"
OPTIMIZE="${OPTIMIZE:-true}"

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

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check Docker Buildx
    if ! docker buildx version &> /dev/null; then
        log_error "Docker Buildx is not available"
        exit 1
    fi
    
    # Check if we can build multi-arch
    if [[ "$PLATFORM" == *","* ]]; then
        log_info "Multi-architecture build requested: $PLATFORM"
        if ! docker buildx ls | grep -q "docker-container"; then
            log_info "Creating buildx builder for multi-arch support..."
            docker buildx create --name multiarch --driver docker-container --use || true
            docker buildx inspect --bootstrap
        fi
    fi
    
    log_success "Prerequisites check passed"
}

# Build arguments
setup_build_args() {
    BUILD_ARGS=(
        --build-arg "BUILD_DATE=${BUILD_DATE}"
        --build-arg "VCS_REF=${VCS_REF}"
        --build-arg "VERSION=${VERSION}"
        --build-arg "BUILDKIT_INLINE_CACHE=1"
    )
    
    # Add platform support
    if [[ -n "$PLATFORM" ]]; then
        BUILD_ARGS+=(--platform "$PLATFORM")
    fi
    
    # Add cache configuration
    if [[ "$CACHE" == "true" ]]; then
        CACHE_FROM="type=registry,ref=${REGISTRY}/${IMAGE_NAME}:cache"
        CACHE_TO="type=registry,ref=${REGISTRY}/${IMAGE_NAME}:cache,mode=max"
        BUILD_ARGS+=(
            --cache-from "$CACHE_FROM"
            --cache-to "$CACHE_TO"
        )
        log_info "Build cache enabled"
    fi
    
    # Tags
    TAGS=(
        -t "${REGISTRY}/${IMAGE_NAME}:${VERSION}"
        -t "${REGISTRY}/${IMAGE_NAME}:latest"
    )
    
    # Add build target specific tags
    if [[ "$BUILD_TARGET" != "production" ]]; then
        TAGS+=(-t "${REGISTRY}/${IMAGE_NAME}:${VERSION}-${BUILD_TARGET}")
        TAGS+=(-t "${REGISTRY}/${IMAGE_NAME}:latest-${BUILD_TARGET}")
    fi
}

# Security scanning
run_security_scan() {
    if [[ "$SECURITY_SCAN" != "true" ]]; then
        log_info "Security scanning disabled"
        return 0
    fi
    
    log_info "Running security scan..."
    
    # Build security scan stage
    docker buildx build \
        --target security-scan \
        --build-arg "BUILD_DATE=${BUILD_DATE}" \
        --build-arg "VCS_REF=${VCS_REF}" \
        --build-arg "VERSION=${VERSION}" \
        --load \
        -t "${REGISTRY}/${IMAGE_NAME}:security-scan" \
        .
    
    # Extract security reports
    log_info "Extracting security scan reports..."
    docker create --name temp-security "${REGISTRY}/${IMAGE_NAME}:security-scan" || true
    docker cp temp-security:/tmp/safety-report.json ./security-reports/safety-report.json 2>/dev/null || log_warning "Safety report not found"
    docker cp temp-security:/tmp/bandit-report.json ./security-reports/bandit-report.json 2>/dev/null || log_warning "Bandit report not found"
    docker rm temp-security 2>/dev/null || true
    
    log_success "Security scan completed"
}

# Optimize image
optimize_image() {
    if [[ "$OPTIMIZE" != "true" ]]; then
        log_info "Image optimization disabled"
        return 0
    fi
    
    log_info "Optimizing Docker image..."
    
    # Use dive to analyze layers (if available)
    if command -v dive &> /dev/null; then
        log_info "Analyzing image layers with dive..."
        dive "${REGISTRY}/${IMAGE_NAME}:${VERSION}" --ci
    fi
    
    log_success "Image optimization completed"
}

# Build image
build_image() {
    log_info "Building Docker image..."
    log_info "Target: $BUILD_TARGET"
    log_info "Version: $VERSION"
    log_info "Platform: $PLATFORM"
    
    # Create security reports directory
    mkdir -p security-reports
    
    # Run security scan first
    run_security_scan
    
    # Main build
    BUILD_CMD="docker buildx build"
    BUILD_CMD+=" --target $BUILD_TARGET"
    BUILD_CMD+=" ${BUILD_ARGS[*]}"
    BUILD_CMD+=" ${TAGS[*]}"
    
    if [[ "$PUSH" == "true" ]]; then
        BUILD_CMD+=" --push"
        log_info "Image will be pushed to registry"
    else
        BUILD_CMD+=" --load"
        log_info "Image will be loaded locally"
    fi
    
    BUILD_CMD+=" ."
    
    log_info "Executing: $BUILD_CMD"
    eval "$BUILD_CMD"
    
    log_success "Docker image built successfully"
    
    # Show image info
    if [[ "$PUSH" != "true" ]]; then
        log_info "Image information:"
        docker images "${REGISTRY}/${IMAGE_NAME}" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
        
        # Optimize image
        optimize_image
    fi
}

# Test image
test_image() {
    if [[ "$PUSH" == "true" ]]; then
        log_info "Skipping local tests for push build"
        return 0
    fi
    
    log_info "Testing built image..."
    
    # Basic smoke test
    log_info "Running smoke test..."
    if docker run --rm --name test-container \
        "${REGISTRY}/${IMAGE_NAME}:${VERSION}" \
        python -c "import sys; print(f'Python {sys.version}'); exit(0)" 2>/dev/null; then
        log_success "Smoke test passed"
    else
        log_error "Smoke test failed"
        return 1
    fi
    
    # Health check test
    log_info "Testing health endpoint..."
    CONTAINER_ID=$(docker run -d --name health-test \
        -p 8000:8000 \
        "${REGISTRY}/${IMAGE_NAME}:${VERSION}")
    
    # Wait for container to start
    sleep 10
    
    if curl -f -s http://localhost:8000/health > /dev/null; then
        log_success "Health check passed"
    else
        log_error "Health check failed"
        docker logs "$CONTAINER_ID"
        docker stop "$CONTAINER_ID" && docker rm "$CONTAINER_ID"
        return 1
    fi
    
    docker stop "$CONTAINER_ID" && docker rm "$CONTAINER_ID"
    log_success "All tests passed"
}

# Build different variants
build_variants() {
    local variants=("production" "development" "testing" "model-serving" "worker" "monitoring")
    
    if [[ "$BUILD_TARGET" == "all" ]]; then
        log_info "Building all variants..."
        for variant in "${variants[@]}"; do
            log_info "Building variant: $variant"
            BUILD_TARGET="$variant" build_image
        done
    else
        build_image
    fi
}

# Generate build report
generate_report() {
    log_info "Generating build report..."
    
    REPORT_FILE="build-report-${VERSION}.json"
    cat > "$REPORT_FILE" << EOF
{
  "build_info": {
    "image_name": "${REGISTRY}/${IMAGE_NAME}",
    "version": "${VERSION}",
    "build_date": "${BUILD_DATE}",
    "vcs_ref": "${VCS_REF}",
    "build_target": "${BUILD_TARGET}",
    "platform": "${PLATFORM}",
    "pushed": ${PUSH}
  },
  "build_options": {
    "cache_enabled": ${CACHE},
    "security_scan": ${SECURITY_SCAN},
    "optimization": ${OPTIMIZE}
  },
  "security_reports": {
    "safety_report": "security-reports/safety-report.json",
    "bandit_report": "security-reports/bandit-report.json"
  }
}
EOF
    
    log_success "Build report generated: $REPORT_FILE"
}

# Main execution
main() {
    log_info "Starting advanced Docker build process..."
    log_info "Image: ${REGISTRY}/${IMAGE_NAME}:${VERSION}"
    
    check_prerequisites
    setup_build_args
    build_variants
    test_image
    generate_report
    
    log_success "Build process completed successfully!"
    
    if [[ "$PUSH" == "true" ]]; then
        log_info "Image pushed to: ${REGISTRY}/${IMAGE_NAME}:${VERSION}"
    else
        log_info "Image available locally: ${REGISTRY}/${IMAGE_NAME}:${VERSION}"
        log_info "To run: docker run -p 8000:8000 ${REGISTRY}/${IMAGE_NAME}:${VERSION}"
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --target)
            BUILD_TARGET="$2"
            shift 2
            ;;
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        --push)
            PUSH="true"
            shift
            ;;
        --no-cache)
            CACHE="false"
            shift
            ;;
        --no-security-scan)
            SECURITY_SCAN="false"
            shift
            ;;
        --no-optimize)
            OPTIMIZE="false"
            shift
            ;;
        --registry)
            REGISTRY="$2"
            shift 2
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --target TARGET      Build target (default: production)"
            echo "  --platform PLATFORM  Target platform (default: linux/amd64,linux/arm64)"
            echo "  --push               Push to registry"
            echo "  --no-cache           Disable build cache"
            echo "  --no-security-scan   Disable security scanning"
            echo "  --no-optimize        Disable image optimization"
            echo "  --registry REGISTRY  Docker registry (default: terragon)"
            echo "  --version VERSION    Image version (default: git commit hash)"
            echo "  --help               Show this help"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main function
main
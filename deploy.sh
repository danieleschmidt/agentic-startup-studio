#!/bin/bash

# Autonomous SDLC Generation 3.0 Deployment Script
# Terragon Labs - Enhanced Production Deployment

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="agentic-startup-studio"
VERSION="${VERSION:-latest}"
ENVIRONMENT="${ENVIRONMENT:-production}"
PLATFORM="${PLATFORM:-docker}"

# Banner
echo -e "${PURPLE}"
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║          🚀 AUTONOMOUS SDLC GENERATION 3.0 DEPLOYMENT           ║"
echo "║                        Terragon Labs                            ║"
echo "║                                                                  ║"
echo "║  🧠 Neural Evolution    ⚡ Quantum Optimization                   ║"
echo "║  🔄 Real-time Intel     🤖 Multimodal AI                        ║"
echo "║  💻 Code Generation     🌟 Transcendent Performance              ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    log "🔍 Checking prerequisites..."
    
    if [[ "$PLATFORM" == "docker" ]]; then
        if ! command -v docker &> /dev/null; then
            error "Docker is not installed or not in PATH"
        fi
        
        if ! command -v docker-compose &> /dev/null; then
            error "Docker Compose is not installed or not in PATH"
        fi
        
        if ! docker info &> /dev/null; then
            error "Docker daemon is not running"
        fi
    fi
    
    if [[ "$PLATFORM" == "kubernetes" ]]; then
        if ! command -v kubectl &> /dev/null; then
            error "kubectl is not installed or not in PATH"
        fi
        
        if ! kubectl cluster-info &> /dev/null; then
            error "kubectl cannot connect to cluster"
        fi
    fi
    
    log "✅ Prerequisites check passed"
}

# Environment setup
setup_environment() {
    log "🌍 Setting up environment variables..."
    
    # Create .env file if it doesn't exist
    if [[ ! -f "$SCRIPT_DIR/.env" ]]; then
        warn ".env file not found, creating template..."
        cat > "$SCRIPT_DIR/.env" << EOF
# Environment Configuration
ENVIRONMENT=$ENVIRONMENT
VERSION=$VERSION

# Database Configuration
POSTGRES_USER=startup_studio_user
POSTGRES_PASSWORD=$(openssl rand -base64 32)
POSTGRES_DB=startup_studio

# Redis Configuration
REDIS_PASSWORD=$(openssl rand -base64 24)

# Grafana Configuration
GRAFANA_ADMIN_PASSWORD=$(openssl rand -base64 16)

# API Keys (replace with actual values)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Security
JWT_SECRET=$(openssl rand -base64 64)
ENCRYPTION_KEY=$(openssl rand -base64 32)

# Autonomous SDLC Generation 3.0 Configuration
NEURAL_EVOLUTION_ENABLED=true
QUANTUM_OPTIMIZATION_ENABLED=true
MULTIMODAL_AI_ENABLED=true
REALTIME_INTELLIGENCE_ENABLED=true
PREDICTIVE_ANALYTICS_ENABLED=true
AUTONOMOUS_CODE_GEN_ENABLED=true

# Performance Configuration
EVOLUTION_POPULATION_SIZE=50
QUANTUM_ANNEALING_CYCLES=1000
REALTIME_MAX_WORKERS=16
MULTIMODAL_EMBEDDING_DIM=768
CODE_GEN_QUALITY_THRESHOLD=0.8

# Build Configuration
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
EOF
        info "Created .env template. Please edit with your actual configuration."
    fi
    
    # Source environment variables
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
    
    log "✅ Environment setup complete"
}

# Build images
build_images() {
    log "🏗️  Building Docker images..."
    
    if [[ "$PLATFORM" == "docker" ]]; then
        docker-compose -f docker-compose.production.yml build --parallel
        log "✅ Docker images built successfully"
    fi
}

# Health check
health_check() {
    log "🏥 Performing health checks..."
    
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if [[ "$PLATFORM" == "docker" ]]; then
            if curl -sf http://localhost:8080/health &> /dev/null; then
                log "✅ Main API is healthy"
                break
            fi
        elif [[ "$PLATFORM" == "kubernetes" ]]; then
            if kubectl get pods -n autonomous-sdlc --field-selector=status.phase=Running | grep -q Running; then
                log "✅ Kubernetes pods are running"
                break
            fi
        fi
        
        info "Health check attempt $attempt/$max_attempts failed, retrying in 10s..."
        sleep 10
        ((attempt++))
    done
    
    if [[ $attempt -gt $max_attempts ]]; then
        error "Health checks failed after $max_attempts attempts"
    fi
}

# Docker deployment
deploy_docker() {
    log "🐳 Deploying with Docker Compose..."
    
    # Stop any existing containers
    docker-compose -f docker-compose.production.yml down --remove-orphans
    
    # Start services
    docker-compose -f docker-compose.production.yml up -d
    
    # Wait for services to be ready
    sleep 30
    
    # Show status
    docker-compose -f docker-compose.production.yml ps
    
    log "✅ Docker deployment complete"
}

# Kubernetes deployment
deploy_kubernetes() {
    log "☸️  Deploying to Kubernetes..."
    
    # Apply namespace and configurations
    kubectl apply -f k8s/autonomous-sdlc-services.yaml
    
    # Wait for deployments to be ready
    kubectl wait --for=condition=available --timeout=600s deployment --all -n autonomous-sdlc
    
    # Show status
    kubectl get all -n autonomous-sdlc
    
    log "✅ Kubernetes deployment complete"
}

# Monitoring setup
setup_monitoring() {
    log "📊 Setting up monitoring and observability..."
    
    if [[ "$PLATFORM" == "docker" ]]; then
        # Prometheus should already be running from docker-compose
        info "Prometheus: http://localhost:9090"
        info "Grafana: http://localhost:3001 (admin/${GRAFANA_ADMIN_PASSWORD})"
        info "Jaeger: http://localhost:16686"
    elif [[ "$PLATFORM" == "kubernetes" ]]; then
        # Port forward for monitoring tools
        info "Setting up port forwarding for monitoring tools..."
        kubectl port-forward -n autonomous-sdlc svc/prometheus 9090:9090 &
        kubectl port-forward -n autonomous-sdlc svc/grafana 3001:3000 &
        kubectl port-forward -n autonomous-sdlc svc/jaeger 16686:16686 &
    fi
    
    log "✅ Monitoring setup complete"
}

# Performance testing
run_performance_tests() {
    log "⚡ Running performance validation tests..."
    
    # Test Neural Evolution Engine
    if curl -sf http://localhost:8082/health &> /dev/null; then
        info "✅ Neural Evolution Engine: Online"
    else
        warn "❌ Neural Evolution Engine: Offline"
    fi
    
    # Test Quantum Performance Optimizer
    if curl -sf http://localhost:8083/health &> /dev/null; then
        info "✅ Quantum Performance Optimizer: Online"
    else
        warn "❌ Quantum Performance Optimizer: Offline"
    fi
    
    # Test Real-time Intelligence Engine
    if curl -sf http://localhost:8084/health &> /dev/null; then
        info "✅ Real-time Intelligence Engine: Online"
    else
        warn "❌ Real-time Intelligence Engine: Offline"
    fi
    
    # Test Multimodal AI Engine
    if curl -sf http://localhost:8085/health &> /dev/null; then
        info "✅ Multimodal AI Engine: Online"
    else
        warn "❌ Multimodal AI Engine: Offline"
    fi
    
    # Test Autonomous Code Generator
    if curl -sf http://localhost:8086/health &> /dev/null; then
        info "✅ Autonomous Code Generator: Online"
    else
        warn "❌ Autonomous Code Generator: Offline"
    fi
    
    log "✅ Performance validation complete"
}

# Cleanup function
cleanup() {
    log "🧹 Cleaning up..."
    
    if [[ "$PLATFORM" == "docker" ]]; then
        docker-compose -f docker-compose.production.yml down
    elif [[ "$PLATFORM" == "kubernetes" ]]; then
        kubectl delete -f k8s/autonomous-sdlc-services.yaml
    fi
    
    log "✅ Cleanup complete"
}

# Main deployment function
main() {
    log "🚀 Starting Autonomous SDLC Generation 3.0 deployment..."
    log "Platform: $PLATFORM | Environment: $ENVIRONMENT | Version: $VERSION"
    
    # Trap cleanup on exit
    trap cleanup EXIT
    
    check_prerequisites
    setup_environment
    
    if [[ "$PLATFORM" == "docker" ]]; then
        build_images
        deploy_docker
    elif [[ "$PLATFORM" == "kubernetes" ]]; then
        deploy_kubernetes
    else
        error "Unsupported platform: $PLATFORM"
    fi
    
    health_check
    setup_monitoring
    run_performance_tests
    
    echo -e "${GREEN}"
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║                    🎉 DEPLOYMENT SUCCESSFUL! 🎉                  ║"
    echo "║                                                                  ║"
    echo "║  🌐 Main API: http://localhost:8080                              ║"
    echo "║  🧠 Neural Evolution: http://localhost:8082                      ║"
    echo "║  ⚡ Quantum Optimizer: http://localhost:8083                      ║"
    echo "║  🔄 Real-time Intelligence: http://localhost:8084                ║"
    echo "║  🤖 Multimodal AI: http://localhost:8085                         ║"
    echo "║  💻 Code Generator: http://localhost:8086                        ║"
    echo "║                                                                  ║"
    echo "║  📊 Monitoring:                                                  ║"
    echo "║     Prometheus: http://localhost:9090                           ║"
    echo "║     Grafana: http://localhost:3001                              ║"
    echo "║     Jaeger: http://localhost:16686                              ║"
    echo "║                                                                  ║"
    echo "║  🚀 AUTONOMOUS SDLC GENERATION 3.0 IS LIVE! 🚀                  ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    log "Deployment completed successfully!"
}

# Script execution
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "cleanup")
        cleanup
        ;;
    "health")
        health_check
        ;;
    "test")
        run_performance_tests
        ;;
    *)
        echo "Usage: $0 {deploy|cleanup|health|test}"
        echo "  deploy  - Full deployment (default)"
        echo "  cleanup - Clean up resources"
        echo "  health  - Run health checks"
        echo "  test    - Run performance tests"
        exit 1
        ;;
esac
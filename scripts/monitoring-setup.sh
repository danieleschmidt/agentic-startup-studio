#!/bin/bash

# =============================================================================
# Monitoring Stack Setup Script for Agentic Startup Studio
# Configures Prometheus, Grafana, AlertManager, and related monitoring tools
# =============================================================================

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MONITORING_DIR="${PROJECT_ROOT}/monitoring"

# Default configuration
PROMETHEUS_PORT=${PROMETHEUS_PORT:-9090}
GRAFANA_PORT=${GRAFANA_PORT:-3000}
ALERTMANAGER_PORT=${ALERTMANAGER_PORT:-9093}
JAEGER_PORT=${JAEGER_PORT:-16686}

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
Monitoring Stack Setup Script for Agentic Startup Studio

Usage: $0 [OPTIONS] [COMMAND]

COMMANDS:
    setup           Setup complete monitoring stack
    start           Start monitoring services
    stop            Stop monitoring services
    restart         Restart monitoring services
    status          Show status of monitoring services
    configure       Configure monitoring components
    dashboards      Install Grafana dashboards
    alerts          Configure alerting rules
    validate        Validate monitoring configuration
    clean           Clean up monitoring data

OPTIONS:
    --prometheus-port PORT    Prometheus port (default: 9090)
    --grafana-port PORT       Grafana port (default: 3000)
    --alertmanager-port PORT  AlertManager port (default: 9093)
    --data-retention DAYS     Data retention period (default: 15)
    --environment ENV         Environment (dev|staging|prod) (default: dev)
    --enable-tracing          Enable distributed tracing with Jaeger
    --enable-logging          Enable log aggregation with Loki
    --dry-run                 Show what would be done without making changes
    -h, --help                Display this help message

EXAMPLES:
    # Setup complete monitoring stack
    $0 setup --environment prod --enable-tracing

    # Configure and start monitoring
    $0 configure && $0 start

    # Install Grafana dashboards
    $0 dashboards

EOF
}

# Parse command line arguments
COMMAND=""
ENVIRONMENT="dev"
DATA_RETENTION="15d"
ENABLE_TRACING=false
ENABLE_LOGGING=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        setup|start|stop|restart|status|configure|dashboards|alerts|validate|clean)
            COMMAND="$1"
            shift
            ;;
        --prometheus-port)
            PROMETHEUS_PORT="$2"
            shift 2
            ;;
        --grafana-port)
            GRAFANA_PORT="$2"
            shift 2
            ;;
        --alertmanager-port)
            ALERTMANAGER_PORT="$2"
            shift 2
            ;;
        --data-retention)
            DATA_RETENTION="$2"
            shift 2
            ;;
        --environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --enable-tracing)
            ENABLE_TRACING=true
            shift
            ;;
        --enable-logging)
            ENABLE_LOGGING=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
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

# Validate command
if [[ -z "${COMMAND}" ]]; then
    log_error "Command is required"
    usage
    exit 1
fi

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check if ports are available
    local ports=("${PROMETHEUS_PORT}" "${GRAFANA_PORT}" "${ALERTMANAGER_PORT}")
    if [[ "${ENABLE_TRACING}" == true ]]; then
        ports+=("${JAEGER_PORT}")
    fi
    
    for port in "${ports[@]}"; do
        if netstat -ln | grep -q ":${port} "; then
            log_warning "Port ${port} is already in use"
        fi
    done
    
    log_success "Prerequisites check completed"
}

# Configure Prometheus
configure_prometheus() {
    log_info "Configuring Prometheus..."
    
    local prometheus_config="${MONITORING_DIR}/prometheus.yml"
    
    if [[ "${DRY_RUN}" == true ]]; then
        log_info "DRY RUN: Would configure Prometheus at ${prometheus_config}"
        return
    fi
    
    # Update Prometheus configuration for environment
    cat > "${prometheus_config}" << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    environment: '${ENVIRONMENT}'
    cluster: 'agentic-startup-studio'

rule_files:
  - "alerts.yml"
  - "alerts-enhanced.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - 'alertmanager:${ALERTMANAGER_PORT}'

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:${PROMETHEUS_PORT}']

  - job_name: 'agentic-startup-studio-api'
    static_configs:
      - targets: ['host.docker.internal:8000']
    scrape_interval: 10s
    scrape_timeout: 5s
    metrics_path: '/metrics'
    
  - job_name: 'agentic-startup-studio-health'
    static_configs:
      - targets: ['host.docker.internal:8080']
    scrape_interval: 30s
    metrics_path: '/health/metrics'

  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['postgres-exporter:9187']
    scrape_interval: 30s

  - job_name: 'redis-exporter'
    static_configs:
      - targets: ['redis-exporter:9121']
    scrape_interval: 30s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
    scrape_interval: 30s

  - job_name: 'blackbox-exporter'
    static_configs:
      - targets: ['blackbox-exporter:9115']
    scrape_interval: 60s
EOF

    if [[ "${ENABLE_TRACING}" == true ]]; then
        cat >> "${prometheus_config}" << EOF

  - job_name: 'jaeger'
    static_configs:
      - targets: ['jaeger:14269']
    scrape_interval: 30s
EOF
    fi

    log_success "Prometheus configured"
}

# Configure Grafana
configure_grafana() {
    log_info "Configuring Grafana..."
    
    if [[ "${DRY_RUN}" == true ]]; then
        log_info "DRY RUN: Would configure Grafana"
        return
    fi
    
    # Create Grafana provisioning directories
    mkdir -p "${MONITORING_DIR}/grafana/provisioning/datasources"
    mkdir -p "${MONITORING_DIR}/grafana/provisioning/dashboards"
    
    # Configure datasources
    cat > "${MONITORING_DIR}/grafana/provisioning/datasources/prometheus.yml" << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:${PROMETHEUS_PORT}
    isDefault: true
    editable: true
    
  - name: AlertManager
    type: alertmanager
    access: proxy
    url: http://alertmanager:${ALERTMANAGER_PORT}
    editable: true
EOF

    if [[ "${ENABLE_TRACING}" == true ]]; then
        cat >> "${MONITORING_DIR}/grafana/provisioning/datasources/prometheus.yml" << EOF
    
  - name: Jaeger
    type: jaeger
    access: proxy
    url: http://jaeger:${JAEGER_PORT}
    editable: true
EOF
    fi

    if [[ "${ENABLE_LOGGING}" == true ]]; then
        cat >> "${MONITORING_DIR}/grafana/provisioning/datasources/prometheus.yml" << EOF
    
  - name: Loki
    type: loki
    access: proxy
    url: http://loki:3100
    editable: true
EOF
    fi

    # Configure dashboard provisioning
    cat > "${MONITORING_DIR}/grafana/provisioning/dashboards/dashboards.yml" << EOF
apiVersion: 1

providers:
  - name: 'Agentic Startup Studio'
    orgId: 1
    folder: 'Agentic Studio'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
EOF

    log_success "Grafana configured"
}

# Configure AlertManager
configure_alertmanager() {
    log_info "Configuring AlertManager..."
    
    if [[ "${DRY_RUN}" == true ]]; then
        log_info "DRY RUN: Would configure AlertManager"
        return
    fi
    
    local alertmanager_config="${MONITORING_DIR}/alertmanager.yml"
    
    cat > "${alertmanager_config}" << EOF
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@terragonlabs.com'
  
route:
  group_by: ['alertname', 'environment']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 12h
  receiver: 'web.hook'
  routes:
  - match:
      severity: critical
    receiver: 'critical-alerts'
    group_wait: 10s
    repeat_interval: 5m
  - match:
      severity: warning
    receiver: 'warning-alerts'
    repeat_interval: 30m

receivers:
- name: 'web.hook'
  webhook_configs:
  - url: 'http://localhost:5001/webhook'
    send_resolved: true

- name: 'critical-alerts'
  webhook_configs:
  - url: 'http://localhost:5001/webhook/critical'
    send_resolved: true
  slack_configs:
  - api_url: '\${SLACK_WEBHOOK_URL}'
    channel: '#critical-alerts'
    title: 'Critical Alert - \${ENVIRONMENT}'
    text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'

- name: 'warning-alerts'
  slack_configs:
  - api_url: '\${SLACK_WEBHOOK_URL}'
    channel: '#alerts'
    title: 'Warning Alert - \${ENVIRONMENT}'
    text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'instance']
EOF

    log_success "AlertManager configured"
}

# Generate Docker Compose for monitoring
generate_docker_compose() {
    log_info "Generating monitoring Docker Compose..."
    
    local compose_file="${MONITORING_DIR}/docker-compose.monitoring.yml"
    
    if [[ "${DRY_RUN}" == true ]]; then
        log_info "DRY RUN: Would generate ${compose_file}"
        return
    fi
    
    cat > "${compose_file}" << EOF
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "${PROMETHEUS_PORT}:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - ./alerts.yml:/etc/prometheus/alerts.yml
      - ./alerts-enhanced.yml:/etc/prometheus/alerts-enhanced.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=${DATA_RETENTION}'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
      - '--web.enable-admin-api'
    restart: unless-stopped
    networks:
      - monitoring

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "${GRAFANA_PORT}:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning
      - ./grafana/dashboards:/var/lib/grafana/dashboards
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=\${GRAFANA_ADMIN_PASSWORD:-admin}
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_INSTALL_PLUGINS=grafana-piechart-panel,grafana-worldmap-panel
    restart: unless-stopped
    networks:
      - monitoring

  alertmanager:
    image: prom/alertmanager:latest
    container_name: alertmanager
    ports:
      - "${ALERTMANAGER_PORT}:9093"
    volumes:
      - ./alertmanager.yml:/etc/alertmanager/alertmanager.yml
      - alertmanager_data:/alertmanager
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'
      - '--storage.path=/alertmanager'
      - '--web.external-url=http://localhost:${ALERTMANAGER_PORT}'
    restart: unless-stopped
    networks:
      - monitoring

  node-exporter:
    image: prom/node-exporter:latest
    container_name: node-exporter
    ports:
      - "9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.rootfs=/rootfs'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
    restart: unless-stopped
    networks:
      - monitoring

  postgres-exporter:
    image: prometheuscommunity/postgres-exporter:latest
    container_name: postgres-exporter
    ports:
      - "9187:9187"
    environment:
      - DATA_SOURCE_NAME=postgresql://studio:studio@postgres:5432/studio?sslmode=disable
    restart: unless-stopped
    networks:
      - monitoring

  redis-exporter:
    image: oliver006/redis_exporter:latest
    container_name: redis-exporter
    ports:
      - "9121:9121"
    environment:
      - REDIS_ADDR=redis://redis:6379
    restart: unless-stopped
    networks:
      - monitoring

  blackbox-exporter:
    image: prom/blackbox-exporter:latest
    container_name: blackbox-exporter
    ports:
      - "9115:9115"
    volumes:
      - ./blackbox-config.yml:/etc/blackbox_exporter/config.yml
    restart: unless-stopped
    networks:
      - monitoring
EOF

    if [[ "${ENABLE_TRACING}" == true ]]; then
        cat >> "${compose_file}" << EOF

  jaeger:
    image: jaegertracing/all-in-one:latest
    container_name: jaeger
    ports:
      - "${JAEGER_PORT}:16686"
      - "14268:14268"
      - "14269:14269"
    environment:
      - COLLECTOR_OTLP_ENABLED=true
    restart: unless-stopped
    networks:
      - monitoring
EOF
    fi

    if [[ "${ENABLE_LOGGING}" == true ]]; then
        cat >> "${compose_file}" << EOF

  loki:
    image: grafana/loki:latest
    container_name: loki
    ports:
      - "3100:3100"
    volumes:
      - ./loki-config.yml:/etc/loki/local-config.yaml
      - loki_data:/loki
    restart: unless-stopped
    networks:
      - monitoring

  promtail:
    image: grafana/promtail:latest
    container_name: promtail
    volumes:
      - ./promtail-config.yml:/etc/promtail/config.yml
      - /var/log:/var/log:ro
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
    restart: unless-stopped
    networks:
      - monitoring
EOF
    fi

    cat >> "${compose_file}" << EOF

volumes:
  prometheus_data:
  grafana_data:
  alertmanager_data:
EOF

    if [[ "${ENABLE_LOGGING}" == true ]]; then
        cat >> "${compose_file}" << EOF
  loki_data:
EOF
    fi

    cat >> "${compose_file}" << EOF

networks:
  monitoring:
    driver: bridge
EOF

    log_success "Docker Compose configuration generated"
}

# Setup monitoring stack
setup_monitoring() {
    log_info "Setting up monitoring stack..."
    
    check_prerequisites
    configure_prometheus
    configure_grafana
    configure_alertmanager
    generate_docker_compose
    
    if [[ "${DRY_RUN}" == false ]]; then
        cd "${MONITORING_DIR}"
        docker-compose -f docker-compose.monitoring.yml pull
    fi
    
    log_success "Monitoring stack setup completed"
}

# Start monitoring services
start_monitoring() {
    log_info "Starting monitoring services..."
    
    if [[ "${DRY_RUN}" == true ]]; then
        log_info "DRY RUN: Would start monitoring services"
        return
    fi
    
    cd "${MONITORING_DIR}"
    docker-compose -f docker-compose.monitoring.yml up -d
    
    # Wait for services to be ready
    sleep 10
    
    log_info "Waiting for services to be ready..."
    local services=("prometheus:${PROMETHEUS_PORT}" "grafana:${GRAFANA_PORT}" "alertmanager:${ALERTMANAGER_PORT}")
    
    for service in "${services[@]}"; do
        local name="${service%:*}"
        local port="${service#*:}"
        
        for i in {1..30}; do
            if curl -sf "http://localhost:${port}" >/dev/null 2>&1; then
                log_success "${name} is ready on port ${port}"
                break
            fi
            
            if [[ $i -eq 30 ]]; then
                log_warning "${name} not ready after 30 attempts"
            fi
            
            sleep 2
        done
    done
    
    log_success "Monitoring services started"
    log_info "Access URLs:"
    log_info "  Prometheus: http://localhost:${PROMETHEUS_PORT}"
    log_info "  Grafana: http://localhost:${GRAFANA_PORT} (admin/admin)"
    log_info "  AlertManager: http://localhost:${ALERTMANAGER_PORT}"
    
    if [[ "${ENABLE_TRACING}" == true ]]; then
        log_info "  Jaeger: http://localhost:${JAEGER_PORT}"
    fi
}

# Stop monitoring services
stop_monitoring() {
    log_info "Stopping monitoring services..."
    
    if [[ "${DRY_RUN}" == true ]]; then
        log_info "DRY RUN: Would stop monitoring services"
        return
    fi
    
    cd "${MONITORING_DIR}"
    docker-compose -f docker-compose.monitoring.yml down
    
    log_success "Monitoring services stopped"
}

# Show status of monitoring services
show_status() {
    log_info "Monitoring services status:"
    
    cd "${MONITORING_DIR}"
    docker-compose -f docker-compose.monitoring.yml ps
}

# Clean monitoring data
clean_monitoring() {
    log_info "Cleaning monitoring data..."
    
    if [[ "${DRY_RUN}" == true ]]; then
        log_info "DRY RUN: Would clean monitoring data"
        return
    fi
    
    cd "${MONITORING_DIR}"
    docker-compose -f docker-compose.monitoring.yml down -v
    docker volume prune -f
    
    log_success "Monitoring data cleaned"
}

# Validate monitoring configuration
validate_monitoring() {
    log_info "Validating monitoring configuration..."
    
    local errors=0
    
    # Check configuration files
    local files=("prometheus.yml" "alertmanager.yml" "blackbox-config.yml")
    for file in "${files[@]}"; do
        if [[ ! -f "${MONITORING_DIR}/${file}" ]]; then
            log_error "Missing configuration file: ${file}"
            ((errors++))
        fi
    done
    
    # Validate Prometheus config
    if command -v promtool >/dev/null 2>&1; then
        if ! promtool check config "${MONITORING_DIR}/prometheus.yml"; then
            log_error "Prometheus configuration validation failed"
            ((errors++))
        fi
        
        if ! promtool check rules "${MONITORING_DIR}/alerts.yml"; then
            log_error "Prometheus rules validation failed"
            ((errors++))
        fi
    else
        log_warning "promtool not available, skipping Prometheus validation"
    fi
    
    # Validate AlertManager config
    if command -v amtool >/dev/null 2>&1; then
        if ! amtool check-config "${MONITORING_DIR}/alertmanager.yml"; then
            log_error "AlertManager configuration validation failed"
            ((errors++))
        fi
    else
        log_warning "amtool not available, skipping AlertManager validation"
    fi
    
    if [[ $errors -eq 0 ]]; then
        log_success "Monitoring configuration validation passed"
    else
        log_error "Monitoring configuration validation failed with ${errors} errors"
        exit 1
    fi
}

# Main execution
main() {
    case "${COMMAND}" in
        setup)
            setup_monitoring
            ;;
        start)
            start_monitoring
            ;;
        stop)
            stop_monitoring
            ;;
        restart)
            stop_monitoring
            start_monitoring
            ;;
        status)
            show_status
            ;;
        configure)
            configure_prometheus
            configure_grafana
            configure_alertmanager
            ;;
        clean)
            clean_monitoring
            ;;
        validate)
            validate_monitoring
            ;;
        *)
            log_error "Unknown command: ${COMMAND}"
            usage
            exit 1
            ;;
    esac
}

# Execute main function
main "$@"
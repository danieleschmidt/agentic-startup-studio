#!/bin/bash

# Agentic Startup Studio - Development Environment Setup Script
# This script sets up a complete development environment for the project

set -euo pipefail

echo "🚀 Setting up Agentic Startup Studio development environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Check if running in devcontainer
if [[ "${REMOTE_CONTAINERS:-}" == "true" ]]; then
    log "Running in VS Code Dev Container"
else
    log "Running in local development environment"
fi

# Update system packages
log "Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# Install additional development tools
log "Installing additional development tools..."
sudo apt-get install -y \
    curl \
    wget \
    git \
    vim \
    nano \
    htop \
    tree \
    jq \
    postgresql-client \
    redis-tools \
    build-essential \
    pkg-config \
    libffi-dev \
    libssl-dev

# Install Python development dependencies
log "Installing Python development dependencies..."
if command -v uv &> /dev/null; then
    log "Using UV for Python package management..."
    uv pip install --upgrade pip
    uv pip install -e ".[dev,test,docs]"
else
    log "Installing UV package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.bashrc || true
    export PATH="$HOME/.cargo/bin:$PATH"
    
    if [[ -f "pyproject.toml" ]]; then
        uv pip install -e ".[dev,test,docs]"
    else
        warn "pyproject.toml not found, installing requirements.txt"
        uv pip install -r requirements.txt
    fi
fi

# Set up pre-commit hooks
log "Setting up pre-commit hooks..."
if [[ -f ".pre-commit-config.yaml" ]]; then
    pre-commit install
    pre-commit install --hook-type commit-msg
    pre-commit install --hook-type pre-push
    log "Pre-commit hooks installed successfully"
else
    warn "No .pre-commit-config.yaml found, skipping pre-commit setup"
fi

# Create necessary directories
log "Creating necessary directories..."
mkdir -p {logs,temp,data,backups,reports}
mkdir -p {tests/reports,docs/generated}

# Set up environment variables
log "Setting up environment variables..."
if [[ ! -f ".env" ]]; then
    if [[ -f ".env.example" ]]; then
        cp .env.example .env
        log "Created .env from .env.example template"
        warn "Please update .env file with your actual configuration values"
    else
        warn "No .env.example file found"
    fi
else
    log ".env file already exists"
fi

# Set proper permissions
log "Setting proper file permissions..."
chmod +x scripts/*.py 2>/dev/null || true
chmod +x .devcontainer/*.sh 2>/dev/null || true
chmod 600 .env 2>/dev/null || true

# Initialize git hooks if not already done
if [[ -d ".git" ]]; then
    log "Setting up git configuration..."
    git config --local core.autocrlf false
    git config --local core.filemode false
    git config --local pull.rebase true
    git config --local init.defaultBranch main
    
    # Set up commit template if exists
    if [[ -f ".gitmessage" ]]; then
        git config --local commit.template .gitmessage
    fi
    
    log "Git configuration updated"
fi

# Install Node.js dependencies if package.json exists
if [[ -f "package.json" ]]; then
    log "Installing Node.js dependencies..."
    npm install
fi

# Set up database connection test
log "Testing database connectivity..."
if [[ -n "${DATABASE_URL:-}" ]]; then
    python -c "
import os
import psycopg2
try:
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    conn.close()
    print('✅ Database connection successful')
except Exception as e:
    print(f'❌ Database connection failed: {e}')
    print('Please ensure PostgreSQL is running and DATABASE_URL is correct')
" || warn "Database connection test failed"
else
    warn "DATABASE_URL not set, skipping database connectivity test"
fi

# Run initial health checks
log "Running initial health checks..."
if [[ -f "scripts/run_health_checks.py" ]]; then
    python scripts/run_health_checks.py --quick || warn "Health checks failed"
fi

# Set up VS Code workspace if in devcontainer
if [[ "${REMOTE_CONTAINERS:-}" == "true" ]]; then
    log "Configuring VS Code workspace..."
    
    # Install VS Code extensions that might not be auto-installed
    code --install-extension ms-python.python --force
    code --install-extension charliermarsh.ruff --force
    code --install-extension ms-python.black-formatter --force
    code --install-extension ms-python.mypy-type-checker --force
    
    log "VS Code extensions installed"
fi

# Display useful information
log "Development environment setup complete! 🎉"
echo
echo -e "${BLUE}📋 Quick Start Commands:${NC}"
echo "  • Run tests:           pytest"
echo "  • Run linting:         ruff check ."
echo "  • Format code:         ruff format ."
echo "  • Type checking:       mypy ."
echo "  • Start API server:    python scripts/serve_api.py"
echo "  • Health checks:       python scripts/run_health_checks.py"
echo "  • CLI help:            python -m pipeline.cli.ingestion_cli --help"
echo
echo -e "${BLUE}📁 Important Directories:${NC}"
echo "  • Source code:         pipeline/, core/, src/"
echo "  • Tests:              tests/"
echo "  • Documentation:      docs/"
echo "  • Scripts:            scripts/"
echo "  • Logs:               logs/"
echo
echo -e "${BLUE}🔧 Configuration Files:${NC}"
echo "  • Environment:        .env"
echo "  • Python config:      pyproject.toml"
echo "  • VS Code settings:   .vscode/settings.json"
echo "  • Git hooks:          .pre-commit-config.yaml"
echo
echo -e "${YELLOW}⚠️  Next Steps:${NC}"
echo "  1. Update .env file with your API keys and database credentials"
echo "  2. Ensure PostgreSQL and Redis are running"
echo "  3. Run 'pytest' to verify everything is working"
echo "  4. Start developing! 🚀"
echo

log "Setup script completed successfully!"
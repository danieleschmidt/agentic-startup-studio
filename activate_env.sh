#!/bin/bash
# Environment activation script for Agentic Startup Studio

# Set the Python path for the project
export PYTHONPATH="/root/repo:$PYTHONPATH"

# Create aliases for consistency
alias python=python3
alias pip=pip3

echo "✅ Environment activated successfully"
echo "Python: $(which python3)"
echo "Python version: $(python3 --version)"
echo "Pytest: $(python3 -m pytest --version 2>/dev/null || echo 'pytest not available')"
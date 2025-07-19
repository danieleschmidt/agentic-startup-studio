#!/bin/bash
# Environment activation script for Agentic Startup Studio

# Set the Python path to use our virtual environment
export PATH="/root/repo/.venv/bin:$PATH"
export PYTHONPATH="/root/repo:$PYTHONPATH"

# Activate virtual environment
source /root/repo/.venv/bin/activate

echo "✅ Environment activated successfully"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "Pytest: $(which pytest)"
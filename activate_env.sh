#!/bin/bash
# Environment activation script for INFRA-001 fix
# This script sets up a working Python environment for the project

# Create and activate virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate the virtual environment
source venv/bin/activate

# Set PYTHONPATH to include src directory
export PYTHONPATH="/root/repo/src:$PYTHONPATH"

# Add aliases for common commands
alias python='python3'
alias pytest='python -m pytest'

echo "Python environment activated. Critical commands available:"
echo "- python: $(which python)"
echo "- pip: $(which pip)"
echo "- PYTHONPATH: $PYTHONPATH"

# Test basic functionality
python -c "import sys; print(f'Python {sys.version} ready')"
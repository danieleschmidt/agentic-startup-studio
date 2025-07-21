#!/usr/bin/env python3
"""
Development Environment Setup Script
Provides a working Python environment for development and testing.
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and handle errors gracefully."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ {description} failed: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False

def main():
    """Set up development environment."""
    print("🚀 Setting up development environment...")
    
    # Set PYTHONPATH
    repo_path = "/root/repo"
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    if repo_path not in current_pythonpath:
        os.environ["PYTHONPATH"] = f"{repo_path}:{current_pythonpath}"
        print(f"✅ PYTHONPATH set to include {repo_path}")
    
    # Check Python version
    version = sys.version_info
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    
    # Check if we can import essential modules
    essential_modules = ['json', 'logging', 'hashlib', 'datetime']
    for module in essential_modules:
        try:
            __import__(module)
            print(f"✅ {module} module available")
        except ImportError:
            print(f"❌ {module} module not available")
    
    # Try to check if testing tools are available
    testing_available = True
    try:
        import pytest
        print("✅ pytest available")
    except ImportError:
        print("⚠️  pytest not available - will need to be installed")
        testing_available = False
    
    try:
        import coverage
        print("✅ coverage available")
    except ImportError:
        print("⚠️  coverage not available - will need to be installed")
    
    # Check if we can run basic imports from our codebase
    try:
        from pipeline.models.idea import IdeaDraft
        print("✅ Pipeline models can be imported")
    except ImportError as e:
        print(f"⚠️  Pipeline models import failed: {e}")
    
    if not testing_available:
        print("\n📋 To enable testing, install: apt install python3-pytest python3-coverage")
        print("   Or use system package manager to install testing dependencies")
    
    print("\n🎉 Development environment setup complete!")
    print(f"   PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
    print(f"   Python executable: {sys.executable}")

if __name__ == "__main__":
    main()
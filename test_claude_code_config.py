#!/usr/bin/env python3
"""
Test script to verify Claude Code Max Plan configuration.
"""

import os
import sys
from pipeline.config.settings import get_claude_code_config, get_settings


def test_claude_code_configuration():
    """Test Claude Code configuration loading."""
    print("Testing Claude Code Configuration")
    print("=" * 50)
    
    # Set environment variables for testing
    os.environ["CLAUDE_CODE_ENABLED"] = "true"
    os.environ["CLAUDE_CODE_PLAN"] = "max"
    os.environ["CLAUDE_CODE_API_KEY"] = "test_api_key"
    os.environ["CLAUDE_CODE_COMPUTE_ENABLED"] = "true"
    
    try:
        # Get configuration
        config = get_claude_code_config()
        
        print(f"Enabled: {config.enabled}")
        print(f"Plan: {config.plan}")
        print(f"Model: {config.model}")
        print(f"Max Tokens: {config.max_tokens}")
        print(f"Compute Enabled: {config.compute_enabled}")
        print(f"Compute Timeout: {config.compute_timeout}s")
        print(f"Compute Max Memory: {config.compute_max_memory} MB")
        print(f"Compute Max CPU: {config.compute_max_cpu} cores")
        print(f"Is Max Plan: {config.is_max_plan()}")
        
        # Test settings integration
        settings = get_settings()
        print(f"\nClaude Code in main settings: {hasattr(settings, 'claude_code')}")
        
        if config.is_max_plan():
            print("\n✅ Claude Code Max Plan is properly configured!")
            print("   Compute features are enabled with the following limits:")
            print(f"   - CPU: {config.compute_max_cpu} cores")
            print(f"   - Memory: {config.compute_max_memory} MB") 
            print(f"   - Timeout: {config.compute_timeout} seconds")
        else:
            print("\n❌ Claude Code Max Plan is not fully configured")
            print("   Please ensure CLAUDE_CODE_PLAN=max and CLAUDE_CODE_COMPUTE_ENABLED=true")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error loading configuration: {e}")
        return False


if __name__ == "__main__":
    success = test_claude_code_configuration()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Production secrets validation script.

This script validates that all required secrets are properly configured
for production deployment and that no hardcoded secrets exist in the codebase.
"""

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

from pipeline.config.secrets_manager import get_secrets_manager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class SecretValidationError(Exception):
    """Exception raised when secret validation fails."""
    pass


def validate_required_secrets(environment: str = "production") -> List[str]:
    """
    Validate that all required secrets are available.
    
    Args:
        environment: Environment to validate for
        
    Returns:
        List of missing secrets
    """
    logger.info(f"Validating secrets for environment: {environment}")
    
    try:
        secrets_manager = get_secrets_manager(environment)
        missing_secrets = secrets_manager.validate_required_secrets()
        
        if missing_secrets:
            logger.error(f"Missing required secrets: {missing_secrets}")
        else:
            logger.info("✅ All required secrets are available")
            
        return missing_secrets
        
    except Exception as e:
        logger.error(f"Failed to validate secrets: {e}")
        raise SecretValidationError(f"Secret validation failed: {e}")


def scan_for_hardcoded_secrets(exclude_paths: List[str] = None) -> List[Tuple[str, int, str]]:
    """
    Scan codebase for potential hardcoded secrets.
    
    Args:
        exclude_paths: Paths to exclude from scanning
        
    Returns:
        List of (file_path, line_number, line_content) tuples with potential secrets
    """
    if exclude_paths is None:
        exclude_paths = [
            "tests/",
            "docs/",
            ".git/",
            "htmlcov/",
            "__pycache__/",
            ".pytest_cache/",
            "node_modules/"
        ]
    
    logger.info("Scanning for hardcoded secrets...")
    
    # Patterns that might indicate hardcoded secrets
    secret_patterns = [
        r'password\s*=\s*["\'][^"\']{8,}["\']',
        r'api_key\s*=\s*["\'][^"\']{16,}["\']',
        r'secret_key\s*=\s*["\'][^"\']{16,}["\']',
        r'token\s*=\s*["\'][^"\']{16,}["\']',
        r'["\'][A-Za-z0-9]{32,}["\']',  # Long random strings
    ]
    
    # Allowlisted patterns (test files, documentation, etc.)
    allowlist_patterns = [
        r'test_.*\.py',
        r'conftest\.py',
        r'.*\.md$',
        r'.*\.yaml$',
        r'dev-secret-key',
        r'test.*password',
        r'example.*key',
        r'your.*key.*here',
    ]
    
    findings = []
    project_root = Path(".")
    
    for file_path in project_root.rglob("*.py"):
        # Skip excluded paths
        if any(exclude_path in str(file_path) for exclude_path in exclude_paths):
            continue
            
        # Skip allowlisted files
        if any(re.search(pattern, str(file_path)) for pattern in allowlist_patterns):
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    # Check if line matches any secret pattern
                    for pattern in secret_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            # Additional check to avoid false positives
                            if not any(re.search(allow, line, re.IGNORECASE) for allow in allowlist_patterns):
                                findings.append((str(file_path), line_num, line.strip()))
                                
        except Exception as e:
            logger.warning(f"Could not scan file {file_path}: {e}")
    
    if findings:
        logger.warning(f"⚠️  Found {len(findings)} potential hardcoded secrets:")
        for file_path, line_num, line_content in findings:
            logger.warning(f"  {file_path}:{line_num} - {line_content[:100]}...")
    else:
        logger.info("✅ No hardcoded secrets detected")
    
    return findings


def validate_secrets_configuration() -> bool:
    """
    Validate that secrets management is properly configured.
    
    Returns:
        True if configuration is valid
    """
    logger.info("Validating secrets management configuration...")
    
    # Check if Google Cloud project is configured for production
    gcp_project = os.getenv("GOOGLE_CLOUD_PROJECT")
    environment = os.getenv("ENVIRONMENT", "development")
    
    if environment == "production":
        if not gcp_project:
            logger.error("❌ GOOGLE_CLOUD_PROJECT not set for production environment")
            return False
        else:
            logger.info(f"✅ Google Cloud project configured: {gcp_project}")
    
    # Test secrets manager initialization
    try:
        secrets_manager = get_secrets_manager(environment)
        logger.info(f"✅ Secrets manager initialized for environment: {environment}")
        
        # Test secret retrieval (without logging the actual values)
        test_secret = secrets_manager.get_secret("SECRET_KEY")
        if test_secret:
            logger.info("✅ Secret retrieval working")
        else:
            logger.warning("⚠️  SECRET_KEY not found - this is expected in development")
            
    except Exception as e:
        logger.error(f"❌ Failed to initialize secrets manager: {e}")
        return False
    
    return True


def main():
    """Main validation script."""
    parser = argparse.ArgumentParser(description="Validate production secrets configuration")
    parser.add_argument(
        "--environment", 
        default="production",
        help="Environment to validate (default: production)"
    )
    parser.add_argument(
        "--scan-hardcoded", 
        action="store_true",
        help="Scan for hardcoded secrets in codebase"
    )
    parser.add_argument(
        "--strict", 
        action="store_true",
        help="Exit with error code if any issues found"
    )
    
    args = parser.parse_args()
    
    issues_found = False
    
    try:
        # Validate secrets configuration
        if not validate_secrets_configuration():
            issues_found = True
        
        # Validate required secrets
        missing_secrets = validate_required_secrets(args.environment)
        if missing_secrets:
            issues_found = True
        
        # Scan for hardcoded secrets if requested
        if args.scan_hardcoded:
            hardcoded_secrets = scan_for_hardcoded_secrets()
            if hardcoded_secrets:
                issues_found = True
        
        # Report results
        if issues_found:
            logger.error("❌ Secrets validation failed - issues found")
            if args.strict:
                sys.exit(1)
        else:
            logger.info("✅ Secrets validation passed - all checks successful")
            
    except Exception as e:
        logger.error(f"❌ Validation script failed: {e}")
        if args.strict:
            sys.exit(1)


if __name__ == "__main__":
    main()
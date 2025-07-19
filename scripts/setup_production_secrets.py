#!/usr/bin/env python3
"""
Production secrets setup script for Agentic Startup Studio.

This script helps set up secure secrets management for production deployment:
1. Validates environment configuration
2. Sets up Google Cloud Secret Manager
3. Migrates secrets from environment variables to Secret Manager
4. Validates the setup

Usage:
    python scripts/setup_production_secrets.py [options]

Prerequisites:
    - Google Cloud SDK installed and authenticated
    - GOOGLE_CLOUD_PROJECT environment variable set
    - Required IAM permissions for Secret Manager
"""

import argparse
import logging
import os
import sys
from typing import Dict, List

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pipeline.config.secrets_manager import (
    GoogleCloudSecretsProvider, 
    get_secrets_manager,
    validate_secret_format
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class ProductionSecretsSetup:
    """Handles production secrets setup and migration."""
    
    def __init__(self, project_id: str = None):
        """Initialize with Google Cloud project ID."""
        self.project_id = project_id or os.getenv('GOOGLE_CLOUD_PROJECT')
        if not self.project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT must be set")
        
        logger.info(f"Setting up secrets for project: {self.project_id}")
    
    def validate_environment(self) -> bool:
        """Validate the environment is ready for production setup."""
        logger.info("Validating environment...")
        
        # Check Google Cloud authentication
        try:
            from google.auth import default
            credentials, project = default()
            logger.info(f"✅ Google Cloud authentication found for project: {project}")
        except Exception as e:
            logger.error(f"❌ Google Cloud authentication failed: {e}")
            logger.error("Run: gcloud auth application-default login")
            return False
        
        # Check required IAM permissions
        try:
            provider = GoogleCloudSecretsProvider(self.project_id)
            logger.info("✅ Google Cloud Secret Manager access verified")
        except Exception as e:
            logger.error(f"❌ Secret Manager access failed: {e}")
            logger.error("Ensure you have Secret Manager Admin role")
            return False
        
        return True
    
    def get_required_secrets(self) -> Dict[str, str]:
        """Get required secrets from current environment."""
        required_secrets = {
            'SECRET_KEY': 'Application secret key for sessions/JWT',
            'DB_PASSWORD': 'Database password',
            'OPENAI_API_KEY': 'OpenAI API key for LLM services',
            'GOOGLE_AI_API_KEY': 'Google AI API key (optional)',
            'EMBEDDING_API_KEY': 'Embedding service API key'
        }
        
        current_values = {}
        missing_secrets = []
        
        logger.info("Checking current environment for required secrets...")
        
        for secret_name, description in required_secrets.items():
            value = os.getenv(secret_name)
            if value:
                # Validate secret format
                if validate_secret_format(value):
                    current_values[secret_name] = value
                    logger.info(f"✅ {secret_name}: Found and valid")
                else:
                    logger.warning(f"⚠️  {secret_name}: Found but may be too weak")
                    current_values[secret_name] = value
            else:
                missing_secrets.append(secret_name)
                logger.warning(f"❌ {secret_name}: Missing ({description})")
        
        if missing_secrets:
            logger.warning(f"Missing secrets: {', '.join(missing_secrets)}")
            logger.warning("Set these in your environment before running production setup")
        
        return current_values
    
    def create_secrets_in_gcp(self, secrets: Dict[str, str], force: bool = False) -> bool:
        """Create secrets in Google Cloud Secret Manager."""
        try:
            from google.cloud import secretmanager
            client = secretmanager.SecretManagerServiceClient()
        except ImportError:
            logger.error("google-cloud-secret-manager not installed")
            logger.error("Install with: pip install google-cloud-secret-manager")
            return False
        
        success_count = 0
        
        for secret_name, secret_value in secrets.items():
            try:
                # Create the secret (metadata)
                parent = f"projects/{self.project_id}"
                secret_id = secret_name
                
                try:
                    secret = client.create_secret(
                        request={
                            "parent": parent,
                            "secret_id": secret_id,
                            "secret": {"replication": {"automatic": {}}},
                        }
                    )
                    logger.info(f"✅ Created secret: {secret_name}")
                except Exception as e:
                    if "already exists" in str(e).lower():
                        if force:
                            logger.info(f"⚠️  Secret {secret_name} exists, updating...")
                        else:
                            logger.warning(f"⚠️  Secret {secret_name} already exists, skipping (use --force to update)")
                            continue
                    else:
                        logger.error(f"❌ Failed to create secret {secret_name}: {e}")
                        continue
                
                # Add the secret version (actual value)
                secret_path = f"{parent}/secrets/{secret_id}"
                version = client.add_secret_version(
                    request={
                        "parent": secret_path,
                        "payload": {"data": secret_value.encode("UTF-8")},
                    }
                )
                logger.info(f"✅ Added version for secret: {secret_name}")
                success_count += 1
                
            except Exception as e:
                logger.error(f"❌ Failed to set up secret {secret_name}: {e}")
        
        logger.info(f"Successfully set up {success_count}/{len(secrets)} secrets")
        return success_count == len(secrets)
    
    def validate_secrets_access(self, secret_names: List[str]) -> bool:
        """Validate that secrets can be accessed from Secret Manager."""
        logger.info("Validating secrets access...")
        
        try:
            provider = GoogleCloudSecretsProvider(self.project_id)
            
            success_count = 0
            for secret_name in secret_names:
                try:
                    value = provider.get_secret(secret_name)
                    if value:
                        logger.info(f"✅ {secret_name}: Access verified")
                        success_count += 1
                    else:
                        logger.error(f"❌ {secret_name}: No value returned")
                except Exception as e:
                    logger.error(f"❌ {secret_name}: Access failed - {e}")
            
            logger.info(f"Successfully accessed {success_count}/{len(secret_names)} secrets")
            return success_count == len(secret_names)
            
        except Exception as e:
            logger.error(f"❌ Failed to validate secrets access: {e}")
            return False
    
    def generate_deployment_config(self) -> str:
        """Generate deployment configuration for production."""
        config = f"""
# Production Deployment Configuration
# Generated by setup_production_secrets.py

# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================
ENVIRONMENT=production
GOOGLE_CLOUD_PROJECT={self.project_id}

# =============================================================================
# SECRETS MANAGEMENT
# =============================================================================
# Secrets are managed by Google Cloud Secret Manager
# The following secrets should be created in Secret Manager:
# - SECRET_KEY
# - DB_PASSWORD
# - OPENAI_API_KEY
# - GOOGLE_AI_API_KEY (optional)
# - EMBEDDING_API_KEY

# =============================================================================
# APPLICATION SETTINGS
# =============================================================================
LOG_LEVEL=INFO
DEBUG=false

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================
DB_HOST=your-production-db-host
DB_PORT=5432
DB_NAME=startup_studio
DB_USER=postgres
# DB_PASSWORD is managed by Secret Manager

# =============================================================================
# INFRASTRUCTURE SETTINGS
# =============================================================================
ENABLE_HEALTH_MONITORING=true
ENABLE_TRACING=true
QUALITY_GATE_ENABLED=true
"""
        return config.strip()


def main():
    """Main setup script."""
    parser = argparse.ArgumentParser(
        description="Set up production secrets management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic setup
    python setup_production_secrets.py
    
    # Force update existing secrets
    python setup_production_secrets.py --force
    
    # Generate config only
    python setup_production_secrets.py --config-only
        """
    )
    
    parser.add_argument(
        '--project-id',
        help='Google Cloud project ID (default: GOOGLE_CLOUD_PROJECT env var)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force update existing secrets'
    )
    parser.add_argument(
        '--config-only',
        action='store_true',
        help='Only generate deployment configuration'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate existing setup'
    )
    
    args = parser.parse_args()
    
    try:
        setup = ProductionSecretsSetup(args.project_id)
        
        if args.config_only:
            config = setup.generate_deployment_config()
            print("\n" + "="*60)
            print("PRODUCTION DEPLOYMENT CONFIGURATION")
            print("="*60)
            print(config)
            print("="*60)
            print("\nSave this to .env.production and customize as needed")
            return
        
        # Validate environment
        if not setup.validate_environment():
            logger.error("Environment validation failed")
            sys.exit(1)
        
        if args.validate_only:
            # Just validate existing secrets
            required_secrets = ['SECRET_KEY', 'DB_PASSWORD', 'OPENAI_API_KEY']
            if setup.validate_secrets_access(required_secrets):
                logger.info("✅ All secrets validated successfully")
            else:
                logger.error("❌ Secrets validation failed")
                sys.exit(1)
            return
        
        # Get current secrets
        current_secrets = setup.get_required_secrets()
        if not current_secrets:
            logger.error("No secrets found in environment")
            logger.error("Set required environment variables first")
            sys.exit(1)
        
        # Create secrets in GCP
        if setup.create_secrets_in_gcp(current_secrets, args.force):
            logger.info("✅ Secrets successfully created in Google Cloud Secret Manager")
            
            # Validate access
            if setup.validate_secrets_access(list(current_secrets.keys())):
                logger.info("✅ Secrets access validation successful")
                
                # Generate deployment config
                config = setup.generate_deployment_config()
                
                # Save to file
                config_file = ".env.production"
                with open(config_file, 'w') as f:
                    f.write(config)
                
                logger.info(f"✅ Production configuration saved to {config_file}")
                logger.info("🚀 Production secrets setup complete!")
                
                print("\n" + "="*60)
                print("NEXT STEPS:")
                print("="*60)
                print("1. Review the generated .env.production file")
                print("2. Update database and infrastructure settings")
                print("3. Deploy your application with ENVIRONMENT=production")
                print("4. Verify secrets are loaded correctly in production")
                print("="*60)
                
            else:
                logger.error("❌ Secrets access validation failed")
                sys.exit(1)
        else:
            logger.error("❌ Failed to create all secrets")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
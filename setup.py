#!/usr/bin/env python3
"""
Agentic Startup Studio Setup Script
Automatically configures the development environment for the pipeline.
"""

import os
import sys
import subprocess
import shutil
import secrets
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from pathlib import Path


class SetupManager:
    """Manages the setup process for Agentic Startup Studio."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.env_file = self.project_root / ".env"
        self.env_example = self.project_root / ".env.example"
        
    def print_header(self, message: str):
        """Print a formatted header message."""
        print(f"\n{'='*60}")
        print(f"  {message}")
        print(f"{'='*60}")
        
    def print_step(self, step: str, status: str = ""):
        """Print a setup step with optional status."""
        status_icon = "[OK]" if status == "success" else "[ERROR]" if status == "error" else "[...]"
        print(f"{status_icon} {step}")
        
    def check_python_version(self):
        """Ensure Python version is compatible."""
        self.print_step("Checking Python version...")
        if sys.version_info < (3, 8):
            self.print_step("Python 3.8+ required", "error")
            sys.exit(1)
        self.print_step(f"Python {sys.version.split()[0]} detected", "success")
        
    def install_dependencies(self):
        """Install Python dependencies."""
        self.print_step("Installing Python dependencies...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ], check=True, capture_output=True)
            self.print_step("Dependencies installed successfully", "success")
        except subprocess.CalledProcessError as e:
            self.print_step(f"Failed to install dependencies: {e}", "error")
            return False
        return True
        
    def create_env_file(self):
        """Create .env file from .env.example."""
        self.print_step("Creating environment configuration...")
        
        if self.env_file.exists():
            self.print_step(".env file already exists", "success")
            return True
            
        if not self.env_example.exists():
            self.print_step(".env.example not found", "error")
            return False
            
        # Copy and customize .env file
        shutil.copy(self.env_example, self.env_file)
        
        # Generate secure secret key
        secret_key = secrets.token_urlsafe(32)
        
        # Read and update .env content
        with open(self.env_file, 'r') as f:
            content = f.read()
            
        # Replace placeholder values
        content = content.replace(
            'your_secret_key_here_change_in_production', 
            secret_key
        )
        
        with open(self.env_file, 'w') as f:
            f.write(content)
            
        self.print_step(".env file created with secure secret key", "success")
        return True
        
    def check_postgresql(self):
        """Check if PostgreSQL is installed and accessible."""
        self.print_step("Checking PostgreSQL installation...")
        
        try:
            # Try to find PostgreSQL installation
            result = subprocess.run([
                "where", "psql"
            ], capture_output=True, text=True, shell=True)
            
            if result.returncode == 0:
                self.print_step("PostgreSQL found", "success")
                return True
            else:
                self.print_step("PostgreSQL not found in PATH", "error")
                return False
                
        except Exception as e:
            self.print_step(f"PostgreSQL check failed: {e}", "error")
            return False
            
    def setup_database(self):
        """Set up PostgreSQL database and user."""
        self.print_step("Setting up PostgreSQL database...")
        
        db_config = {
            'host': 'localhost',
            'port': '5432',
            'user': 'studio',
            'password': 'studio',
            'database': 'studio'
        }
        
        try:
            # Try to connect as postgres superuser first
            self.print_step("Connecting to PostgreSQL as superuser...")
            
            # Prompt for superuser credentials
            postgres_password = input("Enter PostgreSQL superuser password (or press Enter for no password): ").strip()
            
            conn_params = {
                'host': db_config['host'],
                'port': db_config['port'],
                'user': 'postgres',
                'database': 'postgres'
            }
            
            if postgres_password:
                conn_params['password'] = postgres_password
                
            conn = psycopg2.connect(**conn_params)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            # Create user if not exists
            self.print_step("Creating database user...")
            cursor.execute(f"""
                DO $$ 
                BEGIN 
                    IF NOT EXISTS (SELECT FROM pg_user WHERE usename = '{db_config['user']}') THEN
                        CREATE USER {db_config['user']} WITH PASSWORD '{db_config['password']}';
                    END IF;
                END $$;
            """)
            
            # Create database if not exists
            self.print_step("Creating database...")
            cursor.execute(f"""
                SELECT 1 FROM pg_database WHERE datname = '{db_config['database']}'
            """)
            
            if not cursor.fetchone():
                cursor.execute(f"CREATE DATABASE {db_config['database']} OWNER {db_config['user']}")
                self.print_step(f"Database '{db_config['database']}' created", "success")
            else:
                self.print_step(f"Database '{db_config['database']}' already exists", "success")
                
            # Grant privileges
            cursor.execute(f"GRANT ALL PRIVILEGES ON DATABASE {db_config['database']} TO {db_config['user']}")
            
            cursor.close()
            conn.close()
            
            # Test connection with new user
            self.print_step("Testing database connection...")
            test_conn = psycopg2.connect(**{
                'host': db_config['host'],
                'port': db_config['port'],
                'user': db_config['user'],
                'password': db_config['password'],
                'database': db_config['database']
            })
            test_conn.close()
            
            self.print_step("Database setup completed successfully", "success")
            return True
            
        except psycopg2.Error as e:
            self.print_step(f"Database setup failed: {e}", "error")
            return False
        except Exception as e:
            self.print_step(f"Unexpected error during database setup: {e}", "error")
            return False
            
    def test_configuration(self):
        """Test the pipeline configuration."""
        self.print_step("Testing pipeline configuration...")
        
        try:
            # Test configuration loading
            result = subprocess.run([
                sys.executable, "-m", "pipeline.cli.ingestion_cli", "config"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                self.print_step("Configuration test passed", "success")
                return True
            else:
                self.print_step(f"Configuration test failed: {result.stderr}", "error")
                return False
                
        except Exception as e:
            self.print_step(f"Configuration test error: {e}", "error")
            return False
            
    def create_database_tables(self):
        """Create necessary database tables."""
        self.print_step("Creating database tables...")
        
        # Check if there are any migration scripts
        db_dir = self.project_root / "db"
        if db_dir.exists():
            bootstrap_sql = db_dir / "bootstrap.sql"
            if bootstrap_sql.exists():
                try:
                    # Run bootstrap SQL
                    result = subprocess.run([
                        "psql",
                        "-h", "localhost",
                        "-p", "5432", 
                        "-U", "studio",
                        "-d", "studio",
                        "-f", str(bootstrap_sql)
                    ], capture_output=True, text=True, env={**os.environ, 'PGPASSWORD': 'studio'})
                    
                    if result.returncode == 0:
                        self.print_step("Database tables created", "success")
                        return True
                    else:
                        self.print_step(f"Table creation failed: {result.stderr}", "error")
                        return False
                        
                except Exception as e:
                    self.print_step(f"Table creation error: {e}", "error")
                    return False
            else:
                self.print_step("No bootstrap.sql found, skipping table creation")
                return True
        else:
            self.print_step("No database directory found, skipping table creation")
            return True
            
    def run_setup(self):
        """Run the complete setup process."""
        self.print_header("Agentic Startup Studio Setup")
        
        setup_steps = [
            ("Python Version Check", self.check_python_version),
            ("Environment File", self.create_env_file), 
            ("Python Dependencies", self.install_dependencies),
            ("PostgreSQL Check", self.check_postgresql),
            ("Database Setup", self.setup_database),
            ("Database Tables", self.create_database_tables),
            ("Configuration Test", self.test_configuration)
        ]
        
        failed_steps = []
        
        for step_name, step_func in setup_steps:
            try:
                if not step_func():
                    failed_steps.append(step_name)
            except KeyboardInterrupt:
                print("\n\nSetup interrupted by user.")
                sys.exit(1)
            except Exception as e:
                self.print_step(f"{step_name} failed with error: {e}", "error")
                failed_steps.append(step_name)
                
        # Summary
        self.print_header("Setup Summary")
        
        if not failed_steps:
            print("🎉 Setup completed successfully!")
            print("\nNext steps:")
            print("1. Run: python -m pipeline.cli.ingestion_cli config")
            print("2. Run: python -m pipeline.demo_pipeline")
            print("3. Optional: Set OPENAI_API_KEY environment variable for AI features")
        else:
            print("❌ Setup completed with errors:")
            for step in failed_steps:
                print(f"   • {step}")
            print("\nPlease resolve these issues and run setup again.")
            
        print(f"\nFor support, check the documentation in the docs/ directory.")


def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print("Agentic Startup Studio Setup Script")
        print("Usage: python setup.py")
        print("\nThis script will:")
        print("• Create .env configuration file")
        print("• Install Python dependencies") 
        print("• Set up PostgreSQL database")
        print("• Test the configuration")
        return
        
    setup_manager = SetupManager()
    setup_manager.run_setup()


if __name__ == "__main__":
    main()
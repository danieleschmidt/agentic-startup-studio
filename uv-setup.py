#!/usr/bin/env python3
"""
UV-based setup script for Agentic Startup Studio
Fast, modern Python environment setup using UV package manager
"""
import os
import sys
import subprocess
import secrets
import platform
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

class UVSetupManager:
    """UV-based setup manager for modern Python development"""
    
    def __init__(self, debug: bool = False, log_file: Optional[str] = None):
        self.project_root = Path.cwd()
        self.env_file = self.project_root / ".env"
        self.debug = debug or os.getenv("UV_SETUP_DEBUG", "").lower() in ["true", "1", "yes"]
        self.start_time = time.time()
        self.setup_log(log_file)
        
        self.status_icons = {
            "success": "✅" if platform.system() != "Windows" else "[OK]",
            "error": "❌" if platform.system() != "Windows" else "[ERROR]",
            "info": "ℹ️" if platform.system() != "Windows" else "[INFO]",
            "warning": "⚠️" if platform.system() != "Windows" else "[WARN]",
            "debug": "🔍" if platform.system() != "Windows" else "[DEBUG]"
        }
        
        self.log_info("UV Setup Manager initialized")
        self.log_debug(f"Project root: {self.project_root}")
        self.log_debug(f"Environment file: {self.env_file}")
        self.log_debug(f"Debug mode: {self.debug}")
        self.log_debug(f"Platform: {platform.system()} {platform.release()}")

    def setup_log(self, log_file: Optional[str] = None) -> None:
        """Setup logging configuration"""
        log_level = logging.DEBUG if self.debug else logging.INFO
        
        # Create logs directory if it doesn't exist
        log_dir = self.project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Default log file path
        if not log_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"uv_setup_{timestamp}.log"
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout) if self.debug else logging.NullHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.log_file = Path(log_file)
        
        # Log initial setup info
        self.logger.info("=" * 60)
        self.logger.info("UV Setup Manager - Logging Started")
        self.logger.info(f"Log file: {self.log_file}")
        self.logger.info(f"Debug mode: {self.debug}")
        self.logger.info(f"Python version: {sys.version}")
        self.logger.info(f"Platform: {platform.platform()}")
        self.logger.info("=" * 60)

    def log_debug(self, message: str) -> None:
        """Log debug message"""
        if hasattr(self, 'logger'):
            self.logger.debug(message)
        if self.debug:
            self.print_status(message, "debug")

    def log_info(self, message: str) -> None:
        """Log info message"""
        if hasattr(self, 'logger'):
            self.logger.info(message)

    def log_warning(self, message: str) -> None:
        """Log warning message"""
        if hasattr(self, 'logger'):
            self.logger.warning(message)

    def log_error(self, message: str) -> None:
        """Log error message"""
        if hasattr(self, 'logger'):
            self.logger.error(message)

    def log_command(self, command: str, result: subprocess.CompletedProcess) -> None:
        """Log command execution details"""
        self.log_debug(f"Executing command: {command}")
        self.log_debug(f"Return code: {result.returncode}")
        if result.stdout:
            self.log_debug(f"STDOUT: {result.stdout.strip()}")
        if result.stderr:
            self.log_debug(f"STDERR: {result.stderr.strip()}")

    def log_timing(self, operation: str, duration: float) -> None:
        """Log operation timing"""
        self.log_info(f"{operation} completed in {duration:.2f} seconds")

    def print_status(self, message: str, status: str = "info") -> None:
        """Print formatted status message with logging"""
        icon = self.status_icons.get(status, "")
        status_message = f"{icon} {message}"
        print(status_message)
        
        # Also log to file
        if status == "error":
            self.log_error(message)
        elif status == "warning":
            self.log_warning(message)
        elif status == "debug":
            self.log_debug(message)
        else:
            self.log_info(message)

    def run_command(self, command: str, check: bool = True) -> subprocess.CompletedProcess:
        """Run shell command with error handling and detailed logging"""
        start_time = time.time()
        self.log_debug(f"Executing command: {command}")
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                check=check
            )
            
            duration = time.time() - start_time
            self.log_command(command, result)
            self.log_timing(f"Command '{command}'", duration)
            
            if result.returncode != 0 and check:
                self.log_error(f"Command failed with return code {result.returncode}")
                self.log_error(f"STDERR: {result.stderr}")
                
            return result
            
        except subprocess.CalledProcessError as e:
            duration = time.time() - start_time
            self.log_error(f"Command failed after {duration:.2f}s: {command}")
            self.log_error(f"Return code: {e.returncode}")
            self.log_error(f"STDERR: {e.stderr}")
            self.print_status(f"Command failed: {command}", "error")
            self.print_status(f"Error: {e.stderr}", "error")
            raise
        except Exception as e:
            duration = time.time() - start_time
            self.log_error(f"Unexpected error after {duration:.2f}s executing '{command}': {e}")
            self.print_status(f"Unexpected error: {e}", "error")
            raise

    def check_uv_installation(self) -> bool:
        """Check if UV is installed and accessible"""
        try:
            result = self.run_command("uv --version", check=False)
            if result.returncode == 0:
                version = result.stdout.strip()
                self.print_status(f"UV {version} detected", "success")
                return True
            else:
                self.print_status("UV not found in PATH", "error")
                return False
        except Exception:
            self.print_status("UV not available", "error")
            return False

    def install_uv(self) -> bool:
        """Install UV package manager"""
        self.print_status("Installing UV package manager...", "info")
        
        try:
            if platform.system() == "Windows":
                # Windows installation
                install_cmd = "powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\""
            else:
                # Unix-like systems
                install_cmd = "curl -LsSf https://astral.sh/uv/install.sh | sh"
            
            self.run_command(install_cmd)
            self.print_status("UV installed successfully", "success")
            return True
            
        except Exception as e:
            self.print_status(f"Failed to install UV: {e}", "error")
            return False

    def create_virtual_environment(self) -> bool:
        """Create UV virtual environment"""
        try:
            self.print_status("Creating UV virtual environment...", "info")
            self.run_command("uv venv")
            self.print_status("Virtual environment created", "success")
            return True
        except Exception as e:
            self.print_status(f"Failed to create virtual environment: {e}", "error")
            return False

    def sync_dependencies(self) -> bool:
        """Sync dependencies using UV"""
        try:
            self.print_status("Syncing dependencies with UV...", "info")
            
            # Install core dependencies with native TLS support
            self.run_command("uv sync --native-tls")
            
            # Install development dependencies with native TLS support
            self.run_command("uv sync --extra dev --native-tls")
            
            self.print_status("Dependencies synced successfully", "success")
            return True
            
        except Exception as e:
            self.print_status(f"Failed to sync dependencies: {e}", "error")
            return False

    def generate_env_file(self) -> bool:
        """Generate .env file with secure defaults"""
        try:
            if self.env_file.exists():
                self.print_status(".env file already exists, skipping generation", "warning")
                return True

            self.print_status("Generating .env configuration file...", "info")
            
            # Generate secure secret key
            secret_key = secrets.token_urlsafe(32)
            
            env_content = f"""# Agentic Startup Studio Environment Configuration
# Generated by UV setup script

# Core Application Settings
SECRET_KEY={secret_key}
DEBUG=true
ENVIRONMENT=development
LOG_LEVEL=INFO

# Database Configuration
DATABASE_URL=postgresql://studio:studio@localhost:5432/studio
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=studio
DATABASE_USER=studio
DATABASE_PASSWORD=studio

# Redis Configuration  
REDIS_URL=redis://localhost:6379/0
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# API Keys (Add your actual keys here)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Security Settings
JWT_SECRET_KEY={secrets.token_urlsafe(32)}
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=1440

# Pipeline Configuration
PIPELINE_BATCH_SIZE=100
PIPELINE_TIMEOUT_SECONDS=300
PIPELINE_MAX_RETRIES=3

# Development Settings
RELOAD=true
HOST=0.0.0.0
PORT=8000
"""
            
            self.env_file.write_text(env_content, encoding='utf-8')
            self.print_status(".env file created with secure configuration", "success")
            return True
            
        except Exception as e:
            self.print_status(f"Failed to create .env file: {e}", "error")
            return False

    def setup_pre_commit(self) -> bool:
        """Setup pre-commit hooks for code quality"""
        try:
            self.print_status("Setting up pre-commit hooks...", "info")
            
            # Install pre-commit hooks
            self.run_command("uv run pre-commit install")
            
            self.print_status("Pre-commit hooks configured", "success")
            return True
            
        except Exception as e:
            self.print_status(f"Failed to setup pre-commit: {e}", "error")
            return False

    def check_postgresql(self) -> bool:
        """Check if PostgreSQL is available"""
        try:
            result = self.run_command("psql --version", check=False)
            if result.returncode == 0:
                version = result.stdout.strip()
                self.print_status(f"PostgreSQL detected: {version}", "success")
                return True
            else:
                self.print_status("PostgreSQL not found in PATH", "warning")
                return False
        except Exception:
            self.print_status("PostgreSQL not available", "warning")
            return False

    def run_tests(self) -> bool:
        """Run test suite to verify setup"""
        try:
            self.print_status("Running test suite...", "info")
            
            # Run tests with UV
            result = self.run_command("uv run pytest --tb=short", check=False)
            
            if result.returncode == 0:
                self.print_status("All tests passed", "success")
                return True
            else:
                self.print_status("Some tests failed", "warning")
                self.print_status("Check test output for details", "info")
                return False
                
        except Exception as e:
            self.print_status(f"Failed to run tests: {e}", "error")
            return False

    def print_setup_summary(self, total_time: float) -> None:
        """Print comprehensive setup summary with timing and log information"""
        self.print_status(f"\nUV Setup Complete! (Total time: {total_time:.2f}s)", "success")
        
        # Log file information
        print(f"\n📋 Setup Details:")
        print(f"   Log file: {self.log_file}")
        print(f"   Debug mode: {'Enabled' if self.debug else 'Disabled'}")
        print(f"   Total time: {total_time:.2f} seconds")
        
        print("\n🚀 Next Steps:")
        print("1. Activate the virtual environment:")
        print("   source .venv/bin/activate  # Linux/macOS")
        print("   .venv\\Scripts\\activate     # Windows")
        print("\n2. Or use UV to run commands directly:")
        print("   uv run python -m pipeline.cli.ingestion_cli --help")
        print("   uv run pytest")
        print("   uv run python -m pipeline.demo_pipeline")
        
        print("\n3. Install PostgreSQL if not available:")
        print("   https://www.postgresql.org/download/")
        
        print("\n4. Configure API keys in .env file:")
        print("   OPENAI_API_KEY=your_actual_key")
        print("   ANTHROPIC_API_KEY=your_actual_key")
        
        print("\n5. Start development server:")
        print("   uv run uvicorn pipeline.main:app --reload")
        
        print(f"\n🔍 Debug Mode:")
        print(f"   Run with '--debug' flag for detailed logging:")
        print(f"   python uv-setup.py --debug")
        
        print(f"\n📝 For troubleshooting, check the log file:")
        print(f"   {self.log_file}")


def main():
    """Main setup function with comprehensive logging"""
    # Parse command line arguments for debug mode
    debug_mode = "--debug" in sys.argv or "-d" in sys.argv
    
    print("Agentic Startup Studio - UV Setup")
    print("=" * 50)
    
    setup = UVSetupManager(debug=debug_mode)
    setup.log_info("Starting UV setup process")
    
    try:
        # Step 1: Check/Install UV
        step_start = time.time()
        setup.log_info("Step 1: Checking UV installation")
        if not setup.check_uv_installation():
            setup.log_info("UV not found, attempting installation")
            if not setup.install_uv():
                setup.log_error("UV installation failed")
                print("\nSetup failed. Please install UV manually:")
                print("https://docs.astral.sh/uv/getting-started/installation/")
                sys.exit(1)
        setup.log_timing("Step 1 (UV Check/Install)", time.time() - step_start)
        
        # Step 2: Create virtual environment
        step_start = time.time()
        setup.log_info("Step 2: Creating virtual environment")
        if not setup.create_virtual_environment():
            setup.log_error("Virtual environment creation failed")
            sys.exit(1)
        setup.log_timing("Step 2 (Virtual Environment)", time.time() - step_start)
        
        # Step 3: Sync dependencies
        step_start = time.time()
        setup.log_info("Step 3: Syncing dependencies")
        if not setup.sync_dependencies():
            setup.log_error("Dependency synchronization failed")
            sys.exit(1)
        setup.log_timing("Step 3 (Dependencies)", time.time() - step_start)
        
        # Step 4: Generate environment file
        step_start = time.time()
        setup.log_info("Step 4: Generating environment file")
        if not setup.generate_env_file():
            setup.log_error("Environment file generation failed")
            sys.exit(1)
        setup.log_timing("Step 4 (Environment File)", time.time() - step_start)
        
        # Step 5: Setup development tools
        step_start = time.time()
        setup.log_info("Step 5: Setting up development tools")
        setup.setup_pre_commit()
        setup.log_timing("Step 5 (Development Tools)", time.time() - step_start)
        
        # Step 6: Check external dependencies
        step_start = time.time()
        setup.log_info("Step 6: Checking external dependencies")
        setup.check_postgresql()
        setup.log_timing("Step 6 (External Dependencies)", time.time() - step_start)
        
        # Step 7: Run tests
        step_start = time.time()
        setup.log_info("Step 7: Running test suite")
        setup.run_tests()
        setup.log_timing("Step 7 (Test Suite)", time.time() - step_start)
        
        # Step 8: Final summary
        total_time = time.time() - setup.start_time
        setup.log_timing("Total UV Setup", total_time)
        setup.log_info("UV setup completed successfully")
        
        # Print completion message with timing and log info
        setup.print_setup_summary(total_time)
        
    except KeyboardInterrupt:
        setup.log_warning("Setup interrupted by user")
        print("\n\nSetup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        setup.log_error(f"Unexpected error during setup: {e}")
        print(f"\nUnexpected error: {e}")
        print(f"Check log file for details: {setup.log_file}")
        sys.exit(1)

if __name__ == "__main__":
    main()
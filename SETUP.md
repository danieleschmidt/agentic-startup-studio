# Agentic Startup Studio Setup Guide

This guide will help you set up the Agentic Startup Studio pipeline development environment using modern Python tooling.

## Quick Setup

### Recommended: UV-based Setup (Fast & Modern)

Run the UV-based automated setup script:

```bash
python uv-setup.py
```

The UV setup script will automatically:
- ✅ Check/Install UV package manager
- ✅ Create UV virtual environment (.venv)
- ✅ Sync dependencies using UV (much faster than pip)
- ✅ Generate `.env` configuration file with secure defaults
- ✅ Setup pre-commit hooks for code quality
- ✅ Check PostgreSQL installation
- ✅ Run test suite for verification

### Alternative: Traditional pip-based Setup

If you prefer the traditional approach:

```bash
python setup.py
```

The pip setup script provides:
- ✅ Check Python version compatibility (3.10+)
- ✅ Create `.env` configuration file from template
- ✅ Install Python dependencies with pip
- ✅ Check PostgreSQL installation
- ✅ Set up PostgreSQL database and user
- ✅ Test pipeline configuration

## Prerequisites

### Required
- **Python 3.8+** - Check with `python --version`
- **PostgreSQL** - Database server for data storage

### Optional
- **Redis** - For caching (improves performance)
- **OpenAI API Key** - For AI-powered features

## Manual Setup (if automated setup fails)

### 1. Install PostgreSQL

#### Windows
1. Download PostgreSQL from https://www.postgresql.org/download/windows/
2. Run the installer and follow the setup wizard
3. Remember the superuser password you set during installation

#### macOS
```bash
brew install postgresql
brew services start postgresql
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

### 2. Create Database and User

Connect to PostgreSQL as superuser:
```bash
psql -U postgres
```

Create the database and user:
```sql
CREATE USER studio WITH PASSWORD 'studio';
CREATE DATABASE studio OWNER studio;
GRANT ALL PRIVILEGES ON DATABASE studio TO studio;
\q
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

Copy the environment template:
```bash
cp .env.example .env
```

Edit `.env` and configure:
- Database connection string
- API keys (OpenAI, etc.)
- Other service URLs

### 5. Test Configuration

```bash
python -m pipeline.cli.ingestion_cli config
```

## Configuration Options

### Database Configuration
The default configuration uses:
- **Host**: localhost
- **Port**: 5432
- **Database**: studio
- **User**: studio
- **Password**: studio

To change these, update the `DATABASE_URL` in your `.env` file:
```
DATABASE_URL=postgresql://username:password@host:port/database_name
```

### API Keys

#### OpenAI API Key (Required for AI features)
1. Get your API key from https://platform.openai.com/api-keys
2. Add to `.env`:
```
OPENAI_API_KEY=your_api_key_here
```

#### Optional Services
- **Redis**: `REDIS_URL=redis://localhost:6379`
- **PostHog Analytics**: `POSTHOG_API_KEY=your_key_here`

## Verification Commands

Test different components:

```bash
# Test configuration
python -m pipeline.cli.ingestion_cli config

# List available functionality
python -m pipeline.cli.ingestion_cli list

# Run pipeline demonstration
python -m pipeline.demo_pipeline
```

## Troubleshooting

### Common Issues

#### "PostgreSQL not found"
- Install PostgreSQL and ensure it's in your PATH
- On Windows, add PostgreSQL bin directory to PATH

#### "Connection refused"
- Ensure PostgreSQL service is running
- Check if PostgreSQL is listening on the correct port
- Verify firewall settings

#### "psycopg2 installation failed"
- Install system dependencies:
  - **Ubuntu/Debian**: `sudo apt install libpq-dev python3-dev`
  - **CentOS/RHEL**: `sudo yum install postgresql-devel python3-devel`
  - **macOS**: `brew install postgresql`

#### "Permission denied"
- Ensure PostgreSQL user has correct permissions
- Check database connection string in `.env`

### Getting Help

1. Check the logs in the terminal output
2. Verify all prerequisites are installed
3. Ensure all services (PostgreSQL, Redis) are running
4. Check firewall and network connectivity

## Development Workflow

After setup is complete:

1. **Start development**: The pipeline is ready to use
2. **Run tests**: `python -m pytest` (if test suite is available)
3. **Start services**: Database and Redis should be running
4. **Environment**: All configuration is in `.env`

## Next Steps

- Review the documentation in the `docs/` directory
- Explore the pipeline components in the `pipeline/` directory
- Run the demo to see the pipeline in action
- Customize configuration for your specific needs

## File Structure

```
agentic-startup-studio/
├── setup.py              # Automated setup script
├── .env.example          # Environment template
├── .env                  # Your configuration (created by setup)
├── requirements.txt      # Python dependencies
├── pipeline/             # Main pipeline code
├── tests/               # Test suite
├── docs/                # Documentation
└── SETUP.md             # This file
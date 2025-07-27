FROM mcr.microsoft.com/devcontainers/python:3.11

# Install system dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        postgresql-client \
        redis-tools \
        curl \
        wget \
        git \
        vim \
        htop \
        tree \
        jq \
        netcat-openbsd \
        build-essential \
        libpq-dev \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install Node.js for any frontend development
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Install Docker CLI (for docker-in-docker scenarios)
RUN curl -fsSL https://get.docker.com | sh

# Create workspace directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements.txt .
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install \
        pre-commit \
        pytest-cov \
        pytest-xdist \
        pytest-mock \
        black \
        isort \
        mypy \
        ruff \
        bandit \
        safety \
        jupyter \
        notebook \
        ipykernel

# Install UV for fast package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Set up non-root user
USER vscode

# Install Oh My Zsh for better terminal experience
RUN sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

# Add useful aliases and environment setup
RUN echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.zshrc \
    && echo 'export PYTHONPATH="/workspace:$PYTHONPATH"' >> ~/.zshrc

# Default command
CMD ["zsh"]
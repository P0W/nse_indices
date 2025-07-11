# Multi-stage build for optimized image size and faster builds
# Stage 1: Build dependencies and virtual environment
FROM python:3.12-slim AS builder

# Install system dependencies needed for building
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files first (leverage Docker layer caching)
COPY pyproject.toml uv.lock ./

# Create virtual environment and install dependencies
RUN uv sync --frozen --no-dev

# Stage 2: Runtime image
FROM python:3.12-slim AS runtime

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv for runtime
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Create non-root user first
RUN useradd --create-home --shell /bin/bash --uid 1000 app

# Set working directory
WORKDIR /app

# Create logs directory with proper permissions
RUN mkdir -p /app/logs && chown -R app:app /app

# Copy virtual environment from builder stage
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/pyproject.toml /app/uv.lock ./

# Copy only necessary source files (exclude test files, docs, etc.)
COPY --chown=app:app *.py ./
COPY --chown=app:app *.csv ./

# Switch to non-root user
USER app

# Add virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH"
# Environment variables for Python optimization
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# Ensure the application binds to the correct port
ENV PORT=8000
ENV HOST=0.0.0.0

# Expose port
EXPOSE 8000

# Optimized health check with shorter intervals for faster startup
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=2 \
    CMD curl -f http://localhost:8000/health || exit 1

# Use exec form and virtual environment directly with explicit port
CMD ["/app/.venv/bin/python", "-c", "import web_server; web_server.main()"]

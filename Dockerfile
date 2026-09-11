# Ouroboros — Docker image for web UI runtime
# Usage:
#   docker build -t ouroboros-web .
#   docker run --rm -p 8765:8765 ouroboros-web

FROM ghcr.io/astral-sh/uv:0.12.1 AS uv
FROM python:3.10-slim

COPY --from=uv /uv /uvx /bin/

# System dependencies (git + Playwright/Chromium native libs installed via playwright install-deps)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Working directory
ENV APP_HOME=/app
WORKDIR ${APP_HOME}

# Resolve only from the reviewed lock. Keeping dependencies in their own layer
# lets source edits reuse the expensive Python package and browser downloads.
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --extra browser --no-install-project

# Install all Playwright native system dependencies for Chromium/WebKit (authoritative list from Playwright)
RUN python3 -m playwright install-deps chromium webkit

# Install Playwright Chromium/WebKit browser binaries so browser tools work out of the box
RUN PLAYWRIGHT_BROWSERS_PATH=0 python3 -m playwright install chromium webkit

# Copy application
COPY . .
RUN uv sync --locked --no-dev --extra browser --no-editable

# Default environment
ENV OUROBOROS_SERVER_HOST=0.0.0.0 \
    OUROBOROS_SERVER_PORT=8765 \
    OUROBOROS_FILE_BROWSER_DEFAULT=${APP_HOME}

EXPOSE 8765

ENTRYPOINT ["python", "server.py"]

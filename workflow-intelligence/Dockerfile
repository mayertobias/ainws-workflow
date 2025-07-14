# Intelligence Service Dockerfile - Single Stage Build
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy shared library package first
COPY shared-libs/feature-translator ./shared-libs/feature-translator

# Copy requirements and install dependencies
COPY workflow-intelligence/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install the feature translator package
RUN pip install -e ./shared-libs/feature-translator

# Copy application code
COPY workflow-intelligence/app/ ./app/

# Create directories and set permissions
RUN mkdir -p /app/logs /app/data /home/appuser \
    && chown -R appuser:appuser /app /home/appuser

# Set environment variables
ENV PATH="/home/appuser/.local/bin:$PATH"
ENV PORT=8005
ENV DEBUG=false
ENV AI_PROVIDER=auto
ENV CACHE_TTL_SECONDS=7200
ENV MAX_REQUESTS_PER_MINUTE=60

# Optimized health check
COPY <<EOF /app/healthcheck.py
#!/usr/bin/env python3
import sys
import os
try:
    import httpx
    port = os.getenv("PORT", "8005")
    response = httpx.get(f"http://localhost:{port}/health", timeout=10)
    sys.exit(0 if response.status_code == 200 else 1)
except Exception:
    sys.exit(1)
EOF

RUN chmod +x /app/healthcheck.py

EXPOSE 8005

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python /app/healthcheck.py

USER appuser

ENV PORT=8005
ENV DEBUG=false
ENV AI_PROVIDER=auto
ENV CACHE_TTL_SECONDS=7200
ENV MAX_REQUESTS_PER_MINUTE=60

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8005", "--workers", "1"] 
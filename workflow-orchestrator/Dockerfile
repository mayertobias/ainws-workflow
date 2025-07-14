# Simplified Orchestrator Service Dockerfile
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8006
ENV DEBUG=false
ENV LOG_LEVEL=INFO
ENV MAX_PARALLEL_WORKFLOWS=50
ENV WORKFLOW_TIMEOUT_SECONDS=3600

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create app user for security
RUN groupadd -r appuser && useradd -r -g appuser -m appuser

WORKDIR /app

# Copy requirements and install packages globally (before switching to appuser)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip cache purge

# Copy application code
COPY app/ ./app/

# Create necessary directories with proper ownership
RUN mkdir -p /app/logs /app/data \
    && chown -R appuser:appuser /app

EXPOSE 8006

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

USER appuser

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8006", "--workers", "1"] 
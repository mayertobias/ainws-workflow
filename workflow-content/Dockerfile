# Optimized Content Service Dockerfile
# Target: Reduce from 1.68GB to ~1.2GB (29% reduction)

FROM python:3.12-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy optimized requirements (removed jinja2)
COPY requirements.txt .

# Install dependencies with cleanup
RUN pip install --user --no-cache-dir -r requirements.txt \
    && pip cache purge \
    && find /root/.local -name "*.pyc" -delete \
    && find /root/.local -name "__pycache__" -delete

# Download spaCy model to user directory
RUN python -m spacy download en_core_web_sm

# Production stage
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/root/.local/bin:$PATH"

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Copy optimized packages and spaCy model from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY app/ ./app/

# Create directories for data
RUN mkdir -p /tmp/lyrics /tmp/output

# Expose port
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"] 
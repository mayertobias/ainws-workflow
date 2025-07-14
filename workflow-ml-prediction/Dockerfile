# ML Prediction Service Dockerfile - Single Stage Build
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy shared library packages first
COPY shared-libs/feature-translator ./shared-libs/feature-translator
COPY shared-libs/derived-features ./shared-libs/derived-features

# Copy requirements and install dependencies
COPY workflow-ml-prediction/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install the shared library packages
RUN pip install -e ./shared-libs/feature-translator
RUN pip install -e ./shared-libs/derived-features

# Copy application code
COPY workflow-ml-prediction/app/ ./app/

# Create directories for models
RUN mkdir -p /tmp/models

# Set environment variables
ENV PORT=8004

# Expose port
EXPOSE 8004

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8004/health || exit 1

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8004"] 
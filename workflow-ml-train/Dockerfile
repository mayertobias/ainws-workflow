FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy shared library packages first
COPY shared-libs/feature-translator ./shared-libs/feature-translator
COPY shared-libs/derived-features ./shared-libs/derived-features

# Copy requirements and install dependencies
COPY workflow-ml-train/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install the shared library packages
RUN pip install -e ./shared-libs/feature-translator
RUN pip install -e ./shared-libs/derived-features

# Copy application code
COPY workflow-ml-train/app/ ./app/

# Create necessary directories
RUN mkdir -p /app/models /app/mlruns /app/logs /app/shared-data

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8005

# Expose port
EXPOSE 8005

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8005/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8005"] 
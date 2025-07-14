# Optimized Audio Service Dockerfile
# Target: Reduce from 2.49GB to ~800MB (68% reduction)

FROM python:3.9.18-slim AS builder

# Install build dependencies for audio processing
RUN apt-get update && apt-get install -y \
    build-essential \
    libyaml-dev \
    libfftw3-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswresample-dev \
    python3-dev \
    libtag1-dev \
    libsamplerate0-dev \
    libchromaprint-dev \
    libsndfile1-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy optimized requirements (remove matplotlib, seaborn)
COPY requirements.txt .

# Install dependencies and clean up
RUN pip install --user --no-cache-dir -r requirements.txt \
    && pip cache purge \
    && find /root/.local -name "*.pyc" -delete \
    && find /root/.local -name "__pycache__" -delete

# Production stage
FROM python:3.9.18-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/root/.local/bin:$PATH"

# TensorFlow Model Caching Optimizations (Permanent)
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_ENABLE_ONEDNN_OPTS=1
ENV TF_NUM_INTEROP_THREADS=2
ENV TF_NUM_INTRAOP_THREADS=4
ENV OMP_NUM_THREADS=4
ENV ENABLE_MODEL_WARMUP=true
ENV MODEL_CACHE_SIZE=8
ENV TENSORFLOW_ENABLE_MODEL_CACHING=true
ENV TF_ENABLE_EAGER_EXECUTION=false
ENV TF_XLA_FLAGS=--tf_xla_enable_xla_devices

# Install only runtime audio dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    libavcodec59 \
    libavformat59 \
    libavutil57 \
    libswresample4 \
    libtag1v5 \
    libsamplerate0 \
    libchromaprint1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Copy optimized packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY app/ ./app/

# Create directories for audio processing
RUN mkdir -p /app/uploads /app/output /app/temp

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"] 
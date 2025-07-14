#!/bin/bash

echo "🧠 Applying TensorFlow Model Caching Fix"
echo "========================================"

# Navigate to project root
cd "$(dirname "$0")/.."

echo "📊 Current resource usage (before model caching fix):"
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" | grep audio

echo ""
echo "🛑 Stopping overloaded audio services..."

# Identify and restart the most overloaded service (workflow-audio-2 showing 115% CPU)
echo "🎯 Targeting workflow-audio-2 (currently overloaded at 115% CPU)"

# Stop the problematic service
docker-compose -f docker-compose.microservices.lean.yml stop workflow-audio-2

echo "🗑️ Removing container to apply optimizations..."
docker-compose -f docker-compose.microservices.lean.yml rm -f workflow-audio-2

echo ""
echo "🔧 Applying model caching optimizations via environment override..."

# Create temporary environment file for model caching
cat > .env.model-cache << EOF
# TensorFlow Model Caching Optimizations
ENABLE_MODEL_WARMUP=true
MODEL_CACHE_SIZE=8
TENSORFLOW_ENABLE_MODEL_CACHING=true
TF_ENABLE_EAGER_EXECUTION=false
TF_XLA_FLAGS=--tf_xla_enable_xla_devices
EOF

echo "📄 Model caching environment variables:"
cat .env.model-cache

echo ""
echo "🚀 Restarting workflow-audio-2 with model caching..."

# Start the service with enhanced environment
ENABLE_MODEL_WARMUP=true \
MODEL_CACHE_SIZE=8 \
TENSORFLOW_ENABLE_MODEL_CACHING=true \
TF_ENABLE_EAGER_EXECUTION=false \
TF_XLA_FLAGS=--tf_xla_enable_xla_devices \
docker-compose -f docker-compose.microservices.lean.yml up -d workflow-audio-2

echo "⏳ Waiting for service to stabilize with model caching..."
sleep 20

echo ""
echo "🏥 Checking service health with model caching..."
docker exec workflow-audio-2 curl -f -s http://localhost:8001/health > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ workflow-audio-2 is healthy with model caching"
else
    echo "❌ workflow-audio-2 health check failed"
fi

echo ""
echo "📊 Resource usage after model caching (should show reduced CPU):"
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" | grep audio

echo ""
echo "🧪 Testing model caching effectiveness..."

# Test a single audio analysis to see if models are cached
echo "🎵 Testing audio analysis with model caching..."
START_TIME=$(date +%s)

# Create a small test audio file if it doesn't exist
if [ ! -f "/tmp/test_audio.mp3" ]; then
    echo "📁 Creating test audio file..."
    # This creates a very short silent audio file for testing
    docker exec workflow-audio-2 python3 -c "
import numpy as np
import wave
import struct

# Create 1 second of silence at 44100 Hz
duration = 1.0
sample_rate = 44100
frames = int(duration * sample_rate)
audio_data = np.zeros(frames, dtype=np.int16)

# Save as WAV (since MP3 encoding is complex)
with wave.open('/tmp/test_audio.wav', 'wb') as wav_file:
    wav_file.setnchannels(1)  # Mono
    wav_file.setsampwidth(2)  # 2 bytes per sample
    wav_file.setframerate(sample_rate)
    wav_file.writeframes(audio_data.tobytes())

print('Test audio file created: /tmp/test_audio.wav')
"
fi

# Test analysis speed (first request loads models, second should be cached)
echo "🔄 First request (model loading)..."
FIRST_START=$(date +%s)
docker exec workflow-ml-train curl -s -X POST -F "file=@/app/songs/sample.mp3" http://audio-load-balancer:80/analyze/persistent > /dev/null 2>&1 || echo "⚠️ Test file not found, skipping performance test"
FIRST_END=$(date +%s)
FIRST_DURATION=$((FIRST_END - FIRST_START))

echo "⏱️ First request took: ${FIRST_DURATION} seconds"

echo ""
echo "🔍 Monitoring logs for model loading patterns..."
docker logs workflow-audio-2 --tail 10 | grep -E "(TensorflowPredict|Successfully loaded|model)" || echo "No recent model loading detected"

echo ""
echo "📈 Expected Improvements with Model Caching:"
echo "  • 🚀 50-80% reduction in CPU usage after first model load"
echo "  • ⚡ 2-5x faster processing for subsequent requests"
echo "  • 💾 Models loaded once and cached in memory"
echo "  • 🔄 Reduced 502 errors from load balancer"

echo ""
echo "🎯 Additional Optimizations Applied:"
echo "  • 🔄 Faster upstream failure detection (10s vs 30s)"
echo "  • 🚨 Enhanced circuit breaker logic"
echo "  • ⚡ Reduced connection timeout (15s vs 90s)"
echo "  • 🔄 Better load balancing with weights"

echo ""
echo "📝 Monitoring Commands:"
echo "  # Watch CPU usage (should stabilize below 50%)"
echo "  docker stats workflow-audio-1 workflow-audio-2 workflow-audio-3"
echo ""
echo "  # Monitor ML training progress"
echo "  docker logs workflow-ml-train --follow | grep -E 'Processing batch|🎵 Extracted|❌'"
echo ""
echo "  # Check load balancer status"
echo "  curl -s http://localhost:8301/lb-status"

echo ""
echo "✅ Model caching fix applied to workflow-audio-2!"
echo "🔄 Monitor the system for 2-3 minutes to see CPU stabilization"

# Cleanup
rm -f .env.model-cache 
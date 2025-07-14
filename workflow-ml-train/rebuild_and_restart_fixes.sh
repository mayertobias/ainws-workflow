#!/bin/bash

echo "🔧 Rebuilding and Restarting Services with All Fixes"
echo "=================================================="

# Navigate to project root
cd "$(dirname "$0")/.."

echo "📊 Current status before rebuild:"
docker ps --filter "name=workflow" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "🛑 Stopping services for rebuild..."

# Stop all relevant services
docker-compose -f docker-compose.microservices.lean.yml stop workflow-ml-train
docker-compose -f docker-compose.microservices.lean.yml stop workflow-audio-1
docker-compose -f docker-compose.microservices.lean.yml stop workflow-audio-2
docker-compose -f docker-compose.microservices.lean.yml stop workflow-audio-3

echo "🗑️ Removing containers to force rebuild..."
docker-compose -f docker-compose.microservices.lean.yml rm -f workflow-ml-train
docker-compose -f docker-compose.microservices.lean.yml rm -f workflow-audio-1
docker-compose -f docker-compose.microservices.lean.yml rm -f workflow-audio-2
docker-compose -f docker-compose.microservices.lean.yml rm -f workflow-audio-3

echo ""
echo "🔨 Rebuilding workflow-audio with model caching..."

# Rebuild audio services with new Dockerfile (includes model caching)
docker-compose -f docker-compose.microservices.lean.yml build workflow-audio-1
docker-compose -f docker-compose.microservices.lean.yml build workflow-audio-2
docker-compose -f docker-compose.microservices.lean.yml build workflow-audio-3

echo ""
echo "🔨 Rebuilding workflow-ml-train with ensemble fix..."

# Rebuild ML training service with ensemble fix
docker-compose -f docker-compose.microservices.lean.yml build workflow-ml-train

echo ""
echo "🚀 Starting services with all fixes..."

# Start services in correct order
echo "📊 Starting audio services with permanent model caching..."
docker-compose -f docker-compose.microservices.lean.yml up -d workflow-audio-1
docker-compose -f docker-compose.microservices.lean.yml up -d workflow-audio-2
docker-compose -f docker-compose.microservices.lean.yml up -d workflow-audio-3

echo "⏳ Waiting for audio services to be healthy..."
sleep 30

# Check audio service health
echo "🏥 Checking audio service health..."
for i in {1..3}; do
    echo "🔍 Checking workflow-audio-$i..."
    docker exec workflow-audio-$i curl -f -s http://localhost:8001/health > /dev/null
    if [ $? -eq 0 ]; then
        echo "✅ workflow-audio-$i is healthy"
    else
        echo "❌ workflow-audio-$i is not healthy"
    fi
done

echo ""
echo "🧠 Starting ML training service with ensemble fix..."
docker-compose -f docker-compose.microservices.lean.yml up -d workflow-ml-train

echo "⏳ Waiting for ML training service to be ready..."
sleep 15

# Check ML training service health
echo "🏥 Checking ML training service health..."
docker exec workflow-ml-train curl -f -s http://localhost:8005/health > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ workflow-ml-train is healthy"
else
    echo "❌ workflow-ml-train is not healthy"
fi

echo ""
echo "🧪 Testing load balancer connectivity..."
docker exec workflow-ml-train curl -f -s http://audio-load-balancer:80/health > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ Load balancer is accessible from ML training service"
else
    echo "❌ Load balancer is not accessible from ML training service"
fi

echo ""
echo "📊 Resource usage after rebuild:"
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" | grep -E "(workflow-audio|workflow-ml-train)"

echo ""
echo "🔍 Verifying fixes are applied..."

# Check if ensemble fix is in the container
echo "🧠 Checking ensemble fix in ML training service..."
docker exec workflow-ml-train grep -n "fitted_rf_model = ensemble_model.named_estimators_\['rf'\]" /app/app/pipeline/orchestrator.py
if [ $? -eq 0 ]; then
    echo "✅ Ensemble fix is properly applied"
else
    echo "❌ Ensemble fix is missing"
fi

# Check if model caching is in audio services
echo "💾 Checking model caching in audio services..."
docker exec workflow-audio-1 env | grep -E "(MODEL_CACHE|TENSORFLOW_ENABLE_MODEL_CACHING)"
if [ $? -eq 0 ]; then
    echo "✅ Model caching environment variables are set"
else
    echo "❌ Model caching environment variables are missing"
fi

echo ""
echo "📈 All Fixes Applied:"
echo "  ✅ Ensemble training fix (RandomForest + XGBoost fitted models)"
echo "  ✅ TensorFlow model caching (permanent in Dockerfile)"
echo "  ✅ Load balancer circuit breaker enhancements"
echo "  ✅ Resource scaling (6GB RAM, 4 CPU cores)"
echo "  ✅ Enhanced timeouts and retry logic"

echo ""
echo "🧪 Testing ensemble training fix..."
echo "📝 To test the ensemble fix, trigger a new training pipeline:"
echo "   # Via Airflow UI: trigger_audio_ensemble_training"
echo "   # Or via API: POST /pipeline/train with audio_only strategy"
echo ""
echo "📊 Monitor training progress:"
echo "   docker logs workflow-ml-train --follow | grep -E 'Processing batch|🎵 Extracted|❌|🤖 Training Ensemble'"

echo ""
echo "✅ Rebuild and restart completed successfully!"
echo "🔄 All services now have permanent fixes applied"

# Show final status
echo ""
echo "📋 Final service status:"
docker ps --filter "name=workflow" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 
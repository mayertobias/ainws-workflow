#!/bin/bash

echo "🔧 Restarting ML Training Service with Ensemble Fix"
echo "=================================================="

# Stop the service
echo "🛑 Stopping ML training service..."
docker-compose down

# Rebuild to ensure changes are applied
echo "🔨 Rebuilding ML training service..."
docker-compose build

# Start the service
echo "🚀 Starting ML training service..."
docker-compose up -d

# Wait a moment for startup
echo "⏳ Waiting for service startup..."
sleep 10

# Check health
echo "🔍 Checking service health..."
curl -s http://localhost:8005/health | jq '.' || echo "❌ Service not responding"

echo ""
echo "✅ ML Training Service restart complete!"
echo ""
echo "🧪 To test the fix:"
echo "   python test_ensemble_fix.py"
echo ""
echo "🚀 To run ensemble training:"
echo "   1. Open Airflow UI: http://localhost:8080"
echo "   2. Trigger 'trigger_audio_ensemble_training'"
echo "   3. Trigger 'trigger_multimodal_ensemble_training'"
echo ""
echo "📊 Monitor results:"
echo "   - MLflow UI: http://localhost:5001"
echo "   - Service logs: docker-compose logs -f" 
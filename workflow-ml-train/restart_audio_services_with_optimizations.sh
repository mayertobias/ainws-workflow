#!/bin/bash

echo "🔧 Restarting Audio Services with TensorFlow Optimizations"
echo "========================================================="

# Navigate to project root
cd "$(dirname "$0")/.."

echo "📊 Current resource usage before restart:"
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" | grep audio

echo ""
echo "🛑 Stopping audio services and load balancer..."

# Stop services in correct order
docker-compose -f docker-compose.microservices.lean.yml stop audio-load-balancer
docker-compose -f docker-compose.microservices.lean.yml stop workflow-audio-1
docker-compose -f docker-compose.microservices.lean.yml stop workflow-audio-2
docker-compose -f docker-compose.microservices.lean.yml stop workflow-audio-3

echo "🗑️ Removing containers to apply new resource limits..."
docker-compose -f docker-compose.microservices.lean.yml rm -f audio-load-balancer
docker-compose -f docker-compose.microservices.lean.yml rm -f workflow-audio-1
docker-compose -f docker-compose.microservices.lean.yml rm -f workflow-audio-2
docker-compose -f docker-compose.microservices.lean.yml rm -f workflow-audio-3

echo ""
echo "🚀 Starting audio services with new optimizations..."

# Start services in correct order (audio services first, then load balancer)
docker-compose -f docker-compose.microservices.lean.yml up -d workflow-audio-1
docker-compose -f docker-compose.microservices.lean.yml up -d workflow-audio-2
docker-compose -f docker-compose.microservices.lean.yml up -d workflow-audio-3

echo "⏳ Waiting for audio services to be healthy..."
sleep 30

# Check health of audio services
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
echo "🔄 Starting load balancer..."
docker-compose -f docker-compose.microservices.lean.yml up -d audio-load-balancer

echo "⏳ Waiting for load balancer to be ready..."
sleep 10

echo ""
echo "🧪 Testing load balancer connectivity..."
docker exec workflow-ml-train curl -f -s http://audio-load-balancer:80/health > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ Load balancer is accessible from ML training service"
else
    echo "❌ Load balancer is not accessible from ML training service"
fi

echo ""
echo "📊 New resource usage after restart:"
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" | grep audio

echo ""
echo "🎯 Optimizations Applied:"
echo "  • Increased CPU limits: 2.0 → 4.0 cores per service"
echo "  • Increased memory limits: 4G → 6G per service"
echo "  • Added TensorFlow environment optimizations:"
echo "    - TF_NUM_INTEROP_THREADS=2"
echo "    - TF_NUM_INTRAOP_THREADS=4"
echo "    - OMP_NUM_THREADS=4"
echo "    - TF_ENABLE_ONEDNN_OPTS=1"
echo "  • Increased load balancer timeouts: 300s → 600s"
echo "  • Enhanced retry logic: 2 → 3 tries"
echo "  • Increased ML service timeout: 60s → 120s"

echo ""
echo "✅ Audio services restart completed!"
echo ""
echo "🔍 To monitor performance:"
echo "  docker stats workflow-audio-1 workflow-audio-2 workflow-audio-3"
echo ""
echo "📝 To check logs:"
echo "  docker logs workflow-audio-1 --tail 50"
echo "  docker logs workflow-ml-train --tail 50" 
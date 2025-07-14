#!/bin/bash

# Start Workflow Audio Analysis Microservice
# This script builds and starts the audio analysis microservice

set -e

echo "🎵 Starting Workflow Audio Analysis Microservice..."

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p uploads output temp

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Build and start services
echo "🔨 Building and starting services..."
docker-compose up --build -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 15

# Health check
echo "🏥 Checking service health..."
max_attempts=30
attempt=1

while [ $attempt -le $max_attempts ]; do
    if curl -f http://localhost:8001/health > /dev/null 2>&1; then
        echo "✅ Workflow audio service is healthy and ready!"
        echo "🌐 Service available at: http://localhost:8001"
        echo "📊 Health check: http://localhost:8001/health"
        echo "📚 API docs: http://localhost:8001/docs"
        echo "🔧 Service status: http://localhost:8001/status"
        break
    else
        echo "⏳ Attempt $attempt/$max_attempts - Service not ready yet..."
        sleep 2
        attempt=$((attempt + 1))
    fi
done

if [ $attempt -gt $max_attempts ]; then
    echo "❌ Service health check failed after $max_attempts attempts"
    echo "📋 Checking logs..."
    docker-compose logs workflow-audio
    exit 1
fi

echo ""
echo "🎉 Setup complete! Your microservice is running:"
echo "   • Audio Analysis Service: http://localhost:8001"
echo "   • Redis (optional): localhost:6379"
echo ""
echo "📚 Available endpoints:"
echo "   • GET  /health - Health check"
echo "   • GET  /status - Detailed status"
echo "   • POST /analyze/basic - Basic audio analysis"
echo "   • POST /analyze/comprehensive - Full analysis"
echo "   • GET  /extractors - List available extractors"
echo ""
echo "💡 Management commands:"
echo "   • Stop services: docker-compose down"
echo "   • View logs: docker-compose logs -f workflow-audio"
echo "   • Restart: docker-compose restart workflow-audio" 
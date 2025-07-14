#!/bin/bash

# Microservices Stop Script
# This script stops all services in the microservices architecture

set -e

echo "🛑 Stopping Workflow Microservices Architecture"
echo "==============================================="

# Function to stop services gracefully
stop_services() {
    if [ -f "docker-compose.microservices.yml" ]; then
        echo "🔄 Stopping all microservices..."
        docker-compose -f docker-compose.microservices.yml down
        
        echo "✅ All services stopped successfully"
        echo ""
        echo "🧹 Cleanup options:"
        echo "   - Remove volumes: docker-compose -f docker-compose.microservices.yml down -v"
        echo "   - Remove images: docker-compose -f docker-compose.microservices.yml down --rmi all"
        echo "   - Full cleanup: docker system prune -f"
    else
        echo "❌ docker-compose.microservices.yml not found"
        echo "   Please run this script from the project root directory"
        exit 1
    fi
}

# Function to force stop if needed
force_stop() {
    echo "🚨 Force stopping all containers..."
    docker stop $(docker ps -q) 2>/dev/null || echo "No running containers to stop"
    echo "✅ Force stop complete"
}

# Function to clean up resources
cleanup() {
    echo "🧹 Cleaning up Docker resources..."
    
    # Stop and remove containers
    docker-compose -f docker-compose.microservices.yml down -v --remove-orphans 2>/dev/null || true
    
    # Remove unused networks
    docker network prune -f
    
    # Remove unused volumes
    docker volume prune -f
    
    echo "✅ Cleanup complete"
}

# Main execution
case "${1:-stop}" in
    "stop")
        stop_services
        ;;
    "force")
        force_stop
        ;;
    "cleanup")
        cleanup
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [stop|force|cleanup|help]"
        echo ""
        echo "Commands:"
        echo "  stop    - Stop all microservices gracefully (default)"
        echo "  force   - Force stop all running containers"
        echo "  cleanup - Stop services and clean up resources"
        echo "  help    - Show this help message"
        ;;
    *)
        echo "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac 
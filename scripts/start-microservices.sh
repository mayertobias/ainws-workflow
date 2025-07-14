#!/bin/bash

# Enhanced Microservices Startup Script
# This script intelligently starts only needed services in the microservices architecture
# Usage: ./start-microservices.sh [start|status|health|force|help]
# Compatible with bash 3.2+ (macOS default bash)

set -e

# Colors for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Service configuration - using indexed arrays for compatibility
SERVICES=(
    "workflow-gateway:http://localhost:8000/health:8000"
    "workflow-orchestrator:http://localhost:8006/health:8006"
    "workflow-audio:http://localhost:8001/health:8001"
    "workflow-content:http://localhost:8002/health:8002"
    "workflow-ml-training:http://localhost:8003/health:8003"
    "workflow-ml-prediction:http://localhost:8004/health:8004"
    "workflow-intelligence:http://localhost:8005/health:8005"
    "workflow-storage:http://localhost:8007/health:8007"
)

# Infrastructure services (no health endpoints but can check ports)
INFRASTRUCTURE=(
    "postgres-gateway:5437"
    "postgres-orchestrator:5436"
    "postgres-content:5433"
    "postgres-ml-training:5434"
    "postgres-intelligence:5435"
    "postgres-storage:5438"
    "redis-gateway:6380"
    "redis-orchestrator:6381"
    "redis-content:6379"
    "redis-ml:6382"
    "redis-ai:6383"
    "redis-storage:6384"
    "minio:9000"
    "pgadmin:5050"
    "redis-commander:8081"
)

echo -e "${BLUE}🚀 Enhanced Workflow Microservices Manager${NC}"
echo "============================================="

# Function to print colored messages
print_status() {
    local status=$1
    local message=$2
    case $status in
        "success") echo -e "${GREEN}✅ $message${NC}" ;;
        "error") echo -e "${RED}❌ $message${NC}" ;;
        "warning") echo -e "${YELLOW}⚠️  $message${NC}" ;;
        "info") echo -e "${BLUE}ℹ️  $message${NC}" ;;
        "progress") echo -e "${YELLOW}🔄 $message${NC}" ;;
    esac
}

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_status "error" "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    print_status "success" "Docker is running"
}

# Function to check if Docker Compose is available
check_docker_compose() {
    if ! command -v docker-compose > /dev/null 2>&1; then
        print_status "error" "Docker Compose is not installed"
        exit 1
    fi
    print_status "success" "Docker Compose is available"
}

# Function to check if a port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to check service health
check_service_health() {
    local health_url=$1
    
    if curl -f -s "$health_url" > /dev/null 2>&1; then
        return 0  # Service is healthy
    else
        return 1  # Service is not healthy
    fi
}

# Function to get service status
get_service_status() {
    local service_config=$1
    local service_name=$(echo "$service_config" | cut -d':' -f1)
    local health_url=$(echo "$service_config" | cut -d':' -f2)
    
    if check_service_health "$health_url"; then
        echo "running"
    else
        echo "stopped"
    fi
}

# Function to get infrastructure status
get_infrastructure_status() {
    local infra_config=$1
    local service_name=$(echo "$infra_config" | cut -d':' -f1)
    local port=$(echo "$infra_config" | cut -d':' -f2)
    
    if check_port "$port"; then
        echo "running"
    else
        echo "stopped"
    fi
}

# Function to get running services status
get_services_status() {
    local running_services=()
    local stopped_services=()
    local infrastructure_running=()
    local infrastructure_stopped=()
    
    print_status "info" "Checking current service status..."
    
    # Check main services
    for service_config in "${SERVICES[@]}"; do
        local service_name=$(echo "$service_config" | cut -d':' -f1)
        local status=$(get_service_status "$service_config")
        
        if [ "$status" = "running" ]; then
            running_services+=("$service_name")
        else
            stopped_services+=("$service_name")
        fi
    done
    
    # Check infrastructure services
    for infra_config in "${INFRASTRUCTURE[@]}"; do
        local service_name=$(echo "$infra_config" | cut -d':' -f1)
        local status=$(get_infrastructure_status "$infra_config")
        
        if [ "$status" = "running" ]; then
            infrastructure_running+=("$service_name")
        else
            infrastructure_stopped+=("$service_name")
        fi
    done
    
    # Print status summary
    echo ""
    print_status "info" "Service Status Summary:"
    echo "======================="
    
    if [ ${#running_services[@]} -gt 0 ]; then
        echo -e "${GREEN}Running Services (${#running_services[@]}):${NC}"
        for service in "${running_services[@]}"; do
            echo "  ✅ $service"
        done
    fi
    
    if [ ${#stopped_services[@]} -gt 0 ]; then
        echo -e "${RED}Stopped Services (${#stopped_services[@]}):${NC}"
        for service in "${stopped_services[@]}"; do
            echo "  ❌ $service"
        done
    fi
    
    if [ ${#infrastructure_running[@]} -gt 0 ]; then
        echo -e "${GREEN}Running Infrastructure (${#infrastructure_running[@]}):${NC}"
        for service in "${infrastructure_running[@]}"; do
            echo "  ✅ $service"
        done
    fi
    
    if [ ${#infrastructure_stopped[@]} -gt 0 ]; then
        echo -e "${RED}Stopped Infrastructure (${#infrastructure_stopped[@]}):${NC}"
        for service in "${infrastructure_stopped[@]}"; do
            echo "  ❌ $service"
        done
    fi
    
    echo ""
    
    # Export counts for decision making
    export RUNNING_COUNT=${#running_services[@]}
    export STOPPED_COUNT=${#stopped_services[@]}
    export INFRA_RUNNING_COUNT=${#infrastructure_running[@]}
    export INFRA_STOPPED_COUNT=${#infrastructure_stopped[@]}
    
    # Store stopped services for starting
    export STOPPED_SERVICES_LIST="${stopped_services[*]}"
    export STOPPED_INFRA_LIST="${infrastructure_stopped[*]}"
}

# Function to start only needed services
start_needed_services() {
    local force_rebuild=${1:-false}
    
    get_services_status
    
    if [ "$force_rebuild" = true ]; then
        print_status "warning" "Force mode enabled - rebuilding and starting all services"
        docker-compose -f docker-compose.microservices.yml down
        docker-compose -f docker-compose.microservices.yml up -d --build
        
        print_status "progress" "Waiting for all services to start..."
        sleep 15
        
    elif [ "$STOPPED_COUNT" -eq 0 ] && [ "$INFRA_STOPPED_COUNT" -eq 0 ]; then
        print_status "success" "All services are already running!"
        show_service_urls
        return 0
        
    else
        print_status "info" "Starting only needed services..."
        
        # Convert space-separated lists to arrays
        IFS=' ' read -ra stopped_services <<< "$STOPPED_SERVICES_LIST"
        IFS=' ' read -ra stopped_infra <<< "$STOPPED_INFRA_LIST"
        
        # Combine all services that need to be started
        local services_to_start=()
        for service in "${stopped_services[@]}"; do
            if [[ -n "$service" ]]; then
                services_to_start+=("$service")
            fi
        done
        
        for service in "${stopped_infra[@]}"; do
            if [[ -n "$service" ]]; then
                services_to_start+=("$service")
            fi
        done
        
        if [ ${#services_to_start[@]} -gt 0 ]; then
            print_status "progress" "Starting services: ${services_to_start[*]}"
            
            # Start specific services
            docker-compose -f docker-compose.microservices.yml up -d "${services_to_start[@]}"
            
            print_status "progress" "Waiting for services to become ready..."
            sleep 10
        fi
    fi
    
    # Verify services are healthy
    verify_services_health
}

# Function to verify services health after starting
verify_services_health() {
    local max_attempts=30
    local failed_services=()
    
    print_status "info" "Verifying service health..."
    
    for service_config in "${SERVICES[@]}"; do
        local service_name=$(echo "$service_config" | cut -d':' -f1)
        local health_url=$(echo "$service_config" | cut -d':' -f2)
        local attempt=1
        local service_healthy=false
        
        print_status "progress" "Checking $service_name health..."
        
        while [ $attempt -le $max_attempts ]; do
            if check_service_health "$health_url"; then
                print_status "success" "$service_name is healthy"
                service_healthy=true
                break
            fi
            
            if [ $((attempt % 5)) -eq 0 ]; then
                print_status "progress" "Still waiting for $service_name... (attempt $attempt/$max_attempts)"
            fi
            
            sleep 2
            attempt=$((attempt + 1))
        done
        
        if [ "$service_healthy" = false ]; then
            failed_services+=("$service_name")
            print_status "error" "$service_name failed to become healthy"
        fi
    done
    
    if [ ${#failed_services[@]} -gt 0 ]; then
        print_status "error" "Failed services: ${failed_services[*]}"
        print_status "info" "Try checking logs: docker-compose -f docker-compose.microservices.yml logs <service-name>"
        return 1
    else
        print_status "success" "All services are healthy!"
        return 0
    fi
}

# Function to display service URLs and status
show_service_urls() {
    echo ""
    print_status "info" "Service URLs and Access Points:"
    echo "================================="
    
    # Main services
    echo -e "${BLUE}🔗 Main Services:${NC}"
    echo "   - API Gateway:      http://localhost:8000/docs"
    echo "   - Orchestrator:     http://localhost:8006/docs"  
    echo "   - Audio Service:    http://localhost:8001/docs"
    echo "   - Content Service:  http://localhost:8002/docs"
    echo "   - ML Training:      http://localhost:8003/docs"
    echo "   - ML Prediction:    http://localhost:8004/docs"
    echo "   - AI Intelligence:  http://localhost:8005/docs"
    echo "   - Storage Service:  http://localhost:8007/docs"
    
    # Management interfaces
    echo ""
    echo -e "${BLUE}🛠️  Management Interfaces:${NC}"
    echo "   - PgAdmin:          http://localhost:5050"
    echo "   - Redis Commander:  http://localhost:8081"
    echo "   - MinIO Console:    http://localhost:9001"
    
    echo ""
    print_status "info" "Next Steps:"
    echo "   1. Visit http://localhost:8000/docs for the main API gateway"
    echo "   2. Monitor logs: docker-compose -f docker-compose.microservices.yml logs -f"
    echo "   3. Check status: $0 status"
    echo "   4. Stop services: ./stop-microservices.sh"
}

# Function to show detailed service status
show_detailed_status() {
    get_services_status
    
    echo ""
    print_status "info" "Detailed Service Status:"
    echo "========================"
    
    # Check each service with detailed info
    for service_config in "${SERVICES[@]}"; do
        local service_name=$(echo "$service_config" | cut -d':' -f1)
        local health_url=$(echo "$service_config" | cut -d':' -f2)
        
        printf "%-25s " "$service_name:"
        if check_service_health "$health_url"; then
            echo -e "${GREEN}✅ Running${NC} ($health_url)"
        else
            echo -e "${RED}❌ Not responding${NC} ($health_url)"
        fi
    done
    
    echo ""
    echo -e "${BLUE}Infrastructure Services:${NC}"
    for infra_config in "${INFRASTRUCTURE[@]}"; do
        local service_name=$(echo "$infra_config" | cut -d':' -f1)
        local port=$(echo "$infra_config" | cut -d':' -f2)
        
        printf "%-25s " "$service_name:"
        if check_port "$port"; then
            echo -e "${GREEN}✅ Running${NC} (port $port)"
        else
            echo -e "${RED}❌ Not running${NC} (port $port)"
        fi
    done
    
    echo ""
    print_status "info" "Docker Container Status:"
    docker-compose -f docker-compose.microservices.yml ps 2>/dev/null || echo "Docker compose file not found or no containers running"
}

# Function to run health checks
run_health_checks() {
    print_status "info" "Running comprehensive health checks..."
    
    local all_healthy=true
    local unhealthy_services=()
    
    for service_config in "${SERVICES[@]}"; do
        local service_name=$(echo "$service_config" | cut -d':' -f1)
        local health_url=$(echo "$service_config" | cut -d':' -f2)
        
        print_status "progress" "Checking $service_name..."
        if check_service_health "$health_url"; then
            print_status "success" "$service_name is healthy"
        else
            print_status "error" "$service_name is not healthy"
            unhealthy_services+=("$service_name")
            all_healthy=false
        fi
    done
    
    if [ "$all_healthy" = true ]; then
        print_status "success" "All services passed health checks!"
        return 0
    else
        print_status "error" "Unhealthy services: ${unhealthy_services[*]}"
        return 1
    fi
}

# Main execution function
main() {
    local command=${1:-start}
    
    case "$command" in
        "start")
            print_status "info" "Starting microservices with smart detection..."
            check_docker
            check_docker_compose
            
            if [ ! -f "docker-compose.microservices.yml" ]; then
                print_status "error" "docker-compose.microservices.yml not found in current directory"
                print_status "info" "Please run this script from the project root directory"
                exit 1
            fi
            
            start_needed_services false
            
            if [ $? -eq 0 ]; then
                print_status "success" "Microservices startup complete!"
                show_service_urls
            else
                print_status "error" "Some services failed to start properly"
                exit 1
            fi
            ;;
            
        "force")
            print_status "warning" "Force rebuilding all microservices..."
            check_docker
            check_docker_compose
            
            if [ ! -f "docker-compose.microservices.yml" ]; then
                print_status "error" "docker-compose.microservices.yml not found in current directory"
                exit 1
            fi
            
            start_needed_services true
            
            if [ $? -eq 0 ]; then
                print_status "success" "Force rebuild complete!"
                show_service_urls
            else
                print_status "error" "Some services failed during force rebuild"
                exit 1
            fi
            ;;
            
        "status")
            show_detailed_status
            ;;
            
        "health")
            run_health_checks
            ;;
            
        "help"|"-h"|"--help")
            echo -e "${BLUE}Enhanced Microservices Manager${NC}"
            echo "Usage: $0 [command]"
            echo ""
            echo "Commands:"
            echo "  start   - Smart start (only start services that aren't running) [default]"
            echo "  force   - Force rebuild and start all services"
            echo "  status  - Show detailed service status"
            echo "  health  - Run comprehensive health checks"
            echo "  help    - Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                    # Smart start"
            echo "  $0 start              # Smart start" 
            echo "  $0 force              # Force rebuild everything"
            echo "  $0 status             # Check what's running"
            echo "  $0 health             # Verify all services are healthy"
            ;;
            
        *)
            print_status "error" "Unknown command: $command"
            echo "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
}

# Handle script execution
main "$@" 
#!/bin/bash

# Docker Cleanup Script for Workflow Microservices
# This script performs a complete cleanup of Docker resources

set -e

echo "🧹 Workflow Docker Cleanup Script"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    print_success "Docker is running"
}

# Function to stop and remove services using docker-compose
cleanup_docker_compose() {
    print_status "Stopping and removing Docker Compose services..."
    
    if [ -f "docker-compose.microservices.yml" ]; then
        print_status "Stopping microservices..."
        docker-compose -f docker-compose.microservices.yml down --volumes --remove-orphans 2>/dev/null || true
        print_success "Microservices stopped and removed"
    fi
    
    # Check for any other docker-compose files
    for compose_file in docker-compose*.yml; do
        if [ -f "$compose_file" ] && [ "$compose_file" != "docker-compose.microservices.yml" ]; then
            print_status "Stopping services from $compose_file..."
            docker-compose -f "$compose_file" down --volumes --remove-orphans 2>/dev/null || true
        fi
    done
}

# Function to stop all running containers
stop_all_containers() {
    print_status "Stopping all running containers..."
    
    # Get list of running containers
    running_containers=$(docker ps -q)
    
    if [ -n "$running_containers" ]; then
        docker stop $running_containers
        print_success "Stopped $(echo $running_containers | wc -w) containers"
    else
        print_status "No running containers found"
    fi
}

# Function to remove all containers
remove_all_containers() {
    print_status "Removing all containers..."
    
    # Get list of all containers
    all_containers=$(docker ps -aq)
    
    if [ -n "$all_containers" ]; then
        docker rm -f $all_containers
        print_success "Removed $(echo $all_containers | wc -w) containers"
    else
        print_status "No containers found"
    fi
}

# Function to remove workflow-related images
remove_workflow_images() {
    print_status "Removing workflow-related Docker images..."
    
    # Remove images with 'workflow' in the name
    workflow_images=$(docker images --format "table {{.Repository}}:{{.Tag}}" | grep -E "workflow|workflow-" | awk '{print $1}' 2>/dev/null || true)
    
    if [ -n "$workflow_images" ]; then
        echo "$workflow_images" | xargs docker rmi -f 2>/dev/null || true
        print_success "Removed workflow-related images"
    else
        print_status "No workflow-related images found"
    fi
}

# Function to remove all unused images
remove_unused_images() {
    print_status "Removing unused Docker images..."
    
    # Remove dangling images
    dangling_images=$(docker images -f "dangling=true" -q)
    if [ -n "$dangling_images" ]; then
        docker rmi $dangling_images 2>/dev/null || true
        print_success "Removed dangling images"
    fi
    
    # Remove unused images (optional - uncomment if you want aggressive cleanup)
    # docker image prune -a -f
}

# Function to remove all volumes
remove_all_volumes() {
    print_status "Removing all Docker volumes..."
    
    # Remove all volumes
    all_volumes=$(docker volume ls -q)
    
    if [ -n "$all_volumes" ]; then
        docker volume rm $all_volumes 2>/dev/null || true
        print_success "Removed all volumes"
    else
        print_status "No volumes found"
    fi
}

# Function to remove all networks
remove_all_networks() {
    print_status "Removing custom Docker networks..."
    
    # Remove custom networks (keep default ones)
    custom_networks=$(docker network ls --filter type=custom -q)
    
    if [ -n "$custom_networks" ]; then
        docker network rm $custom_networks 2>/dev/null || true
        print_success "Removed custom networks"
    else
        print_status "No custom networks found"
    fi
}

# Function to clean build cache
clean_build_cache() {
    print_status "Cleaning Docker build cache..."
    docker builder prune -a -f 2>/dev/null || true
    print_success "Build cache cleaned"
}

# Function to show disk space before and after
show_disk_usage() {
    print_status "Docker disk usage:"
    docker system df
}

# Function to perform system prune
system_prune() {
    print_status "Performing Docker system prune..."
    docker system prune -a -f --volumes
    print_success "System prune completed"
}

# Main cleanup function
main_cleanup() {
    case "${1:-full}" in
        "containers")
            stop_all_containers
            remove_all_containers
            ;;
        "images")
            remove_workflow_images
            remove_unused_images
            ;;
        "volumes")
            remove_all_volumes
            ;;
        "networks")
            remove_all_networks
            ;;
        "compose")
            cleanup_docker_compose
            ;;
        "cache")
            clean_build_cache
            ;;
        "prune")
            system_prune
            ;;
        "full"|*)
            print_status "Performing full cleanup..."
            show_disk_usage
            echo ""
            
            cleanup_docker_compose
            stop_all_containers
            remove_all_containers
            remove_workflow_images
            remove_unused_images
            remove_all_volumes
            remove_all_networks
            clean_build_cache
            
            echo ""
            print_status "Final disk usage:"
            show_disk_usage
            ;;
    esac
}

# Help function
show_help() {
    echo "Docker Cleanup Script for Workflow Microservices"
    echo ""
    echo "Usage: $0 [option]"
    echo ""
    echo "Options:"
    echo "  full        Complete cleanup (default)"
    echo "  containers  Stop and remove all containers"
    echo "  images      Remove workflow-related and unused images"
    echo "  volumes     Remove all volumes"
    echo "  networks    Remove custom networks"
    echo "  compose     Stop docker-compose services only"
    echo "  cache       Clean build cache"
    echo "  prune       Docker system prune"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 full              # Complete cleanup"
    echo "  $0 containers        # Only remove containers"
    echo "  $0 compose           # Only stop compose services"
}

# Main execution
main() {
    case "${1:-full}" in
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            print_status "Starting Docker cleanup..."
            check_docker
            
            # Confirm for full cleanup
            if [ "${1:-full}" = "full" ]; then
                print_warning "This will remove ALL Docker containers, images, volumes, and networks!"
                read -p "Are you sure you want to continue? (y/N): " -n 1 -r
                echo
                if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                    print_status "Cleanup cancelled"
                    exit 0
                fi
            fi
            
            main_cleanup "$1"
            
            echo ""
            print_success "Docker cleanup completed!"
            print_status "You can now run './start-microservices.sh' to start fresh services"
            ;;
    esac
}

# Run main function with all arguments
main "$@" 
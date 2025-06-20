#!/bin/bash

# 🏥 Radiology AI System - Quick Start Script
# LangChain + LangSmith Local Development Setup

set -e  # Exit on any error

echo "🚀 Starting Radiology AI System Setup..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker Desktop for Windows with WSL 2 integration."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose."
        exit 1
    fi
    
    # Check if Docker is running
    if ! docker info &> /dev/null; then
        print_error "Docker is not running. Please start Docker Desktop."
        exit 1
    fi
    
    print_success "Prerequisites check passed!"
}

# Setup environment
setup_environment() {
    print_status "Setting up environment..."
    
    # Check if .env exists
    if [ ! -f ".env" ]; then
        if [ -f ".env.template" ]; then
            cp .env.template .env
            print_success "Created .env file from template"
        else
            print_error ".env.template not found. Please ensure you have the environment template."
            exit 1
        fi
    else
        print_warning ".env file already exists, skipping creation"
    fi
    
    # Verify API keys are set
    if ! grep -q "LANGCHAIN_API_KEY=lsv2_" .env; then
        print_warning "LangSmith API key not found in .env file"
        print_status "Please ensure your .env file contains all required API keys"
    fi
    
    print_success "Environment setup complete!"
}

# Build Docker images
build_images() {
    print_status "Building Docker images..."
    
    # Build with progress output
    docker-compose build --progress=plain
    
    if [ $? -eq 0 ]; then
        print_success "Docker images built successfully!"
    else
        print_error "Failed to build Docker images"
        exit 1
    fi
}

# Start services
start_services() {
    print_status "Starting services..."
    
    # Start in detached mode
    docker-compose up -d
    
    if [ $? -eq 0 ]; then
        print_success "Services started successfully!"
    else
        print_error "Failed to start services"
        exit 1
    fi
    
    # Wait for services to be ready
    print_status "Waiting for services to be ready..."
    sleep 10
}

# Health checks
run_health_checks() {
    print_status "Running health checks..."
    
    # Check backend
    print_status "Checking backend health..."
    for i in {1..30}; do
        if curl -s http://localhost:8000/health > /dev/null; then
            print_success "Backend is healthy!"
            break
        fi
        if [ $i -eq 30 ]; then
            print_error "Backend health check failed after 30 attempts"
            docker-compose logs backend
            exit 1
        fi
        sleep 2
    done
    
    # Check frontend
    print_status "Checking frontend..."
    for i in {1..30}; do
        if curl -s http://localhost:3000 > /dev/null; then
            print_success "Frontend is accessible!"
            break
        fi
        if [ $i -eq 30 ]; then
            print_error "Frontend health check failed after 30 attempts"
            docker-compose logs frontend
            exit 1
        fi
        sleep 2
    done
    
    # Check database
    print_status "Checking database..."
    if docker-compose exec -T mongodb mongo --eval "db.runCommand({ping: 1})" > /dev/null 2>&1; then
        print_success "Database is healthy!"
    else
        print_warning "Database health check failed, but continuing..."
    fi
}

# Display access information
show_access_info() {
    print_success "🎉 Radiology AI System is ready!"
    echo ""
    echo "📱 Access Points:"
    echo "  🌐 Frontend Application: http://localhost:3000"
    echo "  🔧 Backend API: http://localhost:8000"
    echo "  📚 API Documentation: http://localhost:8000/docs"
    echo "  🗄️  Database Admin: http://localhost:8081"
    echo "  🔴 Redis Admin: http://localhost:8082"
    echo ""
    echo "🔍 LangSmith Dashboard:"
    echo "  📊 Project Dashboard: https://smith.langchain.com/projects/radiology-ai-system"
    echo ""
    echo "🧪 Test the System:"
    echo "  1. Open http://localhost:3000"
    echo "  2. Click 'Analyze Case'"
    echo "  3. Enter a test case"
    echo "  4. Watch real-time traces in LangSmith!"
    echo ""
    echo "📋 Useful Commands:"
    echo "  📜 View logs: docker-compose logs -f"
    echo "  🔄 Restart: docker-compose restart"
    echo "  🛑 Stop: docker-compose down"
    echo "  🧹 Clean: docker-compose down -v && docker system prune -a"
    echo ""
}

# Show service status
show_status() {
    print_status "Service Status:"
    docker-compose ps
    echo ""
    
    print_status "Resource Usage:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"
    echo ""
}

# Main execution
main() {
    echo "🏥 Radiology AI System - LangChain + LangSmith"
    echo "================================================"
    echo ""
    
    check_prerequisites
    setup_environment
    build_images
    start_services
    run_health_checks
    show_status
    show_access_info
    
    print_success "Setup completed successfully! 🎉"
}

# Handle script arguments
case "${1:-}" in
    "start")
        print_status "Starting services..."
        docker-compose up -d
        run_health_checks
        show_access_info
        ;;
    "stop")
        print_status "Stopping services..."
        docker-compose down
        print_success "Services stopped!"
        ;;
    "restart")
        print_status "Restarting services..."
        docker-compose restart
        run_health_checks
        show_access_info
        ;;
    "logs")
        print_status "Showing logs..."
        docker-compose logs -f
        ;;
    "status")
        show_status
        ;;
    "clean")
        print_warning "This will remove all containers, volumes, and images!"
        read -p "Are you sure? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker-compose down -v
            docker system prune -a -f
            print_success "System cleaned!"
        fi
        ;;
    "health")
        run_health_checks
        ;;
    "")
        main
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|logs|status|clean|health}"
        echo ""
        echo "Commands:"
        echo "  start   - Start all services"
        echo "  stop    - Stop all services"
        echo "  restart - Restart all services"
        echo "  logs    - Show service logs"
        echo "  status  - Show service status"
        echo "  clean   - Clean all Docker resources"
        echo "  health  - Run health checks"
        echo "  (none)  - Full setup and start"
        exit 1
        ;;
esac


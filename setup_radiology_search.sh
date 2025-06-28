#!/bin/bash

# =============================================================================
# Docker Connection Fix & Diagnostic Script for RadiologySearch
# =============================================================================
# This script diagnoses and fixes Docker connectivity issues
# Addresses the "URLSchemeUnknown: Not supported URL scheme http+docker" error
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project configuration
PROJECT_DIR="$HOME/radiology-search"

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

print_header() {
    echo -e "\n${BLUE}==============================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}==============================================================================${NC}\n"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to diagnose Docker issues
diagnose_docker() {
    print_header "DIAGNOSING DOCKER ISSUES"
    
    print_status "Current user: $(whoami)"
    print_status "User groups: $(groups)"
    
    # Check if user is in docker group
    if groups $USER | grep -q '\bdocker\b'; then
        print_success "User is in docker group"
    else
        print_error "User is NOT in docker group"
        return 1
    fi
    
    # Check Docker daemon status
    print_status "Checking Docker daemon status..."
    if sudo systemctl is-active --quiet docker; then
        print_success "Docker daemon is running"
    else
        print_error "Docker daemon is not running"
        return 1
    fi
    
    # Check Docker socket permissions
    print_status "Checking Docker socket permissions..."
    if [[ -S /var/run/docker.sock ]]; then
        print_status "Docker socket exists"
        ls -la /var/run/docker.sock
        
        if [[ -r /var/run/docker.sock && -w /var/run/docker.sock ]]; then
            print_success "Docker socket is readable and writable"
        else
            print_error "Docker socket permission issue"
            return 1
        fi
    else
        print_error "Docker socket does not exist"
        return 1
    fi
    
    # Check DOCKER_HOST variable
    if [[ -n "$DOCKER_HOST" ]]; then
        print_warning "DOCKER_HOST is set: $DOCKER_HOST"
        print_status "This might cause connectivity issues"
    else
        print_success "DOCKER_HOST is not set (good)"
    fi
    
    # Test basic Docker command
    print_status "Testing basic Docker command..."
    if docker version >/dev/null 2>&1; then
        print_success "Docker command works"
    else
        print_error "Docker command failed"
        docker version
        return 1
    fi
}

# Function to fix Docker permissions and connectivity
fix_docker() {
    print_header "FIXING DOCKER CONNECTIVITY"
    
    # Stop any existing Docker processes
    print_status "Stopping any existing Docker processes..."
    sudo pkill -f docker-compose || true
    
    # Restart Docker daemon
    print_status "Restarting Docker daemon..."
    sudo systemctl stop docker || true
    sudo systemctl start docker
    sudo systemctl enable docker
    
    # Fix Docker socket permissions
    print_status "Fixing Docker socket permissions..."
    sudo chmod 666 /var/run/docker.sock
    
    # Add user to docker group if not already
    if ! groups $USER | grep -q '\bdocker\b'; then
        print_status "Adding user to docker group..."
        sudo usermod -aG docker $USER
        print_warning "User added to docker group. You may need to logout and login again."
    fi
    
    # Clear any problematic environment variables
    unset DOCKER_HOST
    unset DOCKER_TLS_VERIFY
    unset DOCKER_CERT_PATH
    
    # Test Docker after fixes
    print_status "Testing Docker after fixes..."
    sleep 5
    
    if docker version >/dev/null 2>&1; then
        print_success "Docker is now working"
    else
        print_error "Docker still not working after fixes"
        return 1
    fi
}

# Function to update Docker Compose
update_docker_compose() {
    print_header "UPDATING DOCKER COMPOSE"
    
    # Check current Docker Compose version
    if command_exists docker-compose; then
        print_status "Current docker-compose version:"
        docker-compose --version
    fi
    
    # Remove old docker-compose
    print_status "Removing old docker-compose..."
    sudo apt-get remove -y docker-compose || true
    sudo rm -f /usr/local/bin/docker-compose || true
    sudo rm -f /usr/bin/docker-compose || true
    
    # Install latest Docker Compose V2
    print_status "Installing Docker Compose V2..."
    sudo apt-get update
    sudo apt-get install -y docker-compose-plugin
    
    # Create symlink for backward compatibility
    if [[ ! -f /usr/local/bin/docker-compose ]]; then
        print_status "Creating docker-compose symlink..."
        sudo ln -s /usr/libexec/docker/cli-plugins/docker-compose /usr/local/bin/docker-compose || \
        sudo ln -s /usr/lib/docker/cli-plugins/docker-compose /usr/local/bin/docker-compose || \
        print_warning "Could not create symlink, using 'docker compose' instead"
    fi
    
    # Test Docker Compose
    print_status "Testing Docker Compose..."
    if docker compose version >/dev/null 2>&1; then
        print_success "Docker Compose V2 is working"
        docker compose version
    elif command_exists docker-compose && docker-compose --version >/dev/null 2>&1; then
        print_success "Docker Compose (legacy) is working"
        docker-compose --version
    else
        print_error "Docker Compose is not working"
        return 1
    fi
}

# Function to test Docker connectivity thoroughly
test_docker_connectivity() {
    print_header "TESTING DOCKER CONNECTIVITY"
    
    # Test Docker daemon
    print_status "Testing Docker daemon connectivity..."
    if docker info >/dev/null 2>&1; then
        print_success "Docker daemon is accessible"
    else
        print_error "Cannot connect to Docker daemon"
        docker info
        return 1
    fi
    
    # Test Docker run
    print_status "Testing Docker run..."
    if docker run --rm hello-world >/dev/null 2>&1; then
        print_success "Docker run test passed"
    else
        print_error "Docker run test failed"
        return 1
    fi
    
    # Test Docker Compose
    print_status "Testing Docker Compose..."
    cd "$PROJECT_DIR" || return 1
    
    # Try V2 first
    if docker compose version >/dev/null 2>&1; then
        print_status "Using Docker Compose V2"
        COMPOSE_CMD="docker compose"
    elif docker-compose --version >/dev/null 2>&1; then
        print_status "Using Docker Compose V1"
        COMPOSE_CMD="docker-compose"
    else
        print_error "No working Docker Compose found"
        return 1
    fi
    
    # Test compose config
    if $COMPOSE_CMD config >/dev/null 2>&1; then
        print_success "Docker Compose configuration is valid"
    else
        print_error "Docker Compose configuration is invalid"
        $COMPOSE_CMD config
        return 1
    fi
}

# Function to rebuild project with fixed Docker
rebuild_project() {
    print_header "REBUILDING RADIOLOGY SEARCH PROJECT"
    
    cd "$PROJECT_DIR" || {
        print_error "Project directory not found: $PROJECT_DIR"
        return 1
    }
    
    # Determine which compose command to use
    if docker compose version >/dev/null 2>&1; then
        COMPOSE_CMD="docker compose"
    elif docker-compose --version >/dev/null 2>&1; then
        COMPOSE_CMD="docker-compose"
    else
        print_error "No working Docker Compose found"
        return 1
    fi
    
    print_status "Using command: $COMPOSE_CMD"
    
    # Stop any existing containers
    print_status "Stopping existing containers..."
    $COMPOSE_CMD down || true
    
    # Remove any orphaned containers
    print_status "Removing orphaned containers..."
    docker container prune -f || true
    
    # Build images
    print_status "Building Docker images..."
    if $COMPOSE_CMD build --no-cache; then
        print_success "Docker images built successfully"
    else
        print_error "Failed to build Docker images"
        return 1
    fi
    
    # Start services
    print_status "Starting services..."
    if $COMPOSE_CMD up -d; then
        print_success "Services started successfully"
    else
        print_error "Failed to start services"
        return 1
    fi
    
    # Wait for services
    print_status "Waiting for services to be ready..."
    sleep 30
    
    # Check service status
    print_status "Checking service status..."
    $COMPOSE_CMD ps
}

# Function to validate services
validate_services() {
    print_header "VALIDATING SERVICES"
    
    cd "$PROJECT_DIR" || return 1
    
    # Determine compose command
    if docker compose version >/dev/null 2>&1; then
        COMPOSE_CMD="docker compose"
    else
        COMPOSE_CMD="docker-compose"
    fi
    
    # Check container health
    print_status "Container status:"
    $COMPOSE_CMD ps
    
    # Test frontend
    print_status "Testing Frontend..."
    local max_attempts=10
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -s http://localhost:3000 >/dev/null 2>&1; then
            print_success "Frontend is accessible at http://localhost:3000"
            break
        else
            if [[ $attempt -eq $max_attempts ]]; then
                print_error "Frontend is not accessible"
                print_status "Frontend logs:"
                $COMPOSE_CMD logs frontend | tail -20
                return 1
            fi
            print_status "Attempt $attempt/$max_attempts - Waiting for frontend..."
            sleep 5
            ((attempt++))
        fi
    done
    
    # Test backend
    print_status "Testing Backend..."
    attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -s http://localhost:8000/health >/dev/null 2>&1; then
            print_success "Backend is accessible at http://localhost:8000"
            
            # Get health status
            local health_response=$(curl -s http://localhost:8000/health)
            print_status "Backend health: $health_response"
            break
        else
            if [[ $attempt -eq $max_attempts ]]; then
                print_error "Backend is not accessible"
                print_status "Backend logs:"
                $COMPOSE_CMD logs backend | tail -20
                return 1
            fi
            print_status "Attempt $attempt/$max_attempts - Waiting for backend..."
            sleep 5
            ((attempt++))
        fi
    done
    
    # Test database
    print_status "Testing MongoDB..."
    if $COMPOSE_CMD exec -T mongodb mongosh --eval "db.adminCommand('ping')" >/dev/null 2>&1; then
        print_success "MongoDB is responding"
    else
        print_warning "MongoDB may not be fully ready"
        print_status "MongoDB logs:"
        $COMPOSE_CMD logs mongodb | tail -10
    fi
    
    # Test Redis
    print_status "Testing Redis..."
    if $COMPOSE_CMD exec -T redis redis-cli ping | grep -q "PONG"; then
        print_success "Redis is responding"
    else
        print_warning "Redis may not be fully ready"
        print_status "Redis logs:"
        $COMPOSE_CMD logs redis | tail -10
    fi
}

# Function to create a new, simpler docker-compose file
create_fixed_docker_compose() {
    print_header "CREATING FIXED DOCKER COMPOSE CONFIGURATION"
    
    cd "$PROJECT_DIR" || return 1
    
    # Backup existing docker-compose.yml
    if [[ -f docker-compose.yml ]]; then
        print_status "Backing up existing docker-compose.yml..."
        cp docker-compose.yml "docker-compose.yml.backup.$(date +%Y%m%d_%H%M%S)"
    fi
    
    # Create a simplified, working docker-compose.yml
    print_status "Creating simplified docker-compose.yml..."
    cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  mongodb:
    image: mongo:7.0
    container_name: radiology_mongodb
    restart: unless-stopped
    ports:
      - "27017:27017"
    environment:
      MONGO_INITDB_ROOT_USERNAME: radiology_app
      MONGO_INITDB_ROOT_PASSWORD: radiology_secure_2024
      MONGO_INITDB_DATABASE: radiology_ai_langchain
    volumes:
      - mongodb_data:/data/db
    command: mongod --auth
    healthcheck:
      test: ["CMD", "mongosh", "--quiet", "--eval", "db.adminCommand('ping')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s

  redis:
    image: redis:7-alpine
    container_name: radiology_redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  backend:
    build: 
      context: ./backend
      dockerfile: Dockerfile
    container_name: radiology_backend
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - PYTHONPATH=/app
      - PYTHONUNBUFFERED=1
    env_file:
      - .env
    volumes:
      - ./backend:/app:ro
      - ./data/uploads:/app/uploads
      - ./logs/backend:/app/logs
    depends_on:
      mongodb:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  frontend:
    build: 
      context: ./frontend
      dockerfile: Dockerfile
    container_name: radiology_frontend
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=development
      - REACT_APP_BACKEND_URL=http://localhost:8000
    volumes:
      - ./frontend:/app:ro
      - /app/node_modules
      - ./logs/frontend:/app/logs
    depends_on:
      backend:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

volumes:
  mongodb_data:
    driver: local
  redis_data:
    driver: local

networks:
  default:
    name: radiology_network
EOF

    print_success "Fixed docker-compose.yml created"
}

# Function to show final status
show_final_status() {
    print_header "DOCKER FIX COMPLETE"
    
    cd "$PROJECT_DIR" || return 1
    
    # Determine compose command
    if docker compose version >/dev/null 2>&1; then
        COMPOSE_CMD="docker compose"
    else
        COMPOSE_CMD="docker-compose"
    fi
    
    print_success "Docker connectivity issues have been resolved!"
    
    echo -e "\n${GREEN}🐳 Docker Status:${NC}"
    docker version --format "Client: {{.Client.Version}} | Server: {{.Server.Version}}"
    echo -e "Compose: $(docker compose version --short 2>/dev/null || docker-compose --version 2>/dev/null)"
    
    echo -e "\n${GREEN}📦 Container Status:${NC}"
    $COMPOSE_CMD ps
    
    echo -e "\n${GREEN}🌐 Service URLs:${NC}"
    echo -e "  Frontend:      ${BLUE}http://localhost:3000${NC}"
    echo -e "  Backend API:   ${BLUE}http://localhost:8000${NC}"
    echo -e "  API Docs:      ${BLUE}http://localhost:8000/docs${NC}"
    echo -e "  Health Check:  ${BLUE}http://localhost:8000/health${NC}"
    
    echo -e "\n${GREEN}🔧 Management Commands:${NC}"
    echo -e "  View Status:   ${BLUE}cd $PROJECT_DIR && $COMPOSE_CMD ps${NC}"
    echo -e "  View Logs:     ${BLUE}cd $PROJECT_DIR && $COMPOSE_CMD logs -f${NC}"
    echo -e "  Restart:       ${BLUE}cd $PROJECT_DIR && $COMPOSE_CMD restart${NC}"
    echo -e "  Stop:          ${BLUE}cd $PROJECT_DIR && $COMPOSE_CMD down${NC}"
    echo -e "  Start:         ${BLUE}cd $PROJECT_DIR && $COMPOSE_CMD up -d${NC}"
    
    echo -e "\n${YELLOW}💡 Tips:${NC}"
    echo -e "  - If you get permission errors, run: ${BLUE}newgrp docker${NC}"
    echo -e "  - Or logout and login again to refresh group membership"
    echo -e "  - Use '${BLUE}$COMPOSE_CMD${NC}' for all Docker Compose commands"
}

# Main execution function
main() {
    print_header "DOCKER CONNECTION FIX SCRIPT"
    print_status "Diagnosing and fixing Docker connectivity issues..."
    
    # Check if project directory exists
    if [[ ! -d "$PROJECT_DIR" ]]; then
        print_error "Project directory not found: $PROJECT_DIR"
        print_status "Please run the main setup script first"
        exit 1
    fi
    
    # Diagnose issues
    if ! diagnose_docker; then
        print_status "Docker issues detected. Attempting to fix..."
        fix_docker
    fi
    
    # Update Docker Compose if needed
    update_docker_compose
    
    # Test connectivity
    if ! test_docker_connectivity; then
        print_error "Docker connectivity still has issues"
        exit 1
    fi
    
    # Create fixed configuration
    create_fixed_docker_compose
    
    # Rebuild project
    if ! rebuild_project; then
        print_error "Failed to rebuild project"
        exit 1
    fi
    
    # Validate services
    if ! validate_services; then
        print_warning "Some services may not be fully ready"
    fi
    
    # Show final status
    show_final_status
    
    print_success "Docker fix completed successfully! 🎉"
}

# Handle errors
trap 'print_error "Script failed at line $LINENO"' ERR

# Run main function
main "$@"

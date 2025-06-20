# 🏥 Radiology AI System - Complete WSL Setup Guide

## 📋 Prerequisites

### Windows WSL Setup
1. **Enable WSL 2**:
   ```powershell
   # Run as Administrator in PowerShell
   wsl --install
   # Or if already installed:
   wsl --set-default-version 2
   ```

2. **Install Ubuntu 22.04**:
   ```powershell
   wsl --install -d Ubuntu-22.04
   ```

3. **Verify Docker Desktop**:
   - Install Docker Desktop for Windows
   - Enable WSL 2 integration
   - Enable integration with Ubuntu-22.04

### System Requirements
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: 10GB free space
- **CPU**: Multi-core processor (4+ cores recommended)

## 🚀 Installation Steps

### 1. Clone and Setup Project
```bash
# In WSL Ubuntu terminal
cd ~
git clone <your-repo-url> radiology-langchain-system
cd radiology-langchain-system

# Make scripts executable
chmod +x scripts/*.sh
```

### 2. Environment Configuration
```bash
# Copy environment template
cp .env.template .env

# Edit with your API keys (already configured)
nano .env
```

### 3. Docker Setup Verification
```bash
# Test Docker installation
docker --version
docker-compose --version

# Test Docker connectivity
docker run hello-world
```

### 4. Build and Start Services
```bash
# Build all services
docker-compose build

# Start all services
docker-compose up -d

# Check service status
docker-compose ps
```

### 5. Verify Installation
```bash
# Check backend health
curl http://localhost:8000/health

# Check frontend
curl http://localhost:3000

# Check database
curl http://localhost:8081  # MongoDB Express
```

## 🔧 LangChain Studio Integration

### 1. Install LangChain Studio
```bash
# Install LangChain CLI
pip install langchain-cli

# Install LangChain Studio
npm install -g @langchain/studio
```

### 2. Connect to Local System
```bash
# Start LangChain Studio
langchain studio

# Configure connection:
# - Backend URL: http://localhost:8000
# - LangSmith Project: radiology-ai-system
# - API Key: (your LangSmith key)
```

### 3. Studio Configuration
1. **Open Studio**: http://localhost:3001
2. **Connect Backend**: Point to http://localhost:8000
3. **Import Agents**: Load radiology agents
4. **Configure Tracing**: Enable LangSmith integration

## 📊 LangSmith Dashboard Access

### 1. Web Dashboard
- **URL**: https://smith.langchain.com/projects/radiology-ai-system
- **Login**: Use your LangSmith account
- **Project**: radiology-ai-system

### 2. Local Integration
```bash
# Test LangSmith connection
curl -H "Authorization: Bearer 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" \
     https://api.smith.langchain.com/projects
```

### 3. Real-time Monitoring
- **Traces**: View all agent executions
- **Metrics**: Performance and accuracy tracking
- **Debugging**: Step-by-step agent reasoning
- **Optimization**: Prompt performance analysis

## 🛠️ Development Workflow

### 1. Start Development Environment
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f backend
docker-compose logs -f frontend
```

### 2. Code Development
```bash
# Backend development (hot reload enabled)
cd backend
# Edit files in your IDE
# Changes auto-reload in container

# Frontend development
cd frontend
# Edit React components
# Changes auto-reload via webpack
```

### 3. Prompt Customization
1. **Web Interface**: http://localhost:3000/prompts
2. **Direct Database**: http://localhost:8081 (MongoDB Express)
3. **API Endpoints**: http://localhost:8000/api/prompts

### 4. Testing and Debugging
```bash
# Run backend tests
docker-compose exec backend pytest

# Check LangSmith traces
# Visit: https://smith.langchain.com/projects/radiology-ai-system

# Debug specific agent
docker-compose exec backend python -c "
from main import radiology_system
import asyncio
# Debug code here
"
```

## 📱 Access Points

### Main Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### Development Tools
- **MongoDB Express**: http://localhost:8081
- **Redis Commander**: http://localhost:8082
- **LangChain Studio**: http://localhost:3001

### Cloud Services
- **LangSmith Dashboard**: https://smith.langchain.com/projects/radiology-ai-system
- **LangChain Hub**: https://smith.langchain.com/hub

## 🔍 Monitoring and Observability

### 1. LangSmith Tracing
- **Real-time Traces**: Every agent call tracked
- **Performance Metrics**: Latency, token usage, costs
- **Error Tracking**: Failed executions and debugging
- **Custom Evaluations**: Medical accuracy scoring

### 2. Application Logs
```bash
# View all logs
docker-compose logs -f

# Specific service logs
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f mongodb
```

### 3. System Metrics
```bash
# Container resource usage
docker stats

# Database status
docker-compose exec mongodb mongo --eval "db.stats()"

# Redis status
docker-compose exec redis redis-cli info
```

## 🧪 Testing the System

### 1. Basic Health Check
```bash
# Test all endpoints
curl http://localhost:8000/health
curl http://localhost:3000
curl http://localhost:8081
```

### 2. AI Analysis Test
1. **Open Frontend**: http://localhost:3000
2. **Navigate**: Click "Analyze Case"
3. **Enter Test Case**:
   - Age: 45
   - Sex: Male
   - Modality: MRI
   - Region: Brain
   - Clinical History: "Severe headaches, visual disturbances"
   - Image Description: "T1-weighted MRI shows ring-enhancing lesion"
4. **Start Analysis**: Click "Start AI Analysis"
5. **Monitor**: Watch LangSmith dashboard for real-time traces

### 3. Prompt Customization Test
1. **Open Prompts**: http://localhost:3000/prompts
2. **Select Prompt**: Choose "radiology_context_extractor"
3. **Edit**: Modify prompt text
4. **Save**: Create new version
5. **Test**: Run analysis with new prompt

## 🔧 Troubleshooting

### Common Issues

#### Docker Issues
```bash
# Reset Docker
docker-compose down -v
docker system prune -a
docker-compose up --build

# Check Docker resources
docker system df
```

#### Port Conflicts
```bash
# Check port usage
netstat -tulpn | grep :8000
netstat -tulpn | grep :3000

# Kill processes if needed
sudo kill -9 $(lsof -t -i:8000)
```

#### Database Connection
```bash
# Reset MongoDB
docker-compose restart mongodb
docker-compose exec mongodb mongo --eval "db.runCommand({ping: 1})"
```

#### LangSmith Connection
```bash
# Test API key
curl -H "Authorization: Bearer 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" \
     https://api.smith.langchain.com/projects

# Check environment variables
docker-compose exec backend env | grep LANGCHAIN
```

### Performance Optimization

#### Memory Usage
```bash
# Increase Docker memory limit
# Docker Desktop > Settings > Resources > Memory: 8GB+

# Monitor container memory
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
```

#### Database Performance
```bash
# MongoDB optimization
docker-compose exec mongodb mongo --eval "
db.prompts.createIndex({template_id: 1, version: -1});
db.cases.createIndex({case_id: 1, timestamp: -1});
"
```

## 📚 Additional Resources

### Documentation
- **LangChain Docs**: https://python.langchain.com/docs/
- **LangSmith Docs**: https://docs.smith.langchain.com/
- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **React Docs**: https://react.dev/

### Community
- **LangChain Discord**: https://discord.gg/langchain
- **GitHub Issues**: Report bugs and feature requests
- **Stack Overflow**: Tag questions with `langchain`

### Advanced Configuration
- **Custom Models**: Add new AI model integrations
- **Custom Tools**: Implement specialized medical tools
- **Evaluation Metrics**: Create custom medical accuracy evaluators
- **Deployment**: Production deployment guides

## 🎯 Next Steps

1. **Explore LangSmith**: Analyze agent traces and optimize prompts
2. **Customize Prompts**: Tailor agents for specific medical specialties
3. **Add Evaluations**: Create medical accuracy benchmarks
4. **Scale System**: Deploy to production environment
5. **Integrate Tools**: Add specialized medical databases and APIs

## 📞 Support

For technical support:
1. **Check Logs**: Review Docker and application logs
2. **LangSmith Dashboard**: Monitor traces for errors
3. **GitHub Issues**: Report bugs with detailed logs
4. **Documentation**: Refer to official LangChain/LangSmith docs

---

**🎉 Congratulations!** You now have a complete LangChain + LangSmith radiology AI system running locally with full observability and customization capabilities.


# 🏥 Radiology AI System - LangChain + LangSmith

A sophisticated AI-powered radiology image retrieval and case analysis system built with LangChain, LangSmith, and multi-model AI integration.

![System Architecture](https://img.shields.io/badge/LangChain-Enabled-blue) ![LangSmith](https://img.shields.io/badge/LangSmith-Integrated-green) ![Docker](https://img.shields.io/badge/Docker-Ready-blue) ![AI Models](https://img.shields.io/badge/AI-Claude%20%7C%20Mistral%20%7C%20DeepSeek-orange)

## 🌟 Features

### 🧠 **Advanced Multi-Agent AI System**
- **RadiologyContextExtractor**: Extracts structured radiology context (anatomy, modality, measurements)
- **LiteratureSearchAgent**: Intelligent medical literature search and analysis
- **DiagnosisAgent**: Evidence-based diagnosis generation with confidence scoring
- **MultiModelOrchestrator**: Coordinates Claude, Mistral, and DeepSeek optimally

### 🔍 **Complete Observability with LangSmith**
- **Real-time Tracing**: Every agent call, prompt, and decision tracked
- **Performance Monitoring**: Latency, token usage, success rates
- **Custom Evaluations**: Medical accuracy scoring framework
- **Prompt Optimization**: A/B testing and version comparison

### 📄 **Sophisticated Document Processing**
- **PDF Text Extraction**: Full medical paper content analysis
- **Image Caption Extraction**: Finds textual descriptions of medical images
- **Relevance Scoring**: AI-powered document relevance assessment
- **Literature Caching**: Optimized search result storage

### 🎨 **Modern Web Interface**
- **Real-time Dashboard**: System status and LangSmith integration
- **Case Analysis Interface**: Professional medical form with validation
- **Prompt Editor**: Live prompt customization and testing
- **Results Visualization**: Comprehensive analysis results display

### 🐳 **Production-Ready Deployment**
- **Docker Containerization**: Multi-service architecture
- **WSL Integration**: Optimized for Windows development
- **Health Monitoring**: Comprehensive system monitoring
- **Auto-scaling Ready**: Production deployment configuration

## 🚀 Quick Start

### Prerequisites
- **Windows with WSL 2** (Ubuntu 22.04 recommended)
- **Docker Desktop** with WSL 2 integration enabled
- **Git** for cloning the repository
- **API Keys** for LangSmith, Claude, Mistral, and Brave Search

### Installation

1. **Clone the Repository**
   ```bash
   git clone <your-repo-url>
   cd radiology-langchain-system
   ```

2. **Configure Environment**
   ```bash
   # Copy environment template
   cp .env.template .env
   
   # Edit with your API keys
   nano .env
   ```

3. **Start the System**
   ```bash
   # Make startup script executable
   chmod +x scripts/start.sh
   
   # Start all services
   ./scripts/start.sh
   ```

4. **Access the Application**
   - **Frontend**: http://localhost:3000
   - **Backend API**: http://localhost:8000
   - **API Docs**: http://localhost:8000/docs
   - **LangSmith**: https://smith.langchain.com/projects/radiology-ai-system

## 🔑 API Keys Required

Add these to your `.env` file:

```bash
# LangSmith (Required for observability)
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=radiology-ai-system

# AI Models (Required)
ANTHROPIC_API_KEY=your_claude_api_key
MISTRAL_API_KEY=your_mistral_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key

# Search (Required)
BRAVE_SEARCH_API_KEY=your_brave_search_api_key

# Database (Auto-configured)
MONGODB_URL=mongodb://radiology_app:radiology_secure_2024@mongodb:27017/radiology_ai_langchain
REDIS_URL=redis://redis:6379/0
```

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Frontend │    │  FastAPI Backend │    │   MongoDB       │
│   (Port 3000)   │◄──►│   (Port 8000)    │◄──►│   (Port 27017)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         │              │  LangSmith      │              │
         └──────────────►│  Observability │◄─────────────┘
                        │  (Cloud)        │
                        └─────────────────┘
                                 │
                    ┌─────────────────────────────┐
                    │     AI Model APIs           │
                    │  ┌─────────┐ ┌─────────┐   │
                    │  │ Claude  │ │ Mistral │   │
                    │  └─────────┘ └─────────┘   │
                    │  ┌─────────┐ ┌─────────┐   │
                    │  │DeepSeek │ │ Brave   │   │
                    │  └─────────┘ └─────────┘   │
                    └─────────────────────────────┘
```

## 🧪 Testing the System

### 1. Health Check
```bash
curl http://localhost:8000/health
```

### 2. Complete Case Analysis
1. Open http://localhost:3000
2. Click "Analyze Case"
3. Enter test case:
   - **Age**: 45
   - **Sex**: Male
   - **Modality**: MRI
   - **Region**: Brain
   - **Clinical History**: "Severe headaches, visual disturbances"
   - **Image Description**: "T1-weighted MRI shows ring-enhancing lesion"
4. Click "Start AI Analysis"
5. Monitor real-time traces in LangSmith dashboard

### 3. Prompt Customization
1. Open http://localhost:3000/prompts
2. Select a prompt template
3. Edit and test modifications
4. Monitor performance in LangSmith

## 📊 LangSmith Integration

### Real-time Observability
- **Agent Traces**: Step-by-step reasoning for each AI agent
- **Performance Metrics**: Response times, token usage, costs
- **Error Tracking**: Failed executions with detailed debugging
- **Custom Evaluations**: Medical accuracy and relevance scoring

### Prompt Optimization
- **Version Control**: Track prompt changes and performance
- **A/B Testing**: Compare different prompt variations
- **Performance Analytics**: Identify best-performing prompts
- **Custom Metrics**: Define medical-specific evaluation criteria

## 🛠️ Development

### Project Structure
```
radiology-langchain-system/
├── backend/                 # FastAPI backend
│   ├── main.py             # Main application
│   ├── agents/             # AI agent implementations
│   ├── chains/             # LangChain chains
│   ├── prompts/            # Prompt templates
│   ├── tools/              # Custom tools
│   └── evaluations/        # Custom evaluators
├── frontend/               # React frontend
│   ├── src/
│   │   ├── App.js          # Main React component
│   │   └── App.css         # Styling
│   └── public/
├── docker/                 # Docker configurations
├── scripts/                # Utility scripts
└── docs/                   # Documentation
```

### Development Commands
```bash
# Start development environment
./scripts/start.sh

# View logs
docker-compose logs -f

# Restart services
./scripts/start.sh restart

# Stop services
./scripts/start.sh stop

# Clean environment
./scripts/start.sh clean
```

### Adding Custom Agents
1. Create agent in `backend/agents/`
2. Add prompt templates in `backend/prompts/`
3. Register in main orchestrator
4. Test with LangSmith tracing

### Custom Evaluations
1. Define evaluator in `backend/evaluations/`
2. Configure in LangSmith project
3. Run evaluation on test cases
4. Monitor performance metrics

## 🔧 Configuration

### Environment Variables
- **LANGCHAIN_TRACING_V2**: Enable LangSmith tracing
- **LANGCHAIN_API_KEY**: LangSmith API key
- **LANGCHAIN_PROJECT**: Project name for tracing
- **MODEL_TEMPERATURE**: AI model temperature (0.0-1.0)
- **MAX_TOKENS**: Maximum tokens per response
- **SEARCH_RESULTS_LIMIT**: Number of search results to process

### Database Configuration
- **MongoDB**: Document storage for cases, prompts, results
- **Redis**: Caching for search results and sessions
- **Indexes**: Optimized for medical data queries

### Model Configuration
- **Claude**: Medical reasoning and diagnosis
- **Mistral**: Document processing and analysis
- **DeepSeek**: Search optimization and code generation
- **Brave Search**: Medical literature retrieval

## 📈 Performance Optimization

### Caching Strategy
- **Literature Cache**: 24-hour TTL for search results
- **Prompt Cache**: Version-based caching
- **Model Response Cache**: Configurable TTL

### Monitoring
- **Health Checks**: Automated service monitoring
- **Performance Metrics**: Response time tracking
- **Resource Usage**: Memory and CPU monitoring
- **Error Rates**: Failure tracking and alerting

## 🚀 Production Deployment

### AWS ECS Deployment
```bash
# Build for production
docker-compose -f docker-compose.prod.yml build

# Deploy to ECS
aws ecs update-service --cluster radiology-ai --service radiology-backend
```

### Environment-Specific Configurations
- **Development**: Local Docker with hot reload
- **Staging**: Cloud deployment with test data
- **Production**: Scaled deployment with monitoring

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with proper testing
4. Ensure LangSmith traces are clean
5. Submit a pull request

### Code Standards
- **Python**: Follow PEP 8 with Black formatting
- **JavaScript**: ESLint with Prettier
- **Documentation**: Update README for new features
- **Testing**: Include unit and integration tests

## 📚 Documentation

- **API Documentation**: http://localhost:8000/docs
- **LangChain Docs**: https://python.langchain.com/docs/
- **LangSmith Docs**: https://docs.smith.langchain.com/
- **Setup Guide**: [WSL_SETUP_GUIDE.md](WSL_SETUP_GUIDE.md)

## 🐛 Troubleshooting

### Common Issues
- **Port Conflicts**: Check for running services on ports 3000, 8000, 27017
- **Docker Issues**: Restart Docker Desktop and WSL
- **API Key Errors**: Verify all keys in `.env` file
- **LangSmith Connection**: Check API key and project name

### Debug Commands
```bash
# Check service status
docker-compose ps

# View service logs
docker-compose logs backend

# Test API connectivity
curl http://localhost:8000/health

# Check LangSmith connection
curl -H "Authorization: Bearer $LANGCHAIN_API_KEY" https://api.smith.langchain.com/projects
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain**: For the powerful AI framework
- **LangSmith**: For comprehensive observability
- **Anthropic**: For Claude AI capabilities
- **Mistral AI**: For document processing
- **Brave Search**: For privacy-focused search

---

**🏥 Built for Medical Professionals** | **🔬 Powered by AI** | **📊 Observable with LangSmith**


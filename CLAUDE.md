# Context for Claude - RadiologySearch AI System

## Quick Summary
You're working on a **RadiologySearch AI System** that uses LangChain to orchestrate multiple AI models (Claude, Mistral, DeepSeek) for analyzing radiology cases. The system searches medical literature, extracts relevant images, and provides comprehensive diagnostic assessments with full LangSmith observability.

## Current State
- ✅ System is fully operational with all enhancements
- ✅ Enhanced literature search finds 10+ papers with images
- ✅ Prompt manager allows real-time customization of all AI agents
- ✅ Frontend has image carousel and enhanced UI
- ✅ All models properly integrated with LangSmith tracing

## Key Technical Details

### Architecture
```
Frontend (React:3000) → FastAPI Backend (8000) → AI Agents → MongoDB/Redis
                                                     ↓
                                                LangSmith Tracing
```

### File Structure
```
radiology-langchain-system/
├── backend/
│   ├── main.py                    # Main API with prompt manager
│   ├── enhanced_literature_search.py  # Image extraction logic
│   ├── prompt_templates.py        # Default prompts
│   ├── initialize_prompts.py      # DB initialization
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.js                # Main React app
│   │   ├── ImageCarousel.js      # Image viewer
│   │   └── api.js                # Axios config
│   └── Dockerfile
├── docker-compose.yml
└── .env                          # API keys
```

### Models & Their Roles
- **Claude**: Medical reasoning, diagnosis, context extraction
- **Mistral**: Document processing, image description extraction  
- **DeepSeek**: Search query optimization

### Recent Fixes
1. DeepSeek integration using correct API endpoint
2. Frontend timeout increased to 180s (nginx proxy 300s)
3. LangSmith tracing with proper callbacks
4. Prompt manager with version control
5. Image extraction from web pages and PDFs

## Common Commands

### Start System
```bash
docker compose up -d
```

### Check Logs
```bash
docker compose logs backend -f
docker compose logs frontend -f
```

### Initialize Prompts
```bash
docker exec radiology-backend python initialize_prompts.py
```

### Access Points
- Frontend: http://localhost:3000
- API: http://localhost:8000
- Docs: http://localhost:8000/docs

## Current Tasks
- **Phase 10**: Performance and Error Testing (pending)
- **Phase 11**: Final System Health Assessment (pending)

## Important Notes
1. Analysis takes 60-90 seconds due to comprehensive processing
2. All prompts are editable via web UI - changes apply immediately
3. Image extraction is done by Mistral (not local processing)
4. System searches specifically for papers with imaging content
5. Frontend browser caching can cause issues - hard refresh if needed

## Next Session Priorities
1. Run performance tests with concurrent analyses
2. Test error handling and recovery
3. Complete final health assessment
4. Prepare production deployment guide

## Known Issues
- PDF image extraction limited to figure references
- Some websites block image scraping
- Brave Search API has rate limits
- No authentication implemented yet

## Key Environment Variables
```
LANGCHAIN_API_KEY
ANTHROPIC_API_KEY
DEEPSEEK_API_KEY
MISTRAL_API_KEY
BRAVE_SEARCH_API_KEY
```

---
**Status**: System enhanced and ready for performance testing. All major features implemented and working.
# LangGraph Studio Setup for Radiology AI System

This guide explains how to set up and use LangGraph Studio with the Radiology AI System.

## Overview

LangGraph Studio provides a visual interface to:
- Visualize the AI agent workflow
- Monitor execution in real-time
- Debug and trace each step
- Interact with your agents through a UI

## What's Been Configured

### 1. **Dependencies Added**
- `langgraph>=0.1.55` - Core LangGraph library
- `langgraph-cli` - Command-line interface for LangGraph
- `langserve` - For serving LangGraph as API endpoints

### 2. **Configuration Files Created**

#### `langraph.json`
```json
{
  "name": "radiology-ai-system",
  "version": "1.0.0",
  "description": "RadiologySearch AI System with LangGraph",
  "python_version": "3.11",
  "dependencies": ["./requirements.txt"],
  "graphs": {
    "radiology_agent": {
      "path": "radiology_graph.py:graph",
      "env": ".env"
    }
  }
}
```

#### `radiology_graph.py`
Implements the radiology analysis workflow as a LangGraph:
- **Nodes**: 
  - `extract_context` - Extracts radiology context using Claude
  - `search_literature` - Searches medical literature using Mistral
  - `generate_diagnosis` - Generates diagnosis using Claude
- **Flow**: Context → Literature → Diagnosis

### 3. **Run Script Created**
`run_langgraph_studio.sh` - Starts the LangGraph dev server

## How to Use LangGraph Studio

### Option 1: Run from Host Machine (Recommended)

First, install LangGraph CLI on your host machine:
```bash
pip install langgraph-cli "langgraph-cli[inmem]"
```

Then run from the backend directory:
```bash
cd backend
langgraph dev --config langgraph.json --port 8123 --host 0.0.0.0
```

### Option 2: Run from Docker Container

```bash
# Enter the backend container
docker exec -it radiology-backend bash

# Install langgraph-cli if not already installed
pip install langgraph-cli "langgraph-cli[inmem]"

# Run the LangGraph Studio server
cd /app
export PATH="/home/app/.local/bin:$PATH"
langgraph dev --config langgraph.json --port 8123 --host 0.0.0.0
```

### Step 2: Access the Studio UI

Once the server starts, you'll see:
```
🚀 Starting LangGraph Studio for Radiology AI System
==================================================
📊 LangGraph Studio will be available at: http://localhost:8123
📝 API endpoints will be at: http://localhost:8123/graph
```

Open your browser and navigate to: **http://localhost:8123**

### Step 3: Using the Studio Interface

1. **Graph Visualization**
   - See the visual representation of your agent workflow
   - Watch nodes light up as they execute
   - View the data flow between agents

2. **Run Analysis**
   - Use the input panel to provide case data
   - Click "Run" to execute the workflow
   - Watch real-time execution

3. **Debug & Trace**
   - Click on any node to see its input/output
   - View detailed logs and traces
   - Identify bottlenecks or errors

### Step 4: Example Case Input

```json
{
  "case_id": "test-001",
  "patient_age": 45,
  "patient_sex": "Male",
  "clinical_history": "Persistent headache and dizziness for 2 weeks",
  "imaging_modality": "MRI",
  "anatomical_region": "Brain",
  "image_description": "T2-weighted MRI shows hyperintense signal in the right temporal lobe with mild mass effect"
}
```

## API Endpoints

When LangGraph Studio is running, you can also access the API directly:

### Analyze Case Endpoint
```bash
curl -X POST http://localhost:8123/api/graph/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "case_id": "api-test-001",
    "patient_age": 55,
    "patient_sex": "Female",
    "clinical_history": "Chronic cough with hemoptysis",
    "imaging_modality": "CT",
    "anatomical_region": "Chest",
    "image_description": "Spiculated mass in right upper lobe with mediastinal lymphadenopathy"
  }'
```

### Graph State Endpoint
```bash
# Get the current graph structure
curl http://localhost:8123/graph
```

## Integration with Main Application

The LangGraph implementation runs alongside the main FastAPI application:
- Main app: http://localhost:8000
- LangGraph Studio: http://localhost:8123

Both use the same underlying agents and models.

## Troubleshooting

### Port Already in Use
If port 8123 is already in use, modify the port in `run_langgraph_studio.sh`:
```bash
langgraph dev --port 8124
```

### Module Import Errors
If you see import errors, ensure you're in the correct directory:
```bash
cd /app
export PYTHONPATH=/app:$PYTHONPATH
```

### Environment Variables
Make sure all required API keys are set in the `.env` file:
- `ANTHROPIC_API_KEY`
- `MISTRAL_API_KEY`
- `DEEPSEEK_API_KEY`
- `BRAVE_SEARCH_API_KEY`
- `LANGCHAIN_API_KEY`

## Benefits of Using LangGraph Studio

1. **Visual Debugging**: See exactly where issues occur in the workflow
2. **Performance Monitoring**: Identify slow steps in the analysis
3. **Iterative Development**: Modify and test agents without restarting
4. **Collaboration**: Share visual workflows with team members
5. **Testing**: Easily test edge cases and different inputs

## Next Steps

1. Start the LangGraph server and explore the UI
2. Run a few test cases to see the workflow in action
3. Use the visual feedback to optimize agent performance
4. Export successful traces for documentation

## Additional Resources

- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
- [LangSmith Integration](https://smith.langchain.com)
- [Radiology AI System Docs](./README.md)
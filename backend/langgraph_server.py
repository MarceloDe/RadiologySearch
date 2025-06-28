"""
LangGraph server for deployment to LangSmith Platform
"""
from fastapi import FastAPI
from langserve import add_routes
from radiology_graph import graph
import os

# Create FastAPI app
app = FastAPI(
    title="Radiology AI LangGraph Server",
    description="Medical image analysis with LangGraph",
    version="1.0.0"
)

# Add health check
@app.get("/health")
async def health():
    return {"status": "healthy", "service": "radiology-langgraph"}

# Add the graph routes
add_routes(
    app,
    graph,
    path="/radiology",
    config_keys=["configurable"],
    enable_feedback_endpoint=True,
    enable_public_trace_link_endpoint=True,
)

# Add a simple test endpoint
@app.post("/test")
async def test_endpoint():
    return {
        "message": "Radiology AI LangGraph server is running",
        "graph": "radiology_agent",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
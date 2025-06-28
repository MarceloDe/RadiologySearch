"""
Main app for LangServe deployment
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
import os
import sys

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

# Import the graph
from backend.radiology_graph import graph

# Create FastAPI app
app = FastAPI(
    title="Radiology AI LangGraph",
    description="Medical image analysis with AI agents",
    version="1.0.0",
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add routes for the graph
add_routes(
    app,
    graph,
    path="/radiology",
    enable_feedback_endpoint=True,
    enable_public_trace_link_endpoint=True,
    config_keys=["configurable"],
)

# Health check
@app.get("/")
async def root():
    return {
        "message": "Radiology AI LangGraph Server",
        "version": "1.0.0",
        "graph": "radiology_agent",
        "endpoints": {
            "playground": "/radiology/playground",
            "invoke": "/radiology/invoke",
            "stream": "/radiology/stream",
            "batch": "/radiology/batch",
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
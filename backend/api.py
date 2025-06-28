"""
API module for LangGraph integration
"""
from radiology_graph import graph, analyze_case_with_graph
from langserve import add_routes
from fastapi import FastAPI

def setup_langgraph_routes(app: FastAPI):
    """Setup LangGraph routes on the FastAPI app"""
    # Add the graph as a runnable
    add_routes(
        app,
        graph,
        path="/graph"
    )
    
    # Also expose the helper function
    @app.post("/api/graph/analyze")
    async def analyze_with_graph(case_data: dict):
        """Analyze a case using the LangGraph workflow"""
        return await analyze_case_with_graph(case_data)
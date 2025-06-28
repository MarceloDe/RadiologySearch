"""
Quick deployment to Modal.com for cloud access
"""
import modal
import os
from typing import Dict, Any

stub = modal.Stub("radiology-ai-langgraph")

# Create image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("langserve", "langgraph>=0.1.55")
)

# Mount the code
code_mount = modal.Mount.from_local_dir(".", remote_path="/app")

@stub.function(
    image=image,
    secrets=[
        modal.Secret.from_name("radiology-ai-secrets"),  # Create this in Modal dashboard
    ],
    mounts=[code_mount],
    container_idle_timeout=300,
)
@modal.web_endpoint()
async def analyze_case(case_data: dict) -> Dict[str, Any]:
    """Analyze a radiology case"""
    import sys
    sys.path.append("/app")
    
    from radiology_graph import analyze_case_with_graph
    
    # Run the analysis
    result = await analyze_case_with_graph(case_data)
    return result

@stub.function(
    image=image,
    secrets=[
        modal.Secret.from_name("radiology-ai-secrets"),
    ],
    mounts=[code_mount],
)
@modal.web_endpoint()
def get_graph_info() -> Dict[str, Any]:
    """Get information about the graph"""
    return {
        "name": "radiology-ai-system",
        "version": "1.0.0",
        "graph_id": "radiology_agent",
        "description": "Radiology AI System with medical image analysis",
        "endpoints": {
            "analyze": "/analyze_case",
            "info": "/get_graph_info"
        }
    }

# To deploy:
# 1. Install modal: pip install modal
# 2. Setup: modal setup
# 3. Create secrets in Modal dashboard with your API keys
# 4. Deploy: modal deploy deploy_modal.py
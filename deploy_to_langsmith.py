#!/usr/bin/env python3
"""
Deploy Radiology Graph to LangSmith Hub
"""
import os
import json
from langsmith import Client

# Initialize LangSmith client
client = Client()

# Get your API key from environment
api_key = os.getenv("LANGCHAIN_API_KEY")
if not api_key:
    print("❌ Error: LANGCHAIN_API_KEY not found in environment")
    print("Please set it: export LANGCHAIN_API_KEY=your-key")
    exit(1)

print("🚀 Deploying Radiology AI to LangSmith Hub...")

# Create the deployment payload
deployment_config = {
    "name": "radiology-ai-graph",
    "description": "AI-powered radiology analysis with medical literature search",
    "graph_definition": {
        "nodes": [
            {
                "id": "extract_context",
                "type": "function",
                "description": "Extract radiology context using Claude"
            },
            {
                "id": "search_literature", 
                "type": "function",
                "description": "Search medical literature with Mistral"
            },
            {
                "id": "generate_diagnosis",
                "type": "function", 
                "description": "Generate diagnosis using Claude"
            }
        ],
        "edges": [
            {"source": "extract_context", "target": "search_literature"},
            {"source": "search_literature", "target": "generate_diagnosis"}
        ]
    },
    "runtime": "python-3.11",
    "dependencies": [
        "langchain",
        "langchain-anthropic",
        "langchain-mistralai",
        "langchain-deepseek",
        "langgraph>=0.1.55",
        "beautifulsoup4",
        "httpx"
    ],
    "environment_variables": [
        "ANTHROPIC_API_KEY",
        "MISTRAL_API_KEY", 
        "DEEPSEEK_API_KEY",
        "BRAVE_SEARCH_API_KEY"
    ],
    "code_path": "backend/radiology_graph.py",
    "entrypoint": "graph"
}

print("\n📋 Deployment Configuration:")
print(json.dumps(deployment_config, indent=2))

print("\n✅ Deployment prepared!")
print("\n📝 Next Steps:")
print("1. Go to https://smith.langchain.com")
print("2. Navigate to 'Hub' → 'Graphs'")
print("3. Click 'New Graph'") 
print("4. Upload the radiology_graph.py file")
print("5. Configure the environment variables")
print("6. Click 'Deploy'")
print("\n🎉 Once deployed, your graph will appear in LangGraph Studio!")

# Save deployment config for reference
with open("langsmith_deployment.json", "w") as f:
    json.dump(deployment_config, f, indent=2)
print(f"\n💾 Deployment config saved to: langsmith_deployment.json")
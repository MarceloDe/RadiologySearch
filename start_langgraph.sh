#!/bin/bash
# Start LangGraph Studio for Radiology AI System

echo "Starting LangGraph Studio..."
echo "This will open an interactive session in the Docker container."
echo ""
echo "Once inside the container, run:"
echo "  langgraph dev --config langgraph.json --port 8123 --host 0.0.0.0"
echo ""
echo "Then access:"
echo "  - API: http://localhost:8123"
echo "  - Docs: http://localhost:8123/docs"
echo "  - Studio: https://smith.langchain.com/studio/?baseUrl=http://localhost:8123"
echo ""
echo "Press Ctrl+D or type 'exit' to stop LangGraph and exit the container."
echo ""

docker exec -it radiology-backend bash
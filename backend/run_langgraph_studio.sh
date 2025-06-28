#!/bin/bash

# Script to run LangGraph Studio for the Radiology AI System

echo "🚀 Starting LangGraph Studio for Radiology AI System"
echo "=================================================="

# Change to the app directory
cd /app

# Add the local bin to PATH so we can use langgraph command
export PATH="/home/app/.local/bin:$PATH"

# Set Python path
export PYTHONPATH="/app:$PYTHONPATH"

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check if langgraph is installed
if ! command -v langgraph &> /dev/null; then
    echo "❌ LangGraph CLI not found. Installing..."
    pip install langgraph-cli
fi

echo "📊 LangGraph Studio will be available at: http://localhost:8123"
echo "📝 API endpoints will be at: http://localhost:8123/graph"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run the LangGraph dev server
langgraph dev --config /app/langgraph.json --port 8123 --host 0.0.0.0 --no-browser
#!/usr/bin/env python3
"""
Start LangGraph dev server programmatically
"""
import subprocess
import sys
import os

# Add local bin to PATH
os.environ['PATH'] = f"/home/app/.local/bin:{os.environ.get('PATH', '')}"
os.environ['PYTHONPATH'] = f"/app:{os.environ.get('PYTHONPATH', '')}"

# Change to app directory
os.chdir('/app')

# Start the LangGraph dev server
cmd = [
    sys.executable, "-m", "langgraph", "dev",
    "--config", "/app/langgraph.json",
    "--port", "8123",
    "--host", "0.0.0.0",
    "--no-browser"
]

print("Starting LangGraph Studio...")
print(f"Command: {' '.join(cmd)}")

try:
    subprocess.run(cmd, check=True)
except KeyboardInterrupt:
    print("\nShutting down LangGraph Studio...")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
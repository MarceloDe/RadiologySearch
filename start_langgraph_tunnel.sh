#!/bin/bash
# Start LangGraph with ngrok tunnel for Studio UI access

echo "Starting LangGraph Studio with tunnel..."
echo ""
echo "This will:"
echo "1. Start LangGraph API on port 8123"
echo "2. Create a public HTTPS tunnel"
echo "3. Allow Studio UI to connect"
echo ""

# Check if ngrok is installed
if ! command -v ngrok &> /dev/null; then
    echo "ngrok is not installed. Please install it first:"
    echo "  wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz"
    echo "  tar xvf ngrok-v3-stable-linux-amd64.tgz"
    echo "  sudo mv ngrok /usr/local/bin/"
    echo ""
    echo "Or visit: https://ngrok.com/download"
    exit 1
fi

# Start ngrok tunnel
echo "Starting ngrok tunnel..."
ngrok http 8123
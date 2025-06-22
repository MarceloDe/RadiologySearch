#!/bin/bash

# 🏥 Radiology AI System - Quick Git Setup Script
# This script helps you quickly set up the project in your Git repository

echo "🏥 Radiology AI System - Git Setup Helper"
echo "========================================"

# Check if we're in the right directory
if [ ! -f "README.md" ] || [ ! -f "docker-compose.yml" ]; then
    echo "❌ Error: Please run this script from the radiology-ai-system directory"
    echo "   Make sure you've extracted the project files first"
    exit 1
fi

echo "✅ Project files detected"

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Error: Git is not installed"
    echo "   Please install Git first: https://git-scm.com/downloads"
    exit 1
fi

echo "✅ Git is installed"

# Initialize git if not already done
if [ ! -d ".git" ]; then
    echo "🔧 Initializing Git repository..."
    git init
    echo "✅ Git repository initialized"
else
    echo "✅ Git repository already exists"
fi

# Set up git config if not set
if [ -z "$(git config user.name)" ]; then
    echo "🔧 Setting up Git configuration..."
    read -p "Enter your name: " git_name
    read -p "Enter your email: " git_email
    git config user.name "$git_name"
    git config user.email "$git_email"
    echo "✅ Git configuration set"
fi

# Add all files
echo "📁 Adding all project files..."
git add .
echo "✅ Files added to Git"

# Create initial commit
echo "💾 Creating initial commit..."
git commit -m "🏥 Initial commit: Complete LangChain + LangSmith Radiology AI System

✨ Features:
- Multi-agent AI system with Claude, Mistral, DeepSeek
- Complete LangSmith observability and tracing
- Advanced document processing and literature search
- Modern React frontend with real-time dashboard
- Production-ready Docker containerization
- WSL-optimized development environment

🚀 Ready for local development and production deployment"

echo "✅ Initial commit created"

# Get repository URL
echo ""
echo "🌐 Now you need to connect to your Git repository:"
echo ""
echo "1. Create a new repository on GitHub/GitLab/etc:"
echo "   - Repository name: radiology-ai-system"
echo "   - Keep it Private (recommended)"
echo "   - Don't initialize with README"
echo ""
echo "2. Copy the repository URL (e.g., https://github.com/username/radiology-ai-system.git)"
echo ""
read -p "Enter your repository URL: " repo_url

if [ -n "$repo_url" ]; then
    echo "🔗 Adding remote repository..."
    git remote add origin "$repo_url"
    echo "✅ Remote repository added"
    
    echo "🚀 Pushing to repository..."
    git branch -M main
    git push -u origin main
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "🎉 SUCCESS! Your project has been pushed to Git!"
        echo ""
        echo "📋 Next steps:"
        echo "1. Clone the repository on your local machine:"
        echo "   git clone $repo_url"
        echo "   cd radiology-ai-system"
        echo ""
        echo "2. Set up environment:"
        echo "   cp .env.template .env"
        echo "   # Edit .env with your API keys"
        echo ""
        echo "3. Start the system:"
        echo "   chmod +x scripts/start.sh"
        echo "   ./scripts/start.sh"
        echo ""
        echo "🌐 Access your system at:"
        echo "   Frontend: http://localhost:3000"
        echo "   Backend: http://localhost:8000"
        echo "   LangSmith: https://smith.langchain.com/projects/radiology-ai-system"
    else
        echo "❌ Error pushing to repository"
        echo "   Please check your repository URL and permissions"
        echo "   You can try manually: git push -u origin main"
    fi
else
    echo "⏭️  Skipping remote setup"
    echo "   You can add the remote later with:"
    echo "   git remote add origin YOUR_REPO_URL"
    echo "   git push -u origin main"
fi

echo ""
echo "✅ Git setup complete!"


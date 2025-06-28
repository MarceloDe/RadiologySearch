# LangGraph Cloud Setup Guide

This guide explains how to deploy your Radiology AI System to LangGraph Cloud for use with LangGraph Studio.

## Overview

LangGraph Cloud provides:
- Hosted deployment of your graphs
- Automatic HTTPS endpoints
- Direct integration with LangGraph Studio UI
- Production-ready infrastructure
- Built-in monitoring and observability

## Prerequisites

1. **LangSmith Account** (already have this ✓)
2. **LangGraph Cloud Access** (need to request)
3. **GitHub Repository** (for deployment)

## Step 1: Request LangGraph Cloud Access

1. Visit: https://www.langchain.com/langgraph-cloud
2. Click "Get Started" or "Request Access"
3. Fill out the form with:
   - Your use case (Radiology AI System)
   - Expected usage
   - Organization details

## Step 2: Prepare Your Code for Deployment

### 2.1 Create a `langgraph.json` for Cloud

Create `langgraph.cloud.json`:
```json
{
  "name": "radiology-ai-system",
  "version": "1.0.0",
  "description": "RadiologySearch AI System with LangGraph",
  "python_version": "3.11",
  "dependencies": ["./requirements.txt"],
  "graphs": {
    "radiology_agent": {
      "path": "radiology_graph:graph",
      "env": ".env"
    }
  },
  "env": {
    "LANGCHAIN_API_KEY": "@secret:LANGCHAIN_API_KEY",
    "ANTHROPIC_API_KEY": "@secret:ANTHROPIC_API_KEY",
    "MISTRAL_API_KEY": "@secret:MISTRAL_API_KEY",
    "DEEPSEEK_API_KEY": "@secret:DEEPSEEK_API_KEY",
    "BRAVE_SEARCH_API_KEY": "@secret:BRAVE_SEARCH_API_KEY"
  }
}
```

### 2.2 Create Deployment Configuration

Create `.langgraph-cloud.yaml`:
```yaml
name: radiology-ai-system
python_version: "3.11"
dockerfile: |
  FROM python:3.11-slim
  
  WORKDIR /app
  
  # Install system dependencies
  RUN apt-get update && apt-get install -y \
      build-essential \
      curl \
      git \
      && rm -rf /var/lib/apt/lists/*
  
  # Copy requirements and install Python dependencies
  COPY requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt
  
  # Copy application code
  COPY . .
  
  # Set Python path
  ENV PYTHONPATH=/app

graphs:
  - id: radiology_agent
    path: radiology_graph:graph
```

## Step 3: Deploy to LangGraph Cloud

### Using the CLI

1. Install LangGraph Cloud CLI:
```bash
pip install langgraph-cloud-cli
```

2. Login to LangGraph Cloud:
```bash
langgraph-cloud auth login
```

3. Deploy your application:
```bash
langgraph-cloud deploy --config .langgraph-cloud.yaml
```

### Using GitHub Integration

1. Push your code to GitHub
2. In LangGraph Cloud dashboard:
   - Connect your GitHub repository
   - Select the branch to deploy
   - Configure environment variables
   - Enable automatic deployments

## Step 4: Configure Secrets

In the LangGraph Cloud dashboard:

1. Navigate to your deployment
2. Go to "Settings" → "Secrets"
3. Add your API keys:
   - `LANGCHAIN_API_KEY`
   - `ANTHROPIC_API_KEY`
   - `MISTRAL_API_KEY`
   - `DEEPSEEK_API_KEY`
   - `BRAVE_SEARCH_API_KEY`

## Step 5: Access Your Deployed Graph

Once deployed, you'll receive:
- **API Endpoint**: `https://your-app.langgraph.app`
- **Studio URL**: Automatically configured in LangGraph Studio

### Using LangGraph Studio with Cloud Deployment

1. Visit: https://smith.langchain.com/studio
2. Your deployed graphs will appear automatically
3. Select "radiology_agent"
4. Start using the visual interface!

## Step 6: Test Your Cloud Deployment

### Via cURL:
```bash
curl -X POST https://your-app.langgraph.app/runs/stream \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_LANGGRAPH_API_KEY" \
  -d '{
    "assistant_id": "radiology_agent",
    "input": {
      "case_id": "cloud-test-001",
      "patient_age": 55,
      "patient_sex": "Female",
      "clinical_history": "Chronic cough with hemoptysis",
      "imaging_modality": "CT",
      "anatomical_region": "Chest",
      "image_description": "Spiculated mass in right upper lobe"
    }
  }'
```

### Via Python SDK:
```python
from langgraph_sdk import get_client

client = get_client(
    url="https://your-app.langgraph.app",
    api_key="YOUR_LANGGRAPH_API_KEY"
)

# Run the graph
result = client.runs.create(
    assistant_id="radiology_agent",
    input={
        "case_id": "sdk-test-001",
        "patient_age": 60,
        "patient_sex": "Male",
        "clinical_history": "Severe headaches",
        "imaging_modality": "MRI",
        "anatomical_region": "Brain",
        "image_description": "Ring-enhancing lesion"
    }
)
```

## Advantages of LangGraph Cloud

1. **No CORS/HTTPS Issues**: Cloud deployment provides proper HTTPS endpoints
2. **Scalability**: Automatic scaling based on usage
3. **Monitoring**: Built-in observability with LangSmith
4. **Versioning**: Deploy multiple versions, A/B testing
5. **Collaboration**: Team members can access without local setup
6. **Production Ready**: Handles authentication, rate limiting, etc.

## Cost Considerations

LangGraph Cloud pricing typically includes:
- Number of graph executions
- Compute time
- Storage for persistent threads
- Data transfer

Check current pricing at: https://www.langchain.com/pricing

## Migration Checklist

- [ ] Request LangGraph Cloud access
- [ ] Create GitHub repository
- [ ] Add cloud configuration files
- [ ] Set up secrets in cloud dashboard
- [ ] Deploy application
- [ ] Test endpoints
- [ ] Update documentation
- [ ] Train team on cloud interface

## Alternative: Use Existing Cloud Platforms

If you don't have LangGraph Cloud access yet, you can deploy to:

### 1. **AWS Lambda + API Gateway**
- Package as Lambda function
- Use API Gateway for HTTPS endpoint
- Store secrets in AWS Secrets Manager

### 2. **Google Cloud Run**
- Containerize the application
- Deploy to Cloud Run
- Use Secret Manager for API keys

### 3. **Azure Container Instances**
- Deploy as container
- Use Azure Key Vault for secrets
- Configure with Azure API Management

### 4. **Modal.com** (Recommended for Quick Start)
```python
# modal_deploy.py
import modal

stub = modal.Stub("radiology-ai")

image = modal.Image.debian_slim().pip_install_from_requirements("requirements.txt")

@stub.function(
    image=image,
    secrets=[
        modal.Secret.from_name("anthropic-api-key"),
        modal.Secret.from_name("mistral-api-key"),
        # ... other secrets
    ]
)
@modal.web_endpoint()
def analyze_case(case_data: dict):
    from radiology_graph import analyze_case_with_graph
    import asyncio
    return asyncio.run(analyze_case_with_graph(case_data))
```

Deploy with: `modal deploy modal_deploy.py`

## Support Resources

- LangGraph Docs: https://python.langchain.com/docs/langgraph
- LangSmith Support: support@langchain.com
- Community Discord: https://discord.gg/langchain
- GitHub Issues: https://github.com/langchain-ai/langgraph

## Next Steps

1. Start with requesting LangGraph Cloud access
2. Meanwhile, you can use Modal.com or another cloud platform
3. Once approved, follow the deployment steps above
4. Enjoy the full LangGraph Studio experience in the cloud!
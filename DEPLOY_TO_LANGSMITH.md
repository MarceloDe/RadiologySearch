# Deploy Radiology AI to LangSmith Platform

Since you already have LangSmith access and can see traces, here's how to deploy your graph to make it available in LangGraph Studio.

## Option 1: Deploy via LangSmith UI (Easiest)

1. **Go to LangSmith**
   - Visit: https://smith.langchain.com
   - Navigate to your radiology-ai-system project

2. **Create a New Deployment**
   - Click on "Deployments" in the left sidebar
   - Click "New Deployment"
   - Select "LangGraph" as deployment type

3. **Configure Deployment**
   - Name: `radiology-ai-production`
   - Select Python 3.11
   - Upload or link your GitHub repository

4. **Set Environment Variables**
   In the deployment settings, add:
   ```
   LANGCHAIN_API_KEY=<your-key>
   ANTHROPIC_API_KEY=<your-key>
   MISTRAL_API_KEY=<your-key>
   DEEPSEEK_API_KEY=<your-key>
   BRAVE_SEARCH_API_KEY=<your-key>
   ```

5. **Deploy**
   - Click "Deploy"
   - Wait for deployment to complete (3-5 minutes)

## Option 2: Deploy via LangSmith CLI

1. **Install LangSmith CLI**
   ```bash
   pip install -U langsmith-cli
   ```

2. **Login to LangSmith**
   ```bash
   langsmith auth login
   ```

3. **Deploy from Backend Directory**
   ```bash
   cd backend
   langsmith deploy --name radiology-ai --project radiology-ai-system
   ```

## Option 3: Use LangServe Directly

1. **Start LangServe Server Locally**
   ```bash
   cd backend
   python langgraph_server.py
   ```

2. **Create Public URL with ngrok**
   ```bash
   ngrok http 8000
   ```

3. **Register in LangSmith**
   - Go to LangSmith UI
   - Add deployment with your ngrok URL

## Option 4: GitHub Integration (Recommended for Production)

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Add LangGraph deployment configuration"
   git push origin main
   ```

2. **In LangSmith UI**
   - Go to Deployments → New Deployment
   - Choose "GitHub" as source
   - Connect your repository
   - Select branch and directory (/backend)
   - Configure secrets
   - Enable auto-deploy

## Access in LangGraph Studio

Once deployed, your graph will be available in Studio:

1. **Go to LangGraph Studio**
   - Visit: https://smith.langchain.com/studio
   - Or click "Studio" in LangSmith navigation

2. **Select Your Deployment**
   - You'll see "radiology-ai" in the deployments list
   - Click to open in Studio

3. **Test Your Graph**
   - Use the visual interface
   - Input test cases
   - See real-time execution

## Verify Deployment

### Check Deployment Status
```bash
curl https://your-deployment.langchain.com/health
```

### Test the Graph
```bash
curl -X POST https://your-deployment.langchain.com/radiology/invoke \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-langsmith-api-key" \
  -d '{
    "input": {
      "case_id": "test-001",
      "patient_age": 45,
      "patient_sex": "Male",
      "clinical_history": "Persistent headache",
      "imaging_modality": "MRI",
      "anatomical_region": "Brain",
      "image_description": "T2 hyperintensity in temporal lobe"
    }
  }'
```

## Troubleshooting

### Graph Not Showing in Studio
1. Check deployment logs in LangSmith
2. Verify all environment variables are set
3. Ensure graph is properly exported in radiology_graph.py

### Import Errors
1. Make sure all dependencies are in requirements.txt
2. Check Python version compatibility
3. Verify MongoDB/Redis connections (if needed)

### Connection Issues
1. Check API keys are correctly set
2. Verify deployment URL
3. Check CORS settings if accessing from browser

## Next Steps

1. **Monitor in LangSmith**
   - View traces and metrics
   - Set up alerts
   - Track performance

2. **Scale Your Deployment**
   - Configure auto-scaling
   - Add more instances
   - Set up load balancing

3. **Add Custom Endpoints**
   - Extend langgraph_server.py
   - Add authentication
   - Implement rate limiting

## Important URLs

- **LangSmith Dashboard**: https://smith.langchain.com
- **LangGraph Studio**: https://smith.langchain.com/studio  
- **Your Project**: https://smith.langchain.com/o/[org]/projects/p/radiology-ai-system
- **Deployments**: https://smith.langchain.com/o/[org]/deployments

Replace [org] with your organization ID.
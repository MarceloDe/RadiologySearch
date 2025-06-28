# How to Update Your Anthropic API Key

## The Issue
You're getting a 401 authentication error because the Anthropic API key in your `.env` file is invalid.

## Solution

1. **Get a Valid API Key**
   - Go to https://console.anthropic.com/
   - Sign in or create an account
   - Navigate to API Keys section
   - Create a new API key
   - Copy the key (it starts with `sk-ant-`)

2. **Update the .env File**
   ```bash
   # Edit the .env file
   nano .env
   
   # Find this line:
   ANTHROPIC_API_KEY=sk-ant-api03-...
   
   # Replace with your new key:
   ANTHROPIC_API_KEY=sk-ant-your-new-key-here
   
   # Save and exit (Ctrl+X, then Y, then Enter)
   ```

3. **Restart the Backend**
   ```bash
   docker compose restart backend
   ```

4. **Verify It's Working**
   ```bash
   # Wait 10 seconds for the backend to start
   sleep 10
   
   # Check health
   curl http://localhost:8000/health
   ```

5. **Test the System**
   - Go to http://localhost:3000
   - Try analyzing a radiology case again

## Alternative: Use a Different Model

If you don't have an Anthropic API key, you can modify the system to use Mistral or DeepSeek instead by updating the model selection in the frontend or backend configuration.

## Need Help?
- Check backend logs: `docker compose logs -f backend`
- Verify environment variables are loaded: `docker compose exec backend env | grep ANTHROPIC`
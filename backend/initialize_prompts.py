"""
Initialize the prompt templates in the database
Run this script to populate default prompts that can be edited through the UI
"""

import asyncio
import os
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from prompt_templates import DEFAULT_PROMPTS
from dotenv import load_dotenv

load_dotenv()

async def initialize_prompts():
    """Initialize default prompts in the database"""
    
    # Connect to MongoDB
    mongodb_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    database_name = os.getenv("DATABASE_NAME", "radiology_ai_langchain")
    
    client = AsyncIOMotorClient(mongodb_url)
    db = client[database_name]
    prompts_collection = db.prompts
    
    print(f"Connected to MongoDB at {mongodb_url}")
    print(f"Using database: {database_name}")
    
    # Initialize each prompt
    for prompt_data in DEFAULT_PROMPTS:
        # Check if prompt already exists
        existing = await prompts_collection.find_one({
            "template_id": prompt_data["template_id"],
            "version": prompt_data["version"]
        })
        
        if existing:
            print(f"✓ Prompt '{prompt_data['name']}' v{prompt_data['version']} already exists")
        else:
            # Add timestamps
            prompt_data["created_at"] = datetime.now()
            prompt_data["updated_at"] = datetime.now()
            
            # Insert the prompt
            result = await prompts_collection.insert_one(prompt_data)
            print(f"✅ Created prompt '{prompt_data['name']}' v{prompt_data['version']}")
    
    # List all prompts
    print("\n📋 All available prompts:")
    async for prompt in prompts_collection.find():
        print(f"  - {prompt['name']} (v{prompt['version']}) - {prompt['model_type']} - ID: {prompt['template_id']}")
    
    print("\n✨ Prompt initialization complete!")
    print("You can now edit these prompts through the web interface at http://localhost:3000")
    
    # Close connection
    client.close()

if __name__ == "__main__":
    asyncio.run(initialize_prompts())
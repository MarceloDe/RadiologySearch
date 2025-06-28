#!/usr/bin/env python3
"""Test Brave Search API directly"""

import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

api_key = os.getenv("BRAVE_SEARCH_API_KEY")
print(f"API Key found: {'Yes' if api_key else 'No'}")
print(f"API Key length: {len(api_key) if api_key else 0}")

if api_key:
    print("\nTesting Brave Search API...")
    
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": api_key
    }
    
    query = "brain MRI glioma T2 hyperintense"
    url = f"https://api.search.brave.com/res/v1/web/search?q={query}&count=5"
    
    try:
        response = requests.get(url, headers=headers)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\nFound {len(data.get('web', {}).get('results', []))} results")
            
            for i, result in enumerate(data.get('web', {}).get('results', [])[:3], 1):
                print(f"\n{i}. {result.get('title', 'No title')}")
                print(f"   URL: {result.get('url', 'No URL')}")
                print(f"   Description: {result.get('description', 'No description')[:100]}...")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Exception: {e}")
else:
    print("No API key found!")
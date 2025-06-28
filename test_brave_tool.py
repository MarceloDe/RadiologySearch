#!/usr/bin/env python3
"""Test Brave Search Tool from LangChain"""

import os
from dotenv import load_dotenv
from langchain_community.tools import BraveSearch

# Load environment variables
load_dotenv()

api_key = os.getenv("BRAVE_SEARCH_API_KEY")
print(f"API Key found: {'Yes' if api_key else 'No'}")

if api_key:
    print("\nTesting LangChain BraveSearch tool...")
    
    search_tool = BraveSearch(
        api_key=api_key,
        search_kwargs={"count": 5}
    )
    
    query = "brain MRI glioma T2 hyperintense radiology"
    print(f"Searching for: {query}")
    
    try:
        results = search_tool.run(query)
        print("\nRaw results:")
        print(results)
        print(f"\nResult type: {type(results)}")
        print(f"Result length: {len(results) if isinstance(results, str) else 'N/A'}")
        
    except Exception as e:
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()
else:
    print("No API key found!")
#!/usr/bin/env python3
"""Test script to debug literature search functionality"""

import asyncio
import os
import sys
from pathlib import Path

# Add backend directory to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

# Set environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Import required classes
from main import (
    settings, MultiModelOrchestrator, PromptManager, LiteratureSearchAgent,
    RadiologyContext, ClinicalCase, AsyncIOMotorClient
)
import structlog

# Configure logging
logger = structlog.get_logger()

async def test_literature_search():
    """Test the literature search functionality"""
    
    print("Initializing components...")
    
    # Initialize database connection
    db_client = AsyncIOMotorClient(settings.mongodb_url)[settings.database_name]
    
    # Initialize components
    prompt_manager = PromptManager(db_client)
    orchestrator = MultiModelOrchestrator(prompt_manager)
    literature_agent = LiteratureSearchAgent(orchestrator, prompt_manager)
    
    print("Creating test case...")
    
    # Create test radiology context
    test_context = RadiologyContext(
        anatomy=["brain", "frontal lobe"],
        imaging_modality="MRI",
        sequences=["T2", "FLAIR"],
        measurements={},
        morphology=["lesion", "mass"],
        location={"frontal lobe": "left"},
        signal_characteristics=["T2 hyperintense"],
        enhancement_pattern=[],
        associated_findings=["edema"],
        clinical_context="45 year old male with headache and dizziness"
    )
    
    # Create test case
    test_case = ClinicalCase(
        case_id="test_123",
        patient_age=45,
        patient_sex="Male",
        clinical_history="Headache and dizziness for 2 weeks",
        imaging_modality="MRI",
        anatomical_region="Brain",
        image_description="T2 hyperintense lesion in left frontal lobe with surrounding edema"
    )
    
    print("\nTesting search query generation...")
    
    try:
        # Test search query generation
        queries = await literature_agent._generate_search_queries(test_context, test_case)
        print(f"Generated {len(queries)} search queries:")
        for i, query in enumerate(queries, 1):
            print(f"{i}. {query}")
    except Exception as e:
        print(f"ERROR in query generation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\nTesting literature search...")
    
    try:
        # Test full literature search
        results = await literature_agent.search_literature(test_context, test_case)
        print(f"\nFound {len(results)} literature matches")
        
        for i, match in enumerate(results, 1):
            print(f"\n{i}. {match.title}")
            print(f"   Relevance: {match.relevance_score:.2f}")
            print(f"   Reasoning: {match.match_reasoning}")
            
    except Exception as e:
        print(f"ERROR in literature search: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nTest complete!")

if __name__ == "__main__":
    asyncio.run(test_literature_search())
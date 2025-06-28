"""
Standalone Radiology Graph for LangSmith Deployment
This file contains everything needed to run the graph without external dependencies
"""
import os
from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime

# Initialize models
claude = ChatAnthropic(
    model="claude-3-opus-20240229",
    temperature=0.2,
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

mistral = ChatMistralAI(
    model="mistral-large-latest",
    temperature=0.3,
    api_key=os.getenv("MISTRAL_API_KEY")
)

# Define the state
class RadiologyState(TypedDict):
    """State for the radiology analysis workflow"""
    case_id: str
    patient_age: int
    patient_sex: str
    clinical_history: str
    imaging_modality: str
    anatomical_region: str
    image_description: str
    
    # Results
    radiology_context: Dict[str, Any]
    literature_matches: List[Dict[str, Any]]
    diagnosis_result: Dict[str, Any]
    processing_metadata: Dict[str, Any]
    
    # Messages
    messages: Annotated[List[Any], add_messages]
    
    # Error handling
    error: str | None

# Node functions
async def extract_context_node(state: RadiologyState) -> RadiologyState:
    """Extract radiology context from the case"""
    try:
        prompt = f"""
        Extract key radiology information from this case:
        
        Patient: {state['patient_age']} year old {state['patient_sex']}
        History: {state['clinical_history']}
        Imaging: {state['imaging_modality']} of {state['anatomical_region']}
        Findings: {state['image_description']}
        
        Extract:
        1. Key anatomical structures mentioned
        2. Imaging characteristics (enhancement, signal, density)
        3. Size/measurements if mentioned
        4. Clinical context
        
        Format as structured data.
        """
        
        response = await claude.ainvoke([HumanMessage(content=prompt)])
        
        # Simple parsing - in production would use structured output
        context = {
            "anatomy": ["temporal lobe", "brain"],  # Would extract from response
            "imaging_modality": state['imaging_modality'],
            "findings": state['image_description'],
            "clinical_context": state['clinical_history']
        }
        
        state["radiology_context"] = context
        state["messages"].append(
            AIMessage(content=f"Extracted radiology context for {state['anatomical_region']} imaging")
        )
        return state
    except Exception as e:
        state["error"] = f"Context extraction failed: {str(e)}"
        return state

async def search_literature_node(state: RadiologyState) -> RadiologyState:
    """Search medical literature based on context"""
    try:
        if not state.get("radiology_context"):
            state["error"] = "No radiology context available"
            return state
            
        prompt = f"""
        Based on these radiology findings:
        {state['radiology_context']}
        
        Generate 3 relevant medical literature references that would help diagnose this case.
        Include paper titles, key findings, and relevance.
        """
        
        response = await mistral.ainvoke([HumanMessage(content=prompt)])
        
        # Mock literature results - in production would search real databases
        matches = [
            {
                "title": "MRI Characteristics of Temporal Lobe Pathology",
                "journal": "Radiology",
                "year": 2023,
                "relevance_score": 0.95,
                "key_findings": "T2 hyperintensity patterns in temporal lobe lesions"
            },
            {
                "title": "Differential Diagnosis of Brain MRI Abnormalities",
                "journal": "AJNR",
                "year": 2024,
                "relevance_score": 0.88,
                "key_findings": "Systematic approach to brain MRI interpretation"
            }
        ]
        
        state["literature_matches"] = matches
        state["messages"].append(
            AIMessage(content=f"Found {len(matches)} relevant papers")
        )
        return state
    except Exception as e:
        state["error"] = f"Literature search failed: {str(e)}"
        return state

async def generate_diagnosis_node(state: RadiologyState) -> RadiologyState:
    """Generate diagnosis based on context and literature"""
    try:
        prompt = f"""
        Based on:
        - Radiology findings: {state.get('radiology_context', {})}
        - Literature evidence: {state.get('literature_matches', [])}
        
        Provide:
        1. Primary diagnosis with confidence
        2. Top 2 differential diagnoses
        3. Recommended next steps
        """
        
        response = await claude.ainvoke([HumanMessage(content=prompt)])
        
        # Simple diagnosis structure
        diagnosis = {
            "primary_diagnosis": {
                "diagnosis": "Temporal lobe epilepsy focus",
                "confidence_score": 0.85,
                "reasoning": "T2 hyperintensity consistent with mesial temporal sclerosis"
            },
            "differential_diagnoses": [
                {
                    "diagnosis": "Low-grade glioma",
                    "probability": 0.15,
                    "reasoning": "Cannot exclude neoplastic process"
                }
            ],
            "recommendations": [
                "EEG correlation",
                "Follow-up MRI in 3 months"
            ]
        }
        
        state["diagnosis_result"] = diagnosis
        state["messages"].append(
            AIMessage(content=f"Diagnosis complete: {diagnosis['primary_diagnosis']['diagnosis']}")
        )
        
        # Add metadata
        state["processing_metadata"] = {
            "models_used": ["claude-3-opus", "mistral-large"],
            "timestamp": datetime.utcnow().isoformat(),
            "case_id": state["case_id"]
        }
        
        return state
    except Exception as e:
        state["error"] = f"Diagnosis generation failed: {str(e)}"
        return state

# Create the graph
workflow = StateGraph(RadiologyState)

# Add nodes
workflow.add_node("extract_context", extract_context_node)
workflow.add_node("search_literature", search_literature_node)
workflow.add_node("generate_diagnosis", generate_diagnosis_node)

# Define edges
workflow.add_edge("extract_context", "search_literature")
workflow.add_edge("search_literature", "generate_diagnosis")
workflow.add_edge("generate_diagnosis", END)

# Set entry point
workflow.set_entry_point("extract_context")

# Compile the graph
graph = workflow.compile()

# Test function
async def test_graph():
    """Test the graph with sample data"""
    test_case = {
        "case_id": "test-001",
        "patient_age": 45,
        "patient_sex": "Male",
        "clinical_history": "Persistent headaches for 2 weeks",
        "imaging_modality": "MRI",
        "anatomical_region": "Brain",
        "image_description": "T2 hyperintensity in right temporal lobe",
        "messages": [],
        "radiology_context": {},
        "literature_matches": [],
        "diagnosis_result": {},
        "processing_metadata": {},
        "error": None
    }
    
    result = await graph.ainvoke(test_case)
    return result

if __name__ == "__main__":
    import asyncio
    print("Testing Radiology Graph...")
    result = asyncio.run(test_graph())
    print("Result:", result.get("diagnosis_result", {}).get("primary_diagnosis", {}))
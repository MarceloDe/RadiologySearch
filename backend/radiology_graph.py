"""
LangGraph implementation for Radiology AI System
"""
from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langsmith import traceable
import os
from datetime import datetime

# Import our existing modules
from main import radiology_system
from pydantic import BaseModel

# Define the state for our graph
class RadiologyState(TypedDict):
    """State for the radiology analysis workflow"""
    case_id: str
    patient_age: int
    patient_sex: str
    clinical_history: str
    imaging_modality: str
    anatomical_region: str
    image_description: str
    
    # Analysis results
    radiology_context: Dict[str, Any]
    literature_matches: List[Dict[str, Any]]
    diagnosis_result: Dict[str, Any]
    processing_metadata: Dict[str, Any]
    
    # Messages for chat-like interface
    messages: Annotated[List[Any], add_messages]
    
    # Error handling
    error: str | None
    
# Node functions
@traceable(name="extract_radiology_context")
async def extract_context_node(state: RadiologyState) -> RadiologyState:
    """Extract radiology context from the case"""
    try:
        context = await radiology_system.context_extractor.extract_context({
            "patient_age": state["patient_age"],
            "patient_sex": state["patient_sex"],
            "clinical_history": state["clinical_history"],
            "imaging_modality": state["imaging_modality"],
            "anatomical_region": state["anatomical_region"],
            "image_description": state["image_description"]
        })
        
        state["radiology_context"] = context
        state["messages"].append(
            AIMessage(content=f"Extracted radiology context: {context.get('anatomy', [])} with {context.get('imaging_modality')} findings")
        )
        return state
    except Exception as e:
        state["error"] = f"Context extraction failed: {str(e)}"
        return state

@traceable(name="search_literature")
async def search_literature_node(state: RadiologyState) -> RadiologyState:
    """Search medical literature based on context"""
    try:
        if not state.get("radiology_context"):
            state["error"] = "No radiology context available for literature search"
            return state
            
        matches = await radiology_system.literature_agent.search_literature(
            context=state["radiology_context"],
            case_id=state["case_id"]
        )
        
        state["literature_matches"] = matches
        state["messages"].append(
            AIMessage(content=f"Found {len(matches)} relevant papers with imaging examples")
        )
        return state
    except Exception as e:
        state["error"] = f"Literature search failed: {str(e)}"
        return state

@traceable(name="generate_diagnosis")
async def generate_diagnosis_node(state: RadiologyState) -> RadiologyState:
    """Generate diagnosis based on context and literature"""
    try:
        diagnosis = await radiology_system.diagnosis_agent.generate_diagnosis(
            context=state.get("radiology_context", {}),
            literature_matches=state.get("literature_matches", []),
            case_id=state["case_id"]
        )
        
        state["diagnosis_result"] = diagnosis
        state["messages"].append(
            AIMessage(content=f"Primary diagnosis: {diagnosis.get('primary_diagnosis', {}).get('diagnosis', 'Unknown')}")
        )
        
        # Add processing metadata
        state["processing_metadata"] = {
            "models_used": ["claude-3-opus", "mistral-large", "deepseek-chat"],
            "literature_sources": len(state.get("literature_matches", [])),
            "langsmith_project": os.getenv("LANGCHAIN_PROJECT", "radiology-ai-system"),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return state
    except Exception as e:
        state["error"] = f"Diagnosis generation failed: {str(e)}"
        return state

# Create the graph
def create_radiology_graph():
    """Create the LangGraph workflow for radiology analysis"""
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
    
    return workflow.compile()

# Create the graph instance
graph = create_radiology_graph()

# Helper function to run analysis through the graph
async def analyze_case_with_graph(case_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run a case analysis through the LangGraph workflow"""
    
    # Initialize state
    initial_state = RadiologyState(
        case_id=case_data.get("case_id", f"case_{datetime.now().timestamp()}"),
        patient_age=case_data.get("patient_age", 0),
        patient_sex=case_data.get("patient_sex", ""),
        clinical_history=case_data.get("clinical_history", ""),
        imaging_modality=case_data.get("imaging_modality", ""),
        anatomical_region=case_data.get("anatomical_region", ""),
        image_description=case_data.get("image_description", ""),
        radiology_context={},
        literature_matches=[],
        diagnosis_result={},
        processing_metadata={},
        messages=[HumanMessage(content=f"Analyzing case: {case_data.get('clinical_history', '')[:100]}...")],
        error=None
    )
    
    # Run the graph
    result = await graph.ainvoke(initial_state)
    
    if result.get("error"):
        return {
            "error": True,
            "message": result["error"]
        }
    
    return {
        "case_id": result["case_id"],
        "radiology_context": result["radiology_context"],
        "literature_matches": result["literature_matches"],
        "diagnosis_result": result["diagnosis_result"],
        "processing_metadata": result["processing_metadata"],
        "messages": [msg.content for msg in result["messages"]]
    }
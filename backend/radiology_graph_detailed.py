"""
Detailed LangGraph implementation for Radiology AI System
Each LLM call is a separate node for individual tracking and analysis
"""
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langsmith import traceable
import os
from datetime import datetime
import json

# Import our existing modules
from main import radiology_system
from pydantic import BaseModel

# Define the comprehensive state for our graph
class DetailedRadiologyState(TypedDict):
    """State for the detailed radiology analysis workflow"""
    # Input data
    case_id: str
    patient_age: int
    patient_sex: str
    clinical_history: str
    imaging_modality: str
    anatomical_region: str
    image_description: str
    
    # Node 1: Radiology Context Extraction (Claude)
    radiology_context: Dict[str, Any]
    context_extraction_time: float
    context_extraction_tokens: int
    
    # Node 2: Literature Search Query Generation (DeepSeek)
    search_queries: List[str]
    query_generation_time: float
    query_generation_tokens: int
    
    # Node 3: Web Search Execution (Tool)
    raw_search_results: List[Dict[str, Any]]
    search_execution_time: float
    
    # Node 4: Document Relevance Analysis (Claude)
    document_relevance_scores: List[Dict[str, Any]]
    relevance_analysis_time: float
    relevance_analysis_tokens: int
    
    # Node 5: Image Description Extraction (Mistral)
    extracted_image_descriptions: List[Dict[str, Any]]
    image_extraction_time: float
    image_extraction_tokens: int
    
    # Node 6: Image Relevance Scoring (Claude)
    image_relevance_scores: List[Dict[str, Any]]
    image_scoring_time: float
    image_scoring_tokens: int
    
    # Node 7: Primary Diagnosis Generation (Claude)
    primary_diagnosis: Dict[str, Any]
    primary_diagnosis_time: float
    primary_diagnosis_tokens: int
    
    # Node 8: Differential Diagnosis Generation (Claude)
    differential_diagnoses: List[Dict[str, Any]]
    differential_diagnosis_time: float
    differential_diagnosis_tokens: int
    
    # Node 9: Confidence Assessment (Claude)
    confidence_assessment: Dict[str, Any]
    confidence_assessment_time: float
    confidence_assessment_tokens: int
    
    # Overall results
    literature_matches: List[Dict[str, Any]]
    diagnosis_result: Dict[str, Any]
    processing_metadata: Dict[str, Any]
    
    # Messages for tracking
    messages: Annotated[List[Any], add_messages]
    
    # Error handling
    errors: Dict[str, str]
    
# Node 1: Radiology Context Extraction (Claude)
@traceable(name="radiology_context_extraction")
async def extract_radiology_context_node(state: DetailedRadiologyState) -> DetailedRadiologyState:
    """Extract radiology context using Claude"""
    start_time = datetime.now()
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
        state["context_extraction_time"] = (datetime.now() - start_time).total_seconds()
        state["context_extraction_tokens"] = len(str(context)) // 4  # Rough estimate
        
        state["messages"].append(
            AIMessage(content=f"✅ Context extracted: {len(context.get('anatomy', []))} anatomical structures identified")
        )
    except Exception as e:
        state["errors"]["context_extraction"] = str(e)
        state["messages"].append(
            AIMessage(content=f"❌ Context extraction failed: {str(e)}")
        )
    
    return state

# Node 2: Literature Search Query Generation (DeepSeek)
@traceable(name="search_query_generation")
async def generate_search_queries_node(state: DetailedRadiologyState) -> DetailedRadiologyState:
    """Generate search queries using DeepSeek"""
    start_time = datetime.now()
    try:
        if not state.get("radiology_context"):
            state["errors"]["query_generation"] = "No radiology context available"
            return state
            
        # Generate queries using DeepSeek
        queries = await radiology_system.literature_agent.generate_search_queries(
            state["radiology_context"]
        )
        
        state["search_queries"] = queries
        state["query_generation_time"] = (datetime.now() - start_time).total_seconds()
        state["query_generation_tokens"] = len(" ".join(queries)) // 4
        
        state["messages"].append(
            AIMessage(content=f"✅ Generated {len(queries)} search queries")
        )
    except Exception as e:
        state["errors"]["query_generation"] = str(e)
        state["messages"].append(
            AIMessage(content=f"❌ Query generation failed: {str(e)}")
        )
    
    return state

# Node 3: Web Search Execution (Tool - not LLM)
@traceable(name="web_search_execution")
async def execute_web_search_node(state: DetailedRadiologyState) -> DetailedRadiologyState:
    """Execute web searches using Brave Search API"""
    start_time = datetime.now()
    try:
        if not state.get("search_queries"):
            state["errors"]["search_execution"] = "No search queries available"
            return state
            
        # Execute searches
        results = []
        for query in state["search_queries"][:3]:  # Limit to 3 queries
            search_results = await radiology_system.literature_agent.search_web(query)
            results.extend(search_results)
        
        state["raw_search_results"] = results
        state["search_execution_time"] = (datetime.now() - start_time).total_seconds()
        
        state["messages"].append(
            AIMessage(content=f"✅ Found {len(results)} search results")
        )
    except Exception as e:
        state["errors"]["search_execution"] = str(e)
        state["messages"].append(
            AIMessage(content=f"❌ Search execution failed: {str(e)}")
        )
    
    return state

# Node 4: Document Relevance Analysis (Claude)
@traceable(name="document_relevance_analysis")
async def analyze_document_relevance_node(state: DetailedRadiologyState) -> DetailedRadiologyState:
    """Analyze document relevance using Claude"""
    start_time = datetime.now()
    try:
        if not state.get("raw_search_results"):
            state["errors"]["relevance_analysis"] = "No search results to analyze"
            return state
            
        relevance_scores = []
        total_tokens = 0
        
        for doc in state["raw_search_results"][:10]:  # Analyze top 10
            score = await radiology_system.literature_agent.analyze_relevance(
                doc, state["radiology_context"]
            )
            relevance_scores.append({
                "url": doc.get("url"),
                "title": doc.get("title"),
                "score": score["relevance_score"],
                "reasoning": score["reasoning"]
            })
            total_tokens += len(str(score)) // 4
        
        state["document_relevance_scores"] = relevance_scores
        state["relevance_analysis_time"] = (datetime.now() - start_time).total_seconds()
        state["relevance_analysis_tokens"] = total_tokens
        
        relevant_count = sum(1 for s in relevance_scores if s["score"] > 0.7)
        state["messages"].append(
            AIMessage(content=f"✅ Analyzed {len(relevance_scores)} documents, {relevant_count} highly relevant")
        )
    except Exception as e:
        state["errors"]["relevance_analysis"] = str(e)
        state["messages"].append(
            AIMessage(content=f"❌ Relevance analysis failed: {str(e)}")
        )
    
    return state

# Node 5: Image Description Extraction (Mistral)
@traceable(name="image_description_extraction")
async def extract_image_descriptions_node(state: DetailedRadiologyState) -> DetailedRadiologyState:
    """Extract image descriptions using Mistral"""
    start_time = datetime.now()
    try:
        relevant_docs = [
            doc for doc, score in zip(state.get("raw_search_results", []), 
                                     state.get("document_relevance_scores", []))
            if score.get("score", 0) > 0.7
        ]
        
        if not relevant_docs:
            state["errors"]["image_extraction"] = "No relevant documents for image extraction"
            return state
            
        extracted_images = []
        total_tokens = 0
        
        for doc in relevant_docs[:5]:  # Process top 5 relevant docs
            images = await radiology_system.literature_agent.extract_images_from_content(
                doc.get("url"), doc.get("content", "")
            )
            extracted_images.extend(images)
            total_tokens += len(str(images)) // 4
        
        state["extracted_image_descriptions"] = extracted_images
        state["image_extraction_time"] = (datetime.now() - start_time).total_seconds()
        state["image_extraction_tokens"] = total_tokens
        
        state["messages"].append(
            AIMessage(content=f"✅ Extracted {len(extracted_images)} image descriptions")
        )
    except Exception as e:
        state["errors"]["image_extraction"] = str(e)
        state["messages"].append(
            AIMessage(content=f"❌ Image extraction failed: {str(e)}")
        )
    
    return state

# Node 6: Image Relevance Scoring (Claude)
@traceable(name="image_relevance_scoring")
async def score_image_relevance_node(state: DetailedRadiologyState) -> DetailedRadiologyState:
    """Score image relevance using Claude"""
    start_time = datetime.now()
    try:
        if not state.get("extracted_image_descriptions"):
            state["errors"]["image_scoring"] = "No images to score"
            return state
            
        scored_images = []
        total_tokens = 0
        
        for img in state["extracted_image_descriptions"]:
            score = await radiology_system.literature_agent.score_image_relevance(
                img, state["radiology_context"]
            )
            scored_images.append({
                **img,
                "relevance_score": score,
                "is_relevant": score > 0.7
            })
            total_tokens += 50  # Estimate for scoring
        
        state["image_relevance_scores"] = scored_images
        state["image_scoring_time"] = (datetime.now() - start_time).total_seconds()
        state["image_scoring_tokens"] = total_tokens
        
        relevant_images = sum(1 for img in scored_images if img["is_relevant"])
        state["messages"].append(
            AIMessage(content=f"✅ Scored {len(scored_images)} images, {relevant_images} highly relevant")
        )
    except Exception as e:
        state["errors"]["image_scoring"] = str(e)
        state["messages"].append(
            AIMessage(content=f"❌ Image scoring failed: {str(e)}")
        )
    
    return state

# Node 7: Primary Diagnosis Generation (Claude)
@traceable(name="primary_diagnosis_generation")
async def generate_primary_diagnosis_node(state: DetailedRadiologyState) -> DetailedRadiologyState:
    """Generate primary diagnosis using Claude"""
    start_time = datetime.now()
    try:
        # Prepare literature matches from scored documents and images
        literature_matches = []
        for doc, score in zip(state.get("raw_search_results", []), 
                             state.get("document_relevance_scores", [])):
            if score.get("score", 0) > 0.7:
                # Add relevant images to this document
                doc_images = [
                    img for img in state.get("image_relevance_scores", [])
                    if img.get("source_url") == doc.get("url") and img.get("is_relevant")
                ]
                literature_matches.append({
                    **doc,
                    "relevance_score": score["score"],
                    "match_reasoning": score["reasoning"],
                    "extracted_images": doc_images
                })
        
        state["literature_matches"] = literature_matches
        
        # Generate primary diagnosis
        diagnosis = await radiology_system.diagnosis_agent.generate_primary_diagnosis(
            context=state["radiology_context"],
            literature_matches=literature_matches
        )
        
        state["primary_diagnosis"] = diagnosis
        state["primary_diagnosis_time"] = (datetime.now() - start_time).total_seconds()
        state["primary_diagnosis_tokens"] = len(str(diagnosis)) // 4
        
        state["messages"].append(
            AIMessage(content=f"✅ Primary diagnosis: {diagnosis.get('diagnosis', 'Unknown')} "
                            f"(confidence: {diagnosis.get('confidence_score', 0):.2f})")
        )
    except Exception as e:
        state["errors"]["primary_diagnosis"] = str(e)
        state["messages"].append(
            AIMessage(content=f"❌ Primary diagnosis failed: {str(e)}")
        )
    
    return state

# Node 8: Differential Diagnosis Generation (Claude)
@traceable(name="differential_diagnosis_generation")
async def generate_differential_diagnosis_node(state: DetailedRadiologyState) -> DetailedRadiologyState:
    """Generate differential diagnoses using Claude"""
    start_time = datetime.now()
    try:
        if not state.get("primary_diagnosis"):
            state["errors"]["differential_diagnosis"] = "No primary diagnosis available"
            return state
            
        differentials = await radiology_system.diagnosis_agent.generate_differential_diagnoses(
            primary_diagnosis=state["primary_diagnosis"],
            context=state["radiology_context"]
        )
        
        state["differential_diagnoses"] = differentials
        state["differential_diagnosis_time"] = (datetime.now() - start_time).total_seconds()
        state["differential_diagnosis_tokens"] = len(str(differentials)) // 4
        
        state["messages"].append(
            AIMessage(content=f"✅ Generated {len(differentials)} differential diagnoses")
        )
    except Exception as e:
        state["errors"]["differential_diagnosis"] = str(e)
        state["messages"].append(
            AIMessage(content=f"❌ Differential diagnosis failed: {str(e)}")
        )
    
    return state

# Node 9: Confidence Assessment (Claude)
@traceable(name="confidence_assessment")
async def assess_confidence_node(state: DetailedRadiologyState) -> DetailedRadiologyState:
    """Assess diagnostic confidence using Claude"""
    start_time = datetime.now()
    try:
        if not state.get("primary_diagnosis"):
            state["errors"]["confidence_assessment"] = "No diagnosis to assess"
            return state
            
        assessment = await radiology_system.diagnosis_agent.assess_confidence(
            primary=state["primary_diagnosis"],
            differentials=state.get("differential_diagnoses", []),
            literature_count=len(state.get("literature_matches", [])),
            average_relevance=sum(m.get("relevance_score", 0) for m in state.get("literature_matches", [])) / max(len(state.get("literature_matches", [])), 1)
        )
        
        state["confidence_assessment"] = assessment
        state["confidence_assessment_time"] = (datetime.now() - start_time).total_seconds()
        state["confidence_assessment_tokens"] = len(str(assessment)) // 4
        
        # Compile final results
        state["diagnosis_result"] = {
            "primary_diagnosis": state["primary_diagnosis"],
            "differential_diagnoses": state["differential_diagnoses"],
            "confidence_assessment": assessment
        }
        
        # Add comprehensive processing metadata
        state["processing_metadata"] = {
            "case_id": state["case_id"],
            "timestamp": datetime.utcnow().isoformat(),
            "models_used": ["claude-3-opus", "mistral-large", "deepseek-chat"],
            "langsmith_project": os.getenv("LANGCHAIN_PROJECT", "radiology-ai-system"),
            "total_llm_calls": 8,
            "llm_call_details": {
                "context_extraction": {
                    "model": "claude",
                    "time": state.get("context_extraction_time", 0),
                    "tokens": state.get("context_extraction_tokens", 0)
                },
                "query_generation": {
                    "model": "deepseek",
                    "time": state.get("query_generation_time", 0),
                    "tokens": state.get("query_generation_tokens", 0)
                },
                "relevance_analysis": {
                    "model": "claude",
                    "time": state.get("relevance_analysis_time", 0),
                    "tokens": state.get("relevance_analysis_tokens", 0)
                },
                "image_extraction": {
                    "model": "mistral",
                    "time": state.get("image_extraction_time", 0),
                    "tokens": state.get("image_extraction_tokens", 0)
                },
                "image_scoring": {
                    "model": "claude",
                    "time": state.get("image_scoring_time", 0),
                    "tokens": state.get("image_scoring_tokens", 0)
                },
                "primary_diagnosis": {
                    "model": "claude",
                    "time": state.get("primary_diagnosis_time", 0),
                    "tokens": state.get("primary_diagnosis_tokens", 0)
                },
                "differential_diagnosis": {
                    "model": "claude",
                    "time": state.get("differential_diagnosis_time", 0),
                    "tokens": state.get("differential_diagnosis_tokens", 0)
                },
                "confidence_assessment": {
                    "model": "claude",
                    "time": state.get("confidence_assessment_time", 0),
                    "tokens": state.get("confidence_assessment_tokens", 0)
                }
            },
            "total_processing_time": sum(
                state.get(f"{step}_time", 0) 
                for step in ["context_extraction", "query_generation", "search_execution",
                           "relevance_analysis", "image_extraction", "image_scoring",
                           "primary_diagnosis", "differential_diagnosis", "confidence_assessment"]
            ),
            "total_tokens_used": sum(
                state.get(f"{step}_tokens", 0)
                for step in ["context_extraction", "query_generation", "relevance_analysis",
                           "image_extraction", "image_scoring", "primary_diagnosis",
                           "differential_diagnosis", "confidence_assessment"]
            ),
            "literature_sources": len(state.get("literature_matches", [])),
            "images_analyzed": len(state.get("image_relevance_scores", [])),
            "errors": state.get("errors", {})
        }
        
        state["messages"].append(
            AIMessage(content=f"✅ Confidence assessment complete. Overall confidence: "
                            f"{assessment.get('overall_confidence', 0):.2f}")
        )
    except Exception as e:
        state["errors"]["confidence_assessment"] = str(e)
        state["messages"].append(
            AIMessage(content=f"❌ Confidence assessment failed: {str(e)}")
        )
    
    return state

# Create the detailed graph
def create_detailed_radiology_graph():
    """Create the detailed LangGraph workflow with all 8 LLM nodes"""
    workflow = StateGraph(DetailedRadiologyState)
    
    # Add all nodes (8 LLM + 1 tool)
    workflow.add_node("extract_radiology_context", extract_radiology_context_node)
    workflow.add_node("generate_search_queries", generate_search_queries_node)
    workflow.add_node("execute_web_search", execute_web_search_node)
    workflow.add_node("analyze_document_relevance", analyze_document_relevance_node)
    workflow.add_node("extract_image_descriptions", extract_image_descriptions_node)
    workflow.add_node("score_image_relevance", score_image_relevance_node)
    workflow.add_node("generate_primary_diagnosis", generate_primary_diagnosis_node)
    workflow.add_node("generate_differential_diagnosis", generate_differential_diagnosis_node)
    workflow.add_node("assess_confidence", assess_confidence_node)
    
    # Define the linear flow
    workflow.add_edge("extract_radiology_context", "generate_search_queries")
    workflow.add_edge("generate_search_queries", "execute_web_search")
    workflow.add_edge("execute_web_search", "analyze_document_relevance")
    workflow.add_edge("analyze_document_relevance", "extract_image_descriptions")
    workflow.add_edge("extract_image_descriptions", "score_image_relevance")
    workflow.add_edge("score_image_relevance", "generate_primary_diagnosis")
    workflow.add_edge("generate_primary_diagnosis", "generate_differential_diagnosis")
    workflow.add_edge("generate_differential_diagnosis", "assess_confidence")
    workflow.add_edge("assess_confidence", END)
    
    # Set entry point
    workflow.set_entry_point("extract_radiology_context")
    
    return workflow.compile()

# Create the graph instance
detailed_graph = create_detailed_radiology_graph()

# Helper function to run analysis through the detailed graph
async def analyze_case_with_detailed_graph(case_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run a case analysis through the detailed LangGraph workflow"""
    
    # Initialize state with all fields
    initial_state = DetailedRadiologyState(
        # Input data
        case_id=case_data.get("case_id", f"case_{datetime.now().timestamp()}"),
        patient_age=case_data.get("patient_age", 0),
        patient_sex=case_data.get("patient_sex", ""),
        clinical_history=case_data.get("clinical_history", ""),
        imaging_modality=case_data.get("imaging_modality", ""),
        anatomical_region=case_data.get("anatomical_region", ""),
        image_description=case_data.get("image_description", ""),
        
        # Initialize all tracking fields
        radiology_context={},
        context_extraction_time=0.0,
        context_extraction_tokens=0,
        
        search_queries=[],
        query_generation_time=0.0,
        query_generation_tokens=0,
        
        raw_search_results=[],
        search_execution_time=0.0,
        
        document_relevance_scores=[],
        relevance_analysis_time=0.0,
        relevance_analysis_tokens=0,
        
        extracted_image_descriptions=[],
        image_extraction_time=0.0,
        image_extraction_tokens=0,
        
        image_relevance_scores=[],
        image_scoring_time=0.0,
        image_scoring_tokens=0,
        
        primary_diagnosis={},
        primary_diagnosis_time=0.0,
        primary_diagnosis_tokens=0,
        
        differential_diagnoses=[],
        differential_diagnosis_time=0.0,
        differential_diagnosis_tokens=0,
        
        confidence_assessment={},
        confidence_assessment_time=0.0,
        confidence_assessment_tokens=0,
        
        literature_matches=[],
        diagnosis_result={},
        processing_metadata={},
        messages=[HumanMessage(content=f"Starting detailed analysis for case: {case_data.get('case_id', 'unknown')}")],
        errors={}
    )
    
    # Run the graph
    result = await detailed_graph.ainvoke(initial_state)
    
    # Return structured results
    return {
        "case_id": result["case_id"],
        "radiology_context": result["radiology_context"],
        "literature_matches": result["literature_matches"],
        "diagnosis_result": result["diagnosis_result"],
        "processing_metadata": result["processing_metadata"],
        "messages": [msg.content for msg in result["messages"]],
        "errors": result["errors"]
    }

# For backward compatibility, export as 'graph'
graph = detailed_graph
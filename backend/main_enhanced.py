"""
Advanced Radiology AI System - LangChain + LangSmith
Enhanced version with improved literature search and image extraction
"""

import os
import asyncio
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path
import structlog
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings

# LangChain Core Imports
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.tools import Tool, BaseTool
from langchain_core.language_models import BaseLanguageModel

# LangChain Model Integrations
from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI
from langchain_deepseek import ChatDeepSeek

# LangChain Tools and Utilities
from langchain_community.tools import BraveSearch
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LangSmith Integration
from langsmith import Client
from langsmith.schemas import Run, Example
from langchain.callbacks.tracers import LangChainTracer

# Database and Storage
from motor.motor_asyncio import AsyncIOMotorClient
import pymongo
from bson import ObjectId
from bson.json_util import dumps, loads

# Web Framework
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn

# Import enhanced literature search
from enhanced_literature_search import EnhancedLiteratureSearchAgent, EnhancedLiteratureMatch, ExtractedImage

# Configure structured logging
logger = structlog.get_logger()

class Settings(BaseSettings):
    """Application settings with complete API configuration"""
    
    # LangSmith Configuration
    langchain_tracing_v2: bool = True
    langchain_api_key: str = Field(..., env="LANGCHAIN_API_KEY")
    langchain_project: str = Field("radiology-ai-system", env="LANGCHAIN_PROJECT")
    langchain_endpoint: str = Field("https://api.smith.langchain.com", env="LANGCHAIN_ENDPOINT")
    
    # AI Model Configuration
    anthropic_api_key: str = Field(..., env="ANTHROPIC_API_KEY")
    deepseek_api_key: str = Field(..., env="DEEPSEEK_API_KEY")
    mistral_api_key: str = Field(..., env="MISTRAL_API_KEY")
    
    # Search Configuration
    brave_search_api_key: str = Field(..., env="BRAVE_SEARCH_API_KEY")
    
    # Database Configuration
    mongodb_url: str = Field("mongodb://localhost:27017", env="MONGODB_URL")
    database_name: str = Field("radiology_ai_langchain", env="DATABASE_NAME")
    
    # Application Configuration
    debug: bool = Field(True, env="DEBUG")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    host: str = Field("0.0.0.0", env="HOST")
    port: int = Field(8000, env="PORT")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Initialize settings
settings = Settings()

# Configure LangSmith - MUST be done before any LangChain imports
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint

# Initialize LangSmith client and tracer
langsmith_client = Client(
    api_url=settings.langchain_endpoint,
    api_key=settings.langchain_api_key
)

# Create global tracer for callbacks
langsmith_tracer = LangChainTracer(
    project_name=settings.langchain_project,
    client=langsmith_client
)

# Pydantic Models
class ClinicalCase(BaseModel):
    """Input model for radiology case analysis"""
    case_id: str = Field(..., description="Unique case identifier")
    patient_age: int = Field(..., ge=0, le=150, description="Patient age in years")
    patient_sex: str = Field(..., description="Patient sex", pattern="^(Male|Female|Other)$")
    clinical_history: str = Field(..., description="Clinical history and presentation")
    imaging_modality: str = Field(..., description="Primary imaging modality")
    anatomical_region: str = Field(..., description="Anatomical region of interest")
    image_description: str = Field(..., description="Detailed imaging findings")
    
    @validator('imaging_modality')
    def validate_modality(cls, v):
        valid_modalities = ['MRI', 'CT', 'X-ray', 'Ultrasound', 'PET', 'Nuclear Medicine']
        if v not in valid_modalities:
            raise ValueError(f"Invalid modality. Must be one of: {valid_modalities}")
        return v

class RadiologyContext(BaseModel):
    """Extracted radiology context from case"""
    anatomy: List[str] = Field(..., description="Anatomical structures involved")
    imaging_modality: str = Field(..., description="Imaging modality used")
    sequences: List[str] = Field(default_factory=list, description="Imaging sequences (MRI)")
    measurements: Dict[str, str] = Field(default_factory=dict, description="Key measurements")
    morphology: List[str] = Field(default_factory=list, description="Morphological features")
    location: Dict[str, str] = Field(default_factory=dict, description="Location descriptors")
    signal_characteristics: List[str] = Field(default_factory=list, description="Signal/density characteristics")
    enhancement_pattern: List[str] = Field(default_factory=list, description="Enhancement patterns")
    associated_findings: List[str] = Field(default_factory=list, description="Associated findings")
    clinical_context: str = Field(..., description="Relevant clinical context")

class Diagnosis(BaseModel):
    """Single diagnosis with reasoning"""
    diagnosis: str = Field(..., description="Diagnostic entity")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    reasoning: str = Field(..., description="Clinical reasoning")
    icd_code: Optional[str] = Field(None, description="ICD-10 code if applicable")
    supporting_evidence: Optional[str] = Field(None, description="Supporting evidence from literature")
    imaging_features: Optional[str] = Field(None, description="Key imaging features")

class DifferentialDiagnosis(BaseModel):
    """Differential diagnosis entry"""
    diagnosis: str = Field(..., description="Diagnostic entity")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probability score")
    reasoning: str = Field(..., description="Why this is considered")
    distinguishing_features: Optional[str] = Field(None, description="Features that would distinguish this diagnosis")
    additional_workup: Optional[str] = Field(None, description="Additional tests needed")

class ConfidenceAssessment(BaseModel):
    """Overall confidence assessment"""
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    evidence_quality: str = Field(..., description="Quality of evidence")
    clinical_correlation: str = Field(..., description="Clinical correlation assessment")
    diagnostic_certainty: float = Field(..., ge=0.0, le=1.0)
    recommendation: str = Field(..., description="Next steps recommendation")
    limitations: Optional[str] = Field(None, description="Limitations of the analysis")

class DiagnosisResult(BaseModel):
    """Complete diagnosis result"""
    primary_diagnosis: Diagnosis
    differential_diagnoses: List[DifferentialDiagnosis]
    confidence_assessment: ConfidenceAssessment
    literature_support: str = Field(..., description="Summary of literature support")

class PromptTemplate(BaseModel):
    """Dynamic prompt template with versioning"""
    template_id: str = Field(..., description="Unique template identifier")
    name: str = Field(..., description="Human-readable template name")
    version: str = Field(..., description="Template version")
    description: str = Field(..., description="Template description")
    template_text: str = Field(..., description="Actual prompt template")
    input_variables: List[str] = Field(..., description="Required input variables")
    model_type: str = Field(..., description="Target model type (claude, mistral, deepseek)")
    performance_metrics: Dict[str, float] = Field(default_factory=dict, description="Performance tracking")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class PromptManager:
    """Dynamic prompt management system with database storage"""
    
    def __init__(self, db_client):
        self.db = db_client
        self.collection = self.db.prompts
        
    async def get_prompt(self, template_id: str, version: Optional[str] = None) -> PromptTemplate:
        """Retrieve prompt template by ID and version"""
        query = {"template_id": template_id}
        if version:
            query["version"] = version
        else:
            # Get latest version
            query = {"template_id": template_id}
            
        prompt_doc = await self.collection.find_one(
            query, sort=[("version", pymongo.DESCENDING)]
        )
        
        if not prompt_doc:
            raise ValueError(f"Prompt template {template_id} not found")
            
        # Remove _id field to avoid ObjectId serialization issues
        if '_id' in prompt_doc:
            del prompt_doc['_id']
            
        return PromptTemplate(**prompt_doc)
    
    async def save_prompt(self, prompt: PromptTemplate) -> str:
        """Save or update prompt template"""
        prompt.updated_at = datetime.now()
        result = await self.collection.replace_one(
            {"template_id": prompt.template_id, "version": prompt.version},
            prompt.dict(),
            upsert=True
        )
        return str(result.upserted_id) if result.upserted_id else prompt.template_id

class MultiModelOrchestrator:
    """Orchestrates multiple models for different tasks"""
    
    def __init__(self, prompt_manager: PromptManager):
        self.prompt_manager = prompt_manager
        
        # Initialize models with callbacks
        self.claude = ChatAnthropic(
            model="claude-3-sonnet-20240229",
            api_key=settings.anthropic_api_key,
            temperature=0.1,
            max_tokens=4000,
            callbacks=[langsmith_tracer]
        )
        
        self.mistral = ChatMistralAI(
            model="mistral-large-latest",
            api_key=settings.mistral_api_key,
            temperature=0.2,
            max_tokens=4000,
            callbacks=[langsmith_tracer]
        )
        
        self.deepseek = ChatDeepSeek(
            model="deepseek-chat",
            api_key=settings.deepseek_api_key,
            api_base="https://api.deepseek.com/v1",
            temperature=0.3,
            callbacks=[langsmith_tracer]
        )
        
        logger.info("Initialized all models with LangSmith tracing")
        
    def select_model(self, task_type: str) -> BaseLanguageModel:
        """Select appropriate model based on task"""
        model_selection = {
            "medical_reasoning": self.claude,
            "document_processing": self.mistral,
            "search_optimization": self.deepseek,
            "diagnosis": self.claude,
            "context_extraction": self.claude
        }
        return model_selection.get(task_type, self.claude)

class RadiologyContextExtractor:
    """Extracts structured radiology context from cases"""
    
    def __init__(self, orchestrator: MultiModelOrchestrator):
        self.orchestrator = orchestrator
        
    async def extract_context(self, case: ClinicalCase) -> RadiologyContext:
        """Extract structured radiology context"""
        
        run_name = f"radiology_context_extraction_{case.case_id}"
        
        model = self.orchestrator.select_model("context_extraction")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert radiologist. Extract structured information from the case.
            Focus on anatomical structures, imaging characteristics, and clinically relevant features.
            Be specific and comprehensive."""),
            ("human", """Extract radiology context from this case:
            Patient: {age} year old {sex}
            Clinical History: {clinical_history}
            Imaging Modality: {imaging_modality}
            Anatomical Region: {anatomical_region}
            Image Description: {image_description}
            
            Provide a JSON response with these fields:
            - anatomy: list of anatomical structures
            - imaging_modality: the modality used
            - sequences: list of sequences (for MRI)
            - measurements: dict of key measurements
            - morphology: list of morphological features
            - location: dict of location descriptors
            - signal_characteristics: list of signal/density characteristics
            - enhancement_pattern: list of enhancement patterns
            - associated_findings: list of associated findings
            - clinical_context: summary of clinical relevance""")
        ])
        
        parser = PydanticOutputParser(pydantic_object=RadiologyContext)
        
        chain = prompt | model | parser
        
        try:
            result = await chain.ainvoke({
                "age": case.patient_age,
                "sex": case.patient_sex,
                "clinical_history": case.clinical_history,
                "imaging_modality": case.imaging_modality,
                "anatomical_region": case.anatomical_region,
                "image_description": case.image_description
            }, config={"run_name": run_name, "callbacks": [langsmith_tracer]})
            
            return result
        except Exception as e:
            logger.error(f"Context extraction failed: {e}")
            # Return a basic context as fallback
            return RadiologyContext(
                anatomy=[case.anatomical_region],
                imaging_modality=case.imaging_modality,
                clinical_context=case.clinical_history
            )

class DiagnosisAgent:
    """Generates comprehensive diagnosis with differential"""
    
    def __init__(self, orchestrator: MultiModelOrchestrator):
        self.orchestrator = orchestrator
        
    async def generate_diagnosis(self, case: ClinicalCase, 
                               radiology_context: RadiologyContext,
                               literature_matches: List[EnhancedLiteratureMatch]) -> DiagnosisResult:
        """Generate comprehensive diagnosis"""
        
        model = self.orchestrator.select_model("diagnosis")
        
        # Prepare literature summary including images
        lit_summary = self._prepare_literature_summary(literature_matches)
        
        # Generate primary diagnosis
        primary_diagnosis = await self._generate_primary_diagnosis(
            case, radiology_context, lit_summary, model
        )
        
        # Generate differential diagnoses
        differentials = await self._generate_differential_diagnoses(
            case, radiology_context, primary_diagnosis, model
        )
        
        # Assess confidence
        confidence = await self._assess_confidence(
            primary_diagnosis, differentials, literature_matches, model
        )
        
        return DiagnosisResult(
            primary_diagnosis=primary_diagnosis,
            differential_diagnoses=differentials,
            confidence_assessment=confidence,
            literature_support=lit_summary
        )
    
    def _prepare_literature_summary(self, matches: List[EnhancedLiteratureMatch]) -> str:
        """Prepare literature summary including image references"""
        if not matches:
            return "No relevant literature found."
        
        summary_parts = []
        for i, match in enumerate(matches[:10], 1):
            # Basic paper info
            summary_parts.append(f"Study {i}: {match.title}")
            summary_parts.append(f"Relevance: {match.relevance_score:.2f}")
            summary_parts.append(f"Key findings: {match.match_reasoning}")
            
            # Image findings
            if match.extracted_images:
                summary_parts.append(f"Images found: {len(match.extracted_images)} relevant figures")
                for img in match.extracted_images[:2]:  # Top 2 images
                    summary_parts.append(f"- {img.figure_number or 'Image'}: {img.caption[:100]}...")
            
            summary_parts.append("")  # Blank line
        
        return "\n".join(summary_parts)
    
    async def _generate_primary_diagnosis(self, case, context, lit_summary, model) -> Diagnosis:
        """Generate primary diagnosis"""
        
        run_name = f"primary_diagnosis_generation_{case.case_id}"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert radiologist providing a primary diagnosis.
            Consider imaging findings, clinical context, and literature evidence.
            Be specific and evidence-based."""),
            ("human", """Based on this case, provide the most likely diagnosis:
            
            Clinical Context:
            {clinical_context}
            
            Radiology Findings:
            {radiology_findings}
            
            Literature Evidence:
            {literature_evidence}
            
            Provide a JSON response with:
            - diagnosis: the primary diagnosis name
            - confidence_score: 0.0 to 1.0
            - reasoning: detailed clinical reasoning
            - icd_code: ICD-10 code if applicable
            - supporting_evidence: key supporting evidence
            - imaging_features: characteristic imaging features""")
        ])
        
        parser = PydanticOutputParser(pydantic_object=Diagnosis)
        chain = prompt | model | parser
        
        result = await chain.ainvoke({
            "clinical_context": f"{case.patient_age}yo {case.patient_sex}, {case.clinical_history}",
            "radiology_findings": json.dumps(context.dict(), indent=2),
            "literature_evidence": lit_summary
        }, config={"run_name": run_name, "callbacks": [langsmith_tracer]})
        
        return result
    
    async def _generate_differential_diagnoses(self, case, context, primary, model) -> List[DifferentialDiagnosis]:
        """Generate differential diagnoses"""
        
        run_name = f"differential_diagnosis_generation_{case.case_id}"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert radiologist generating differential diagnoses.
            Consider alternative explanations for the imaging findings.
            Rank by probability."""),
            ("human", """Given this primary diagnosis and case details, provide differential diagnoses:
            
            Primary Diagnosis: {primary_diagnosis}
            
            Case Summary:
            {case_summary}
            
            Provide a JSON array of differential diagnoses, each with:
            - diagnosis: diagnosis name
            - probability: 0.0 to 1.0
            - reasoning: why this is considered
            - distinguishing_features: what would distinguish this
            - additional_workup: tests to confirm/exclude""")
        ])
        
        result_text = await model.ainvoke(
            prompt.format_messages(
                primary_diagnosis=primary.diagnosis,
                case_summary=f"{case.patient_age}yo {case.patient_sex}\n{context.clinical_context}\nKey findings: {', '.join(context.morphology)}"
            ),
            config={"run_name": run_name, "callbacks": [langsmith_tracer]}
        )
        
        # Parse JSON array
        try:
            diff_list = json.loads(result_text.content)
            return [DifferentialDiagnosis(**d) for d in diff_list[:5]]  # Top 5
        except:
            return []
    
    async def _assess_confidence(self, primary, differentials, literature, model) -> ConfidenceAssessment:
        """Assess overall diagnostic confidence"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Assess the overall confidence in the diagnosis based on:
            1. Strength of imaging findings
            2. Literature support
            3. Clinical correlation
            4. Differential considerations"""),
            ("human", """Assess confidence for:
            Primary: {primary}
            Differentials: {differentials}
            Literature matches: {lit_count} papers with average relevance {avg_relevance}
            
            Provide JSON with:
            - overall_confidence: 0.0 to 1.0
            - evidence_quality: High/Medium/Low
            - clinical_correlation: assessment
            - diagnostic_certainty: 0.0 to 1.0
            - recommendation: next steps
            - limitations: any limitations""")
        ])
        
        parser = PydanticOutputParser(pydantic_object=ConfidenceAssessment)
        chain = prompt | model | parser
        
        avg_relevance = sum(m.relevance_score for m in literature) / len(literature) if literature else 0
        
        result = await chain.ainvoke({
            "primary": f"{primary.diagnosis} (confidence: {primary.confidence_score})",
            "differentials": ", ".join([d.diagnosis for d in differentials]),
            "lit_count": len(literature),
            "avg_relevance": f"{avg_relevance:.2f}"
        }, config={"callbacks": [langsmith_tracer]})
        
        return result

class RadiologyAISystem:
    """Main AI system orchestrating all agents"""
    
    def __init__(self):
        # Initialize database
        self.db_client = AsyncIOMotorClient(settings.mongodb_url)
        self.db = self.db_client[settings.database_name]
        
        # Initialize managers
        self.prompt_manager = PromptManager(self.db)
        self.orchestrator = MultiModelOrchestrator(self.prompt_manager)
        
        # Initialize agents
        self.context_extractor = RadiologyContextExtractor(self.orchestrator)
        
        # Initialize enhanced literature search
        self.literature_agent = EnhancedLiteratureSearchAgent(
            deepseek_model=self.orchestrator.deepseek,
            mistral_model=self.orchestrator.mistral,
            claude_model=self.orchestrator.claude,
            brave_api_key=settings.brave_search_api_key,
            langsmith_tracer=langsmith_tracer
        )
        
        self.diagnosis_agent = DiagnosisAgent(self.orchestrator)
        
        logger.info("Initialized RadiologyAISystem with enhanced literature search")
        
    async def analyze_case(self, case: ClinicalCase) -> Dict:
        """Complete case analysis pipeline"""
        
        run_name = f"full_case_analysis_{case.case_id}"
        
        try:
            # Extract radiology context
            logger.info(f"Starting analysis for case {case.case_id}")
            radiology_context = await self.context_extractor.extract_context(case)
            logger.info(f"Extracted radiology context with {len(radiology_context.anatomy)} anatomical structures")
            
            # Search literature with enhanced image extraction
            try:
                literature_matches = await self.literature_agent.search_literature_with_images(
                    radiology_context, case, max_papers=10
                )
                logger.info(f"Found {len(literature_matches)} relevant papers with images")
            except Exception as e:
                logger.error(f"Literature search failed: {e}")
                literature_matches = []
            
            # Generate comprehensive diagnosis
            diagnosis_result = await self.diagnosis_agent.generate_diagnosis(
                case, radiology_context, literature_matches
            )
            
            # Store in database
            result_doc = {
                "case_id": case.case_id,
                "timestamp": datetime.now(),
                "radiology_context": radiology_context.dict(),
                "literature_matches": [match.dict() for match in literature_matches],
                "diagnosis_result": diagnosis_result.dict(),
                "langsmith_run_id": run_name
            }
            
            await self.db.analyses.insert_one(result_doc)
            
            # Prepare response with enhanced literature data
            response = {
                "case_id": case.case_id,
                "radiology_context": radiology_context.dict(),
                "literature_matches": [match.dict() for match in literature_matches],
                "diagnosis_result": diagnosis_result.dict(),
                "processing_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "models_used": ["claude", "mistral", "deepseek"],
                    "literature_sources": len(literature_matches),
                    "images_extracted": sum(len(m.extracted_images) for m in literature_matches),
                    "langsmith_project": settings.langchain_project
                }
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Case analysis failed: {e}", exc_info=True)
            raise

# FastAPI Application
app = FastAPI(
    title="Radiology AI System - Enhanced",
    description="Advanced radiology analysis with LangChain, LangSmith, and enhanced image extraction",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize system
radiology_system = None

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    global radiology_system
    radiology_system = RadiologyAISystem()
    logger.info("Radiology AI System started with enhanced literature search")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Radiology AI System - Enhanced",
        "version": "2.0.0",
        "features": [
            "Multi-model orchestration (Claude, Mistral, DeepSeek)",
            "Enhanced literature search with image extraction",
            "Full LangSmith observability",
            "Dynamic prompt management",
            "Comprehensive diagnosis generation"
        ],
        "endpoints": {
            "health": "/health",
            "analyze": "/api/analyze-case",
            "prompts": "/api/prompts"
        }
    }

@app.get("/health")
async def health_check():
    """System health check"""
    try:
        # Check database connection
        await radiology_system.db.command("ping")
        db_status = True
    except:
        db_status = False
    
    return {
        "status": "healthy" if db_status else "degraded",
        "timestamp": datetime.now().isoformat(),
        "database_connected": db_status,
        "langsmith_enabled": True,
        "langsmith_project": settings.langchain_project,
        "models_available": ["claude", "mistral", "deepseek"],
        "enhanced_features": {
            "image_extraction": True,
            "literature_limit": 10,
            "image_relevance_scoring": True
        }
    }

@app.post("/api/analyze-case")
async def analyze_case(case: ClinicalCase):
    """Analyze a radiology case with enhanced literature search"""
    try:
        logger.info(f"Received case analysis request: {case.case_id}")
        result = await radiology_system.analyze_case(case)
        return result
    except Exception as e:
        logger.error(f"Case analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/prompts")
async def list_prompts():
    """List available prompt templates"""
    try:
        prompts = await radiology_system.db.prompts.find({}).to_list(length=100)
        for prompt in prompts:
            if '_id' in prompt:
                prompt['_id'] = str(prompt['_id'])
        return {"prompts": prompts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/prompts")
async def save_prompt(prompt: PromptTemplate):
    """Save or update a prompt template"""
    try:
        result = await radiology_system.prompt_manager.save_prompt(prompt)
        return {"success": True, "prompt_id": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main_enhanced:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
"""
Advanced Radiology AI System - LangChain + LangSmith
Fixed version with proper tracing and DeepSeek integration
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

class RadiologyContext(BaseModel):
    """Structured radiology context extracted from clinical description"""
    anatomy: List[str] = Field(default_factory=list, description="Anatomical structures mentioned")
    imaging_modality: str = Field(..., description="Primary imaging modality")
    sequences: List[str] = Field(default_factory=list, description="Imaging sequences/protocols")
    measurements: Dict[str, str] = Field(default_factory=dict, description="Size measurements")
    morphology: List[str] = Field(default_factory=list, description="Shape and morphological features")
    location: Dict[str, str] = Field(default_factory=dict, description="Anatomical location details")
    signal_characteristics: List[str] = Field(default_factory=list, description="Signal/density characteristics")
    enhancement_pattern: List[str] = Field(default_factory=list, description="Enhancement patterns")
    associated_findings: List[str] = Field(default_factory=list, description="Associated findings")
    clinical_context: str = Field(..., description="Clinical presentation summary")

class ClinicalCase(BaseModel):
    """Complete clinical case representation"""
    case_id: str = Field(..., description="Unique case identifier")
    patient_age: int = Field(..., description="Patient age in years")
    patient_sex: str = Field(..., description="Patient biological sex")
    clinical_history: str = Field(..., description="Clinical history and symptoms")
    imaging_modality: str = Field(..., description="Type of imaging study")
    anatomical_region: str = Field(..., description="Anatomical region imaged")
    image_description: str = Field(..., description="Detailed imaging findings")
    radiology_context: Optional[RadiologyContext] = Field(None, description="Extracted radiology context")
    
class LiteratureMatch(BaseModel):
    """Medical literature match with relevance scoring"""
    title: str = Field(..., description="Paper title")
    authors: List[str] = Field(default_factory=list, description="Authors")
    journal: str = Field(..., description="Publication journal")
    year: int = Field(..., description="Publication year")
    doi: Optional[str] = Field(None, description="DOI if available")
    url: str = Field(..., description="Access URL")
    abstract: str = Field(..., description="Paper abstract")
    relevant_sections: List[str] = Field(default_factory=list, description="Relevant text sections")
    image_descriptions: List[str] = Field(default_factory=list, description="Extracted image descriptions")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    match_reasoning: str = Field(..., description="Why this paper is relevant")

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
    
    async def create_prompt_version(self, template_id: str, new_template_text: str, 
                                  description: str = "Updated version") -> PromptTemplate:
        """Create new version of existing prompt"""
        latest = await self.get_prompt(template_id)
        
        # Increment version
        version_parts = latest.version.split(".")
        version_parts[-1] = str(int(version_parts[-1]) + 1)
        new_version = ".".join(version_parts)
        
        new_prompt = PromptTemplate(
            template_id=template_id,
            name=latest.name,
            version=new_version,
            description=description,
            template_text=new_template_text,
            input_variables=latest.input_variables,
            model_type=latest.model_type
        )
        
        await self.save_prompt(new_prompt)
        return new_prompt

class MultiModelOrchestrator:
    """Orchestrates multiple AI models for specialized tasks"""
    
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
            temperature=0.1,
            max_tokens=4000,
            callbacks=[langsmith_tracer]
        )
        
        # DeepSeek with proper integration
        self.deepseek = ChatDeepSeek(
            model="deepseek-chat",  # Using DeepSeek-V3 which supports structured output
            api_key=settings.deepseek_api_key,
            temperature=0.1,
            max_tokens=4000,
            callbacks=[langsmith_tracer]
        )
        
        logger.info("MultiModelOrchestrator initialized with all models")
    
    async def get_model_for_task(self, task_type: str) -> BaseLanguageModel:
        """Select optimal model for specific task"""
        model_mapping = {
            "radiology_extraction": self.claude,  # Best for medical reasoning
            "document_processing": self.mistral,  # Good for document analysis
            "search_optimization": self.deepseek,  # Good for code/optimization
            "diagnosis_generation": self.claude,  # Best for medical diagnosis
            "literature_analysis": self.claude,   # Best for medical analysis
            "synthesis": self.claude              # Best for complex reasoning
        }
        
        return model_mapping.get(task_type, self.claude)

class RadiologyContextExtractor:
    """Specialized agent for extracting structured radiology context"""
    
    def __init__(self, orchestrator: MultiModelOrchestrator, prompt_manager: PromptManager):
        self.orchestrator = orchestrator
        self.prompt_manager = prompt_manager
        
    async def extract_context(self, case: ClinicalCase) -> RadiologyContext:
        """Extract structured radiology context from clinical case"""
        
        # Create a run name for LangSmith
        run_name = f"radiology_context_extraction_{case.case_id}"
        
        # Get dynamic prompt
        prompt_template = await self.prompt_manager.get_prompt("radiology_context_extractor")
        
        # Get optimal model for this task
        model = await self.orchestrator.get_model_for_task("radiology_extraction")
        
        # Create prompt with case data
        prompt = ChatPromptTemplate.from_template(prompt_template.template_text)
        
        # Setup output parser
        parser = PydanticOutputParser(pydantic_object=RadiologyContext)
        
        # Create chain with callbacks
        chain = prompt | model | parser
        
        try:
            # Use callbacks in the invoke
            result = await chain.ainvoke({
                "patient_age": case.patient_age,
                "patient_sex": case.patient_sex,
                "clinical_history": case.clinical_history,
                "imaging_modality": case.imaging_modality,
                "anatomical_region": case.anatomical_region,
                "image_description": case.image_description,
                "format_instructions": parser.get_format_instructions()
            }, config={"run_name": run_name, "callbacks": [langsmith_tracer]})
            
            logger.info("Radiology context extracted successfully", 
                       case_id=case.case_id,
                       anatomy_count=len(result.anatomy),
                       findings_count=len(result.associated_findings))
            
            return result
            
        except Exception as e:
            logger.error("Error extracting radiology context", 
                        case_id=case.case_id, error=str(e))
            
            # Return basic context as fallback
            return RadiologyContext(
                anatomy=[case.anatomical_region],
                imaging_modality=case.imaging_modality,
                clinical_context=case.clinical_history
            )

class LiteratureSearchAgent:
    """Advanced literature search with document processing"""
    
    def __init__(self, orchestrator: MultiModelOrchestrator, prompt_manager: PromptManager):
        self.orchestrator = orchestrator
        self.prompt_manager = prompt_manager
        self.search_tool = BraveSearch(
            api_key=settings.brave_search_api_key,
            search_kwargs={"count": 20}
        )
        
    async def search_literature(self, radiology_context: RadiologyContext, 
                              case: ClinicalCase) -> List[LiteratureMatch]:
        """Search for relevant medical literature"""
        
        run_name = f"literature_search_{case.case_id}"
        
        # Generate optimized search queries
        search_queries = await self._generate_search_queries(radiology_context, case)
        
        # Execute searches with rate limiting protection
        all_results = []
        for i, query in enumerate(search_queries):
            if i > 0:
                # Add delay between searches to avoid rate limiting
                await asyncio.sleep(1.0)
            results = await self._execute_search(query)
            all_results.extend(results)
        
        # Process and analyze documents
        processed_results = []
        # Limit to top 3 results to avoid timeout issues
        for i, result in enumerate(all_results[:3]):
            logger.info(f"Processing document {i+1}/3", url=result.get('url', 'N/A'))
            processed = await self._process_document(result, radiology_context)
            if processed:
                processed_results.append(processed)
        
        # Rank by relevance
        ranked_results = sorted(processed_results, 
                              key=lambda x: x.relevance_score, reverse=True)
        
        logger.info("Literature search completed",
                   case_id=case.case_id,
                   queries_executed=len(search_queries),
                   results_found=len(ranked_results))
        
        return ranked_results[:5]  # Return top 5
    
    async def _generate_search_queries(self, context: RadiologyContext, 
                                     case: ClinicalCase) -> List[str]:
        """Generate optimized search queries"""
        
        run_name = f"search_query_generation_{case.case_id}"
        
        prompt_template = await self.prompt_manager.get_prompt("search_query_generator")
        model = await self.orchestrator.get_model_for_task("search_optimization")
        
        prompt = ChatPromptTemplate.from_template(prompt_template.template_text)
        chain = prompt | model | StrOutputParser()
        
        result = await chain.ainvoke({
            "anatomy": ", ".join(context.anatomy),
            "imaging_modality": context.imaging_modality,
            "sequences": ", ".join(context.sequences),
            "enhancement_pattern": ", ".join(context.enhancement_pattern),
            "clinical_context": context.clinical_context
        }, config={"run_name": run_name, "callbacks": [langsmith_tracer]})
        
        # Parse queries from result
        queries = [q.strip() for q in result.split('\n') if q.strip()]
        return queries[:3]  # Limit to 3 queries to avoid rate limiting
    
    async def _execute_search(self, query: str) -> List[Dict]:
        """Execute search and return raw results"""
        try:
            logger.info(f"Executing search query: {query}")
            # Run synchronous search_tool.run() in a thread pool
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, self.search_tool.run, query)
            logger.info(f"Search returned results of type: {type(results)}, length: {len(str(results))}")
            
            # Parse search results
            parsed = self._parse_search_results(results)
            logger.info(f"Parsed {len(parsed)} results from search")
            
            return parsed
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg:
                logger.warning("Brave Search rate limit hit, skipping query", query=query)
            else:
                logger.error("Search execution failed", query=query, error=error_msg, exc_info=True)
            return []
    
    def _parse_search_results(self, results: str) -> List[Dict]:
        """Parse search results into structured format"""
        parsed_results = []
        
        # BraveSearch tool returns results as a formatted string
        # Each result is typically separated by newlines with title and snippet
        if not results:
            return parsed_results
            
        # Try to parse as JSON first (in case the tool returns JSON)
        try:
            import json
            data = json.loads(results)
            if isinstance(data, list):
                # The BraveSearch tool returns a list of dicts with 'title', 'link', 'snippet'
                for item in data:
                    parsed_results.append({
                        'title': item.get('title', ''),
                        'url': item.get('link', item.get('url', '')),
                        'description': item.get('snippet', item.get('description', ''))
                    })
                return parsed_results
            elif isinstance(data, dict) and 'results' in data:
                return data['results']
        except:
            pass
        
        # If not JSON, parse the string format
        # The BraveSearch tool typically returns: "Title: ... URL: ... Snippet: ..."
        # or sometimes just concatenated title and description
        sections = results.split('\n\n')  # Results might be separated by double newlines
        
        for section in sections:
            if not section.strip():
                continue
                
            result = {}
            lines = section.strip().split('\n')
            
            # Try to extract structured data
            for line in lines:
                if ': ' in line:
                    key, value = line.split(': ', 1)
                    key_lower = key.lower()
                    if 'title' in key_lower:
                        result['title'] = value.strip()
                    elif 'url' in key_lower or 'link' in key_lower:
                        result['url'] = value.strip()
                    elif 'description' in key_lower or 'snippet' in key_lower:
                        result['description'] = value.strip()
            
            # If we couldn't parse structured data, treat the whole section as a result
            if not result:
                # Take first line as title, rest as description
                if len(lines) > 0:
                    result['title'] = lines[0].strip()
                    result['description'] = ' '.join(lines[1:]).strip() if len(lines) > 1 else ''
                    # Generate a placeholder URL
                    result['url'] = f"https://search.result/{len(parsed_results) + 1}"
            
            if result and (result.get('title') or result.get('description')):
                parsed_results.append(result)
        
        # If still no results, try simple line-by-line parsing
        if not parsed_results and results.strip():
            lines = [line.strip() for line in results.split('\n') if line.strip()]
            for i, line in enumerate(lines):
                parsed_results.append({
                    'title': line[:100] + '...' if len(line) > 100 else line,
                    'description': line,
                    'url': f"https://search.result/{i + 1}"
                })
        
        return parsed_results[:10]  # Limit to 10 results
    
    async def _process_document(self, search_result: Dict, 
                              context: RadiologyContext) -> Optional[LiteratureMatch]:
        """Process individual document and extract relevant information"""
        
        try:
            # Load document content
            url = search_result.get('url', '')
            if not url:
                logger.warning("Search result has no URL, skipping", result=search_result)
                return None
                
            if url.endswith('.pdf'):
                loader = PyPDFLoader(url)
            else:
                loader = WebBaseLoader(url)
            
            try:
                loop = asyncio.get_event_loop()
                # Add timeout for loading documents
                docs = await asyncio.wait_for(
                    loop.run_in_executor(None, loader.load),
                    timeout=30.0  # 30 second timeout per document
                )
            except asyncio.TimeoutError:
                logger.warning("Document loading timed out", url=url)
                return None
            except Exception as e:
                logger.warning(f"Failed to load document: {e}", url=url)
                return None
            
            if not docs:
                return None
            
            # Extract text content
            full_text = '\n'.join([doc.page_content for doc in docs])
            
            # Analyze relevance using Mistral
            relevance_analysis = await self._analyze_relevance(full_text, context)
            
            if relevance_analysis['relevance_score'] < 0.3:
                return None
            
            # Extract image descriptions
            image_descriptions = await self._extract_image_descriptions(full_text)
            
            return LiteratureMatch(
                title=search_result.get('title', 'Unknown'),
                authors=[],  # Would extract from document
                journal='Unknown',  # Would extract from document
                year=2024,  # Would extract from document
                url=search_result.get('url', ''),
                abstract=search_result.get('description', ''),
                relevant_sections=relevance_analysis.get('relevant_sections', []),
                image_descriptions=image_descriptions,
                relevance_score=relevance_analysis['relevance_score'],
                match_reasoning=relevance_analysis.get('reasoning', '')
            )
            
        except Exception as e:
            logger.error("Document processing failed", 
                        url=search_result.get('url'), error=str(e))
            return None
    
    async def _analyze_relevance(self, document_text: str, 
                               context: RadiologyContext) -> Dict:
        """Analyze document relevance to radiology context"""
        
        run_name = f"relevance_analysis"
        
        prompt_template = await self.prompt_manager.get_prompt("relevance_analyzer")
        model = await self.orchestrator.get_model_for_task("literature_analysis")
        
        prompt = ChatPromptTemplate.from_template(prompt_template.template_text)
        parser = JsonOutputParser()
        
        chain = prompt | model | parser
        
        result = await chain.ainvoke({
            "document_text": document_text[:5000],  # Limit text length
            "anatomy": ", ".join(context.anatomy),
            "imaging_modality": context.imaging_modality,
            "clinical_context": context.clinical_context,
            "format_instructions": parser.get_format_instructions()
        }, config={"run_name": run_name, "callbacks": [langsmith_tracer]})
        
        return result
    
    async def _extract_image_descriptions(self, document_text: str) -> List[str]:
        """Extract image descriptions and captions from document text"""
        
        run_name = f"image_description_extraction"
        
        prompt_template = await self.prompt_manager.get_prompt("image_description_extractor")
        model = await self.orchestrator.get_model_for_task("document_processing")
        
        prompt = ChatPromptTemplate.from_template(prompt_template.template_text)
        chain = prompt | model | StrOutputParser()
        
        result = await chain.ainvoke({
            "document_text": document_text
        }, config={"run_name": run_name, "callbacks": [langsmith_tracer]})
        
        # Parse extracted descriptions
        descriptions = [desc.strip() for desc in result.split('\n') if desc.strip()]
        return descriptions

class DiagnosisAgent:
    """Advanced diagnosis generation with evidence synthesis"""
    
    def __init__(self, orchestrator: MultiModelOrchestrator, prompt_manager: PromptManager):
        self.orchestrator = orchestrator
        self.prompt_manager = prompt_manager
    
    async def generate_diagnosis(self, case: ClinicalCase, 
                               radiology_context: RadiologyContext,
                               literature_matches: List[LiteratureMatch]) -> Dict:
        """Generate comprehensive diagnosis with evidence"""
        
        # Prepare literature evidence
        literature_evidence = self._prepare_literature_evidence(literature_matches)
        
        # Generate primary diagnosis
        primary_diagnosis = await self._generate_primary_diagnosis(
            case, radiology_context, literature_evidence
        )
        
        # Generate differential diagnoses
        differential_diagnoses = await self._generate_differential_diagnoses(
            case, radiology_context, literature_evidence, primary_diagnosis
        )
        
        # Calculate confidence scores
        confidence_assessment = await self._assess_confidence(
            primary_diagnosis, differential_diagnoses, literature_evidence
        )
        
        return {
            "primary_diagnosis": primary_diagnosis,
            "differential_diagnoses": differential_diagnoses,
            "confidence_assessment": confidence_assessment,
            "literature_support": literature_evidence
        }
    
    def _prepare_literature_evidence(self, matches: List[LiteratureMatch]) -> str:
        """Prepare literature evidence for diagnosis"""
        evidence_sections = []
        
        for match in matches:
            evidence = f"Study: {match.title}\n"
            evidence += f"Relevance: {match.relevance_score:.2f}\n"
            evidence += f"Key findings: {match.match_reasoning}\n"
            
            if match.image_descriptions:
                evidence += f"Image descriptions: {'; '.join(match.image_descriptions[:3])}\n"
            
            evidence_sections.append(evidence)
        
        return "\n---\n".join(evidence_sections)
    
    async def _generate_primary_diagnosis(self, case: ClinicalCase,
                                        context: RadiologyContext,
                                        literature: str) -> Dict:
        """Generate primary diagnosis with reasoning"""
        
        run_name = f"primary_diagnosis_generation_{case.case_id}"
        
        prompt_template = await self.prompt_manager.get_prompt("primary_diagnosis_generator")
        model = await self.orchestrator.get_model_for_task("diagnosis_generation")
        
        prompt = ChatPromptTemplate.from_template(prompt_template.template_text)
        parser = JsonOutputParser()
        
        chain = prompt | model | parser
        
        result = await chain.ainvoke({
            "clinical_history": case.clinical_history,
            "image_description": case.image_description,
            "radiology_context": context.dict(),
            "literature_evidence": literature,
            "format_instructions": parser.get_format_instructions()
        }, config={"run_name": run_name, "callbacks": [langsmith_tracer]})
        
        return result
    
    async def _generate_differential_diagnoses(self, case: ClinicalCase,
                                             context: RadiologyContext,
                                             literature: str,
                                             primary: Dict) -> List[Dict]:
        """Generate differential diagnoses"""
        
        run_name = f"differential_diagnosis_generation_{case.case_id}"
        
        prompt_template = await self.prompt_manager.get_prompt("differential_diagnosis_generator")
        model = await self.orchestrator.get_model_for_task("diagnosis_generation")
        
        prompt = ChatPromptTemplate.from_template(prompt_template.template_text)
        parser = JsonOutputParser()
        
        chain = prompt | model | parser
        
        result = await chain.ainvoke({
            "clinical_history": case.clinical_history,
            "image_description": case.image_description,
            "radiology_context": context.dict(),
            "primary_diagnosis": primary,
            "literature_evidence": literature,
            "format_instructions": parser.get_format_instructions()
        }, config={"run_name": run_name, "callbacks": [langsmith_tracer]})
        
        return result.get("differential_diagnoses", [])
    
    async def _assess_confidence(self, primary: Dict, differentials: List[Dict], 
                               literature: str) -> Dict:
        """Assess overall confidence in diagnosis"""
        
        run_name = f"confidence_assessment"
        
        prompt_template = await self.prompt_manager.get_prompt("confidence_assessor")
        model = await self.orchestrator.get_model_for_task("diagnosis_generation")
        
        prompt = ChatPromptTemplate.from_template(prompt_template.template_text)
        parser = JsonOutputParser()
        
        chain = prompt | model | parser
        
        result = await chain.ainvoke({
            "primary_diagnosis": primary,
            "differential_diagnoses": differentials,
            "literature_support": literature,
            "format_instructions": parser.get_format_instructions()
        }, config={"run_name": run_name, "callbacks": [langsmith_tracer]})
        
        return result

class RadiologyAISystem:
    """Main system orchestrating all agents"""
    
    def __init__(self):
        # Initialize database connection
        self.db_client = None
        self.prompt_manager = None
        self.orchestrator = None
        self.context_extractor = None
        self.literature_agent = None
        self.diagnosis_agent = None
        
    async def initialize(self):
        """Initialize all system components"""
        
        # Connect to MongoDB
        self.db_client = AsyncIOMotorClient(settings.mongodb_url)[settings.database_name]
        
        # Initialize components
        self.prompt_manager = PromptManager(self.db_client)
        self.orchestrator = MultiModelOrchestrator(self.prompt_manager)
        self.context_extractor = RadiologyContextExtractor(self.orchestrator, self.prompt_manager)
        self.literature_agent = LiteratureSearchAgent(self.orchestrator, self.prompt_manager)
        self.diagnosis_agent = DiagnosisAgent(self.orchestrator, self.prompt_manager)
        
        # Initialize default prompts
        await self._initialize_default_prompts()
        
        logger.info("RadiologyAISystem initialized successfully")
    
    async def _initialize_default_prompts(self):
        """Initialize default prompt templates"""
        
        default_prompts = [
            PromptTemplate(
                template_id="radiology_context_extractor",
                name="Radiology Context Extractor",
                version="1.0",
                description="Extracts structured radiology context from clinical descriptions",
                template_text="""You are an expert radiologist analyzing clinical cases. Extract structured radiology context from the following case:

Patient: {patient_age} year old {patient_sex}
Clinical History: {clinical_history}
Imaging: {imaging_modality} of {anatomical_region}
Findings: {image_description}

Extract the following structured information:
{format_instructions}

Focus on:
- Anatomical structures mentioned
- Imaging sequences and protocols
- Measurements and dimensions
- Morphological characteristics
- Enhancement patterns
- Signal/density characteristics
- Associated findings
- Clinical context summary

Be precise and use standard radiological terminology.""",
                input_variables=["patient_age", "patient_sex", "clinical_history", "imaging_modality", "anatomical_region", "image_description", "format_instructions"],
                model_type="claude"
            ),
            
            PromptTemplate(
                template_id="search_query_generator",
                name="Search Query Generator",
                version="1.0",
                description="Generates optimized search queries for medical literature",
                template_text="""Generate 5 optimized search queries for finding relevant medical literature based on this radiology case:

Anatomy: {anatomy}
Imaging Modality: {imaging_modality}
Sequences: {sequences}
Enhancement Pattern: {enhancement_pattern}
Clinical Context: {clinical_context}

Create targeted search queries that will find:
1. Diagnostic imaging papers
2. Case reports with similar findings
3. Systematic reviews and meta-analyses
4. Recent research on this condition
5. Imaging technique papers

Format each query on a new line. Use medical terminology and include imaging modality.""",
                input_variables=["anatomy", "imaging_modality", "sequences", "enhancement_pattern", "clinical_context"],
                model_type="deepseek"
            ),
            
            PromptTemplate(
                template_id="relevance_analyzer",
                name="Document Relevance Analyzer",
                version="1.0",
                description="Analyzes document relevance to radiology case",
                template_text="""Analyze the relevance of this medical document to the radiology case:

Document Text (excerpt): {document_text}

Case Context:
- Anatomy: {anatomy}
- Imaging Modality: {imaging_modality}
- Clinical Context: {clinical_context}

Provide analysis in JSON format:
{format_instructions}

Include:
- relevance_score (0.0-1.0)
- reasoning (why relevant/not relevant)
- relevant_sections (list of relevant text excerpts)
- key_findings (important medical findings mentioned)""",
                input_variables=["document_text", "anatomy", "imaging_modality", "clinical_context", "format_instructions"],
                model_type="claude"
            ),
            
            PromptTemplate(
                template_id="image_description_extractor",
                name="Image Description Extractor",
                version="1.0",
                description="Extracts image descriptions and captions from medical documents",
                template_text="""Extract all image descriptions, figure captions, and imaging findings from this medical document:

Document Text: {document_text}

Find and extract:
- Figure captions (Figure 1, Fig. 2, etc.)
- Image descriptions in text
- Radiological findings descriptions
- Imaging technique descriptions

Return each description on a new line. Focus on descriptions that include:
- Imaging modality
- Anatomical structures
- Pathological findings
- Enhancement patterns
- Measurements""",
                input_variables=["document_text"],
                model_type="mistral"
            ),
            
            PromptTemplate(
                template_id="primary_diagnosis_generator",
                name="Primary Diagnosis Generator",
                version="1.0",
                description="Generates primary diagnosis with medical reasoning",
                template_text="""Generate the most likely primary diagnosis for this radiology case:

Clinical History: {clinical_history}
Imaging Findings: {image_description}
Radiology Context: {radiology_context}
Literature Evidence: {literature_evidence}

Provide diagnosis in JSON format:
{format_instructions}

Include:
- diagnosis (primary diagnosis name)
- confidence_score (0.0-1.0)
- reasoning (detailed medical reasoning)
- icd_code (if applicable)
- supporting_evidence (key evidence from literature)
- imaging_features (key imaging characteristics)

Use evidence-based medicine principles and current medical guidelines.""",
                input_variables=["clinical_history", "image_description", "radiology_context", "literature_evidence", "format_instructions"],
                model_type="claude"
            ),
            
            PromptTemplate(
                template_id="differential_diagnosis_generator",
                name="Differential Diagnosis Generator",
                version="1.0",
                description="Generates differential diagnoses with ranking",
                template_text="""Generate a ranked list of differential diagnoses for this case:

Clinical History: {clinical_history}
Imaging Findings: {image_description}
Radiology Context: {radiology_context}
Primary Diagnosis: {primary_diagnosis}
Literature Evidence: {literature_evidence}

Provide differential diagnoses in JSON format:
{format_instructions}

For each differential diagnosis include:
- diagnosis (name)
- probability (0.0-1.0)
- reasoning (why this diagnosis is considered)
- distinguishing_features (how to differentiate from primary)
- additional_workup (what tests would help confirm/exclude)

Rank by likelihood and clinical importance.""",
                input_variables=["clinical_history", "image_description", "radiology_context", "primary_diagnosis", "literature_evidence", "format_instructions"],
                model_type="claude"
            ),
            
            PromptTemplate(
                template_id="confidence_assessor",
                name="Confidence Assessor",
                version="1.0",
                description="Assesses confidence in diagnosis based on available evidence",
                template_text="""Assess the overall confidence in this diagnostic assessment:

Primary Diagnosis: {primary_diagnosis}
Differential Diagnoses: {differential_diagnoses}
Literature Support: {literature_support}

Provide confidence assessment in JSON format:
{format_instructions}

Include:
- overall_confidence (0.0-1.0)
- evidence_quality (assessment of literature quality)
- clinical_correlation (how well imaging matches clinical presentation)
- diagnostic_certainty (confidence in primary diagnosis)
- recommendation (next steps or additional workup needed)
- limitations (what factors limit diagnostic confidence)

Consider imaging quality, clinical correlation, and literature support.""",
                input_variables=["primary_diagnosis", "differential_diagnoses", "literature_support", "format_instructions"],
                model_type="claude"
            )
        ]
        
        # Save default prompts to database
        for prompt in default_prompts:
            try:
                await self.prompt_manager.save_prompt(prompt)
                logger.info(f"Initialized prompt: {prompt.template_id}")
            except Exception as e:
                logger.error(f"Failed to initialize prompt {prompt.template_id}: {e}")
    
    async def analyze_case(self, case: ClinicalCase) -> Dict:
        """Complete case analysis pipeline"""
        
        logger.info("Starting full case analysis", case_id=case.case_id)
        
        # Create main run for LangSmith
        run_name = f"full_case_analysis_{case.case_id}"
        
        try:
            # Step 1: Extract radiology context
            radiology_context = await self.context_extractor.extract_context(case)
            case.radiology_context = radiology_context
            
            # Step 2: Search literature
            try:
                literature_matches = await self.literature_agent.search_literature(
                    radiology_context, case
                )
            except Exception as e:
                logger.warning(f"Literature search failed: {e}, continuing without literature")
                literature_matches = []
            
            # Step 3: Generate diagnosis
            diagnosis_result = await self.diagnosis_agent.generate_diagnosis(
                case, radiology_context, literature_matches
            )
            
            # Step 4: Compile final result
            final_result = {
                "case_id": case.case_id,
                "radiology_context": radiology_context.dict(),
                "literature_matches": [match.dict() for match in literature_matches],
                "diagnosis_result": diagnosis_result,
                "processing_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "models_used": ["claude", "mistral", "deepseek"],
                    "literature_sources": len(literature_matches),
                    "langsmith_project": settings.langchain_project
                }
            }
            
            logger.info("Case analysis completed successfully",
                       case_id=case.case_id,
                       primary_diagnosis=diagnosis_result.get("primary_diagnosis", {}).get("diagnosis"),
                       literature_count=len(literature_matches))
            
            return final_result
            
        except Exception as e:
            logger.error("Case analysis failed", case_id=case.case_id, error=str(e))
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# Initialize the system
radiology_system = RadiologyAISystem()

# FastAPI Application
app = FastAPI(
    title="Radiology AI System - LangChain + LangSmith",
    description="Advanced multi-agent radiology case analysis with complete observability",
    version="2.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    await radiology_system.initialize()
    logger.info("Radiology AI System started successfully")

@app.get("/health")
async def health_check():
    """Health check with system status"""
    return {
        "status": "healthy",
        "system": "radiology-ai-langchain",
        "langsmith_enabled": settings.langchain_tracing_v2,
        "langsmith_project": settings.langchain_project,
        "models_available": ["claude", "mistral", "deepseek"],
        "database_connected": radiology_system.db_client is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/analyze-case")
async def analyze_radiology_case(case: ClinicalCase):
    """Analyze radiology case with complete multi-agent pipeline"""
    
    logger.info("Received case analysis request", 
               case_id=case.case_id,
               imaging_modality=case.imaging_modality,
               anatomical_region=case.anatomical_region)
    
    try:
        result = await radiology_system.analyze_case(case)
        return result
    except Exception as e:
        logger.error(f"Case analysis failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/api/prompts")
async def list_prompts():
    """List all available prompt templates"""
    collection = radiology_system.db_client.prompts
    prompts = await collection.find({}, {"template_text": 0}).to_list(length=100)
    
    # Convert ObjectId to string and datetime to ISO format for JSON serialization
    for prompt in prompts:
        if '_id' in prompt:
            prompt['_id'] = str(prompt['_id'])
        if 'created_at' in prompt and isinstance(prompt['created_at'], datetime):
            prompt['created_at'] = prompt['created_at'].isoformat()
        if 'updated_at' in prompt and isinstance(prompt['updated_at'], datetime):
            prompt['updated_at'] = prompt['updated_at'].isoformat()
    
    return {"prompts": prompts}

@app.get("/api/prompts/{template_id}")
async def get_prompt(template_id: str, version: Optional[str] = None):
    """Get specific prompt template"""
    prompt = await radiology_system.prompt_manager.get_prompt(template_id, version)
    return prompt.dict()

@app.put("/api/prompts/{template_id}")
async def update_prompt(template_id: str, new_template_text: str, description: str = "Updated"):
    """Create new version of prompt template"""
    new_prompt = await radiology_system.prompt_manager.create_prompt_version(
        template_id, new_template_text, description
    )
    return {"message": "Prompt updated", "new_version": new_prompt.version}

@app.get("/api/langsmith-dashboard")
async def get_langsmith_info():
    """Get LangSmith dashboard information"""
    return {
        "langsmith_url": f"https://smith.langchain.com/projects/{settings.langchain_project}",
        "project_name": settings.langchain_project,
        "tracing_enabled": settings.langchain_tracing_v2,
        "api_endpoint": settings.langchain_endpoint
    }

@app.get("/api/test-search")
async def test_search(query: str = "brain MRI glioma"):
    """Test search functionality directly"""
    try:
        # Test Brave Search directly
        search_tool = BraveSearch(
            api_key=settings.brave_search_api_key,
            search_kwargs={"count": 5}
        )
        
        results = search_tool.run(query)
        
        return {
            "query": query,
            "raw_results": results,
            "result_type": str(type(results)),
            "result_length": len(str(results)),
            "brave_api_key_present": bool(settings.brave_search_api_key),
            "brave_api_key_length": len(settings.brave_search_api_key)
        }
    except Exception as e:
        return {
            "error": str(e),
            "error_type": type(e).__name__,
            "query": query
        }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        timeout_keep_alive=180  # 3 minutes timeout for long-running AI analysis
    )
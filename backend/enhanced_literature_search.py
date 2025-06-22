"""
Enhanced Literature Search with Image Extraction
Searches for radiology papers with images and extracts relevant figures
"""

import asyncio
import re
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import aiohttp
from bs4 import BeautifulSoup
# PDF handling is done via LangChain's PyPDFLoader
import io
import base64
from urllib.parse import urljoin, urlparse
import structlog

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import BraveSearch
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI
from langchain_deepseek import ChatDeepSeek
from langchain.callbacks.tracers import LangChainTracer

from pydantic import BaseModel, Field

logger = structlog.get_logger()

class ExtractedImage(BaseModel):
    """Represents an extracted image from literature"""
    url: str = Field(..., description="Image URL or data URI")
    caption: str = Field(..., description="Image caption or description")
    figure_number: Optional[str] = Field(None, description="Figure number (e.g., 'Figure 1')")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance to the case")
    source_page: Optional[int] = Field(None, description="Page number in PDF")
    alt_text: Optional[str] = Field(None, description="Alternative text for accessibility")

class EnhancedLiteratureMatch(BaseModel):
    """Enhanced literature match with extracted images"""
    title: str = Field(..., description="Paper title")
    authors: List[str] = Field(default_factory=list, description="Paper authors")
    journal: str = Field(..., description="Journal name")
    year: int = Field(..., description="Publication year")
    doi: Optional[str] = Field(None, description="DOI if available")
    url: str = Field(..., description="Access URL")
    abstract: str = Field(..., description="Paper abstract")
    relevant_sections: List[str] = Field(default_factory=list, description="Relevant text sections")
    extracted_images: List[ExtractedImage] = Field(default_factory=list, description="Extracted images with captions")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    match_reasoning: str = Field(..., description="Why this paper is relevant")
    has_images: bool = Field(default=False, description="Whether the paper contains relevant images")

class EnhancedLiteratureSearchAgent:
    """Enhanced literature search agent with image extraction capabilities"""
    
    def __init__(self, deepseek_model, mistral_model, claude_model, brave_api_key: str, langsmith_tracer):
        self.deepseek = deepseek_model
        self.mistral = mistral_model
        self.claude = claude_model
        self.langsmith_tracer = langsmith_tracer
        self.search_tool = BraveSearch(api_key=brave_api_key, search_kwargs={"count": 20})
        
        # Enhanced search query generation prompt
        self.search_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert medical literature search specialist.
Generate targeted search queries to find radiology papers and resources with IMAGES.

Focus on:
1. Papers with imaging figures, radiological images, or diagnostic images
2. Case reports with visual findings
3. Radiology atlases and visual guides
4. Papers that explicitly mention figures or images

Include terms like: "with images", "figure", "radiographic findings", "imaging features", "case report with images"
"""),
            ("human", """Generate 5 search queries to find radiology literature WITH IMAGES for:
Anatomy: {anatomy}
Modality: {imaging_modality}
Key features: {features}
Clinical context: {clinical_context}

Format: One query per line, each targeting papers with visual content.""")
        ])
        
        # Image relevance scoring prompt
        self.image_relevance_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a radiologist evaluating if an image is relevant to a specific case.
Score the relevance of the image based on:
1. Matching anatomical location
2. Similar imaging modality
3. Similar pathological features
4. Educational value for the case

Provide a relevance score from 0.0 to 1.0."""),
            ("human", """Case details:
Anatomy: {anatomy}
Modality: {modality}
Features: {features}

Image caption: {caption}
Image context: {context}

Provide only a decimal score between 0.0 and 1.0.""")
        ])
        
    async def search_literature_with_images(self, radiology_context, case, max_papers: int = 10) -> List[EnhancedLiteratureMatch]:
        """Search for literature with relevant images"""
        try:
            run_name = f"enhanced_literature_search_{case.case_id}"
            
            # Generate targeted search queries
            queries = await self._generate_image_focused_queries(radiology_context, run_name)
            logger.info(f"Generated {len(queries)} image-focused search queries")
            
            # Execute searches in parallel
            search_tasks = [self._execute_search(query) for query in queries]
            search_results = await asyncio.gather(*search_tasks)
            
            # Flatten and deduplicate results
            all_results = []
            seen_urls = set()
            for results in search_results:
                for result in results:
                    if result.get('url') not in seen_urls:
                        seen_urls.add(result.get('url'))
                        all_results.append(result)
            
            logger.info(f"Found {len(all_results)} unique search results")
            
            # Filter for results likely to have images
            image_candidates = self._filter_image_candidates(all_results)
            logger.info(f"Identified {len(image_candidates)} candidates likely to have images")
            
            # Process top candidates with image extraction
            processed_results = []
            for i, result in enumerate(image_candidates[:max_papers * 2]):  # Process more to get max_papers good ones
                logger.info(f"Processing document {i+1}/{len(image_candidates[:max_papers * 2])}", url=result.get('url', 'N/A'))
                processed = await self._process_document_with_images(result, radiology_context, run_name)
                if processed and processed.has_images:
                    processed_results.append(processed)
                    if len(processed_results) >= max_papers:
                        break
            
            # Rank by relevance and image quality
            ranked_results = sorted(processed_results, 
                                  key=lambda x: (x.relevance_score * 0.7 + 
                                               (0.3 * len(x.extracted_images) / 10)), 
                                  reverse=True)
            
            return ranked_results[:max_papers]
            
        except Exception as e:
            logger.error("Enhanced literature search failed", error=str(e), exc_info=True)
            return []
    
    async def _generate_image_focused_queries(self, context, run_name: str) -> List[str]:
        """Generate search queries focused on finding papers with images"""
        chain = self.search_prompt | self.deepseek | StrOutputParser()
        
        # Extract key features for search
        features = []
        if context.morphology:
            features.extend(context.morphology)
        if context.enhancement_pattern:
            features.extend(context.enhancement_pattern)
        if context.signal_characteristics:
            features.extend(context.signal_characteristics)
        
        result = await chain.ainvoke({
            "anatomy": ", ".join(context.anatomy),
            "imaging_modality": context.imaging_modality,
            "features": ", ".join(features),
            "clinical_context": context.clinical_context
        }, config={"run_name": f"{run_name}_query_generation", "callbacks": [self.langsmith_tracer]})
        
        queries = [q.strip() for q in result.split('\n') if q.strip()]
        return queries[:5]  # Use 5 queries for better coverage
    
    def _filter_image_candidates(self, results: List[Dict]) -> List[Dict]:
        """Filter results likely to contain images"""
        image_keywords = [
            'figure', 'image', 'imaging', 'radiograph', 'scan', 'mri', 'ct', 
            'case report', 'atlas', 'visual', 'illustration', 'diagram',
            'fig.', 'panel', 'arrow', 'demonstrates', 'shows', 'reveals'
        ]
        
        filtered = []
        for result in results:
            title = result.get('title', '').lower()
            snippet = result.get('snippet', '').lower()
            url = result.get('url', '').lower()
            
            # Check for image indicators
            has_image_keyword = any(keyword in title + snippet for keyword in image_keywords)
            is_pdf = url.endswith('.pdf')
            is_pmc = 'ncbi.nlm.nih.gov/pmc' in url
            is_radiopaedia = 'radiopaedia.org' in url
            
            # Prioritize sources known to have images
            if has_image_keyword or is_pmc or is_radiopaedia or is_pdf:
                result['image_likelihood'] = 1.0 if (is_pmc or is_radiopaedia) else 0.7
                filtered.append(result)
        
        # Sort by image likelihood
        return sorted(filtered, key=lambda x: x.get('image_likelihood', 0), reverse=True)
    
    async def _process_document_with_images(self, result: Dict, context, run_name: str) -> Optional[EnhancedLiteratureMatch]:
        """Process document and extract images"""
        try:
            url = result.get('url', '')
            logger.info(f"Processing document with image extraction: {url}")
            
            # Determine document type and load accordingly
            if url.endswith('.pdf'):
                content, images = await self._extract_from_pdf(url)
            else:
                content, images = await self._extract_from_webpage(url)
            
            if not content:
                return None
            
            # Analyze relevance with Claude
            relevance_score, reasoning = await self._analyze_relevance(content, context, run_name)
            
            if relevance_score < 0.3:  # Skip low relevance
                return None
            
            # Score and filter images based on relevance
            relevant_images = await self._score_images(images, context, run_name)
            
            # Extract metadata
            metadata = self._extract_metadata(content, result)
            
            return EnhancedLiteratureMatch(
                title=metadata.get('title', result.get('title', 'Unknown Title')),
                authors=metadata.get('authors', []),
                journal=metadata.get('journal', 'Unknown Journal'),
                year=metadata.get('year', datetime.now().year),
                doi=metadata.get('doi'),
                url=url,
                abstract=metadata.get('abstract', result.get('snippet', '')),
                relevant_sections=self._extract_relevant_sections(content, context),
                extracted_images=relevant_images,
                relevance_score=relevance_score,
                match_reasoning=reasoning,
                has_images=len(relevant_images) > 0
            )
            
        except Exception as e:
            logger.error(f"Failed to process document: {e}", url=result.get('url'))
            return None
    
    async def _extract_from_webpage(self, url: str) -> Tuple[str, List[Dict]]:
        """Extract content and images from webpage"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Extract text content
                    content = soup.get_text(separator=' ', strip=True)
                    
                    # Extract images with captions
                    images = []
                    
                    # Look for figure elements
                    for figure in soup.find_all('figure'):
                        img = figure.find('img')
                        if img and img.get('src'):
                            caption = figure.find('figcaption')
                            images.append({
                                'url': urljoin(url, img['src']),
                                'caption': caption.get_text(strip=True) if caption else img.get('alt', ''),
                                'figure_number': self._extract_figure_number(caption.get_text() if caption else ''),
                                'alt_text': img.get('alt', '')
                            })
                    
                    # Look for images with nearby captions
                    for img in soup.find_all('img'):
                        if not any(img['src'] == i['url'] for i in images if 'url' in i):
                            # Look for caption in parent or sibling elements
                            parent = img.parent
                            caption_text = ''
                            
                            # Check for caption in parent
                            if parent and parent.name in ['div', 'p', 'td']:
                                caption_text = parent.get_text(strip=True)
                            
                            # Check next sibling
                            next_sibling = img.find_next_sibling()
                            if next_sibling and next_sibling.name in ['p', 'div', 'span']:
                                caption_text = next_sibling.get_text(strip=True)
                            
                            if img.get('src'):
                                images.append({
                                    'url': urljoin(url, img['src']),
                                    'caption': caption_text or img.get('alt', ''),
                                    'figure_number': self._extract_figure_number(caption_text),
                                    'alt_text': img.get('alt', '')
                                })
                    
                    return content[:50000], images  # Limit content size
                    
        except Exception as e:
            logger.error(f"Failed to extract from webpage: {e}")
            return "", []
    
    async def _extract_from_pdf(self, url: str) -> Tuple[str, List[Dict]]:
        """Extract content and image references from PDF"""
        try:
            # For PDFs, we'll extract text and look for figure references
            # Actual image extraction would require more complex PDF processing
            loader = PyPDFLoader(url)
            pages = await asyncio.get_event_loop().run_in_executor(None, loader.load)
            
            content = " ".join([page.page_content for page in pages])
            
            # Extract figure references
            images = []
            figure_pattern = r'(Figure|Fig\.?)\s*(\d+[A-Za-z]?):?\s*([^.]+\.)'
            
            for match in re.finditer(figure_pattern, content):
                figure_num = f"{match.group(1)} {match.group(2)}"
                caption = match.group(3).strip()
                
                images.append({
                    'url': f"pdf:{url}#figure{match.group(2)}",  # Placeholder for PDF figures
                    'caption': caption,
                    'figure_number': figure_num,
                    'alt_text': caption
                })
            
            return content[:50000], images
            
        except Exception as e:
            logger.error(f"Failed to extract from PDF: {e}")
            return "", []
    
    def _extract_figure_number(self, text: str) -> Optional[str]:
        """Extract figure number from text"""
        if not text:
            return None
        
        patterns = [
            r'(Figure|Fig\.?)\s*(\d+[A-Za-z]?)',
            r'(Panel|Image)\s*([A-Za-z\d]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return f"{match.group(1)} {match.group(2)}"
        
        return None
    
    async def _score_images(self, images: List[Dict], context, run_name: str) -> List[ExtractedImage]:
        """Score images for relevance"""
        scored_images = []
        
        for img in images:
            # Skip placeholder or invalid images
            if not img.get('url') or img['url'].startswith('pdf:'):
                continue
            
            # Score relevance based on caption
            chain = self.image_relevance_prompt | self.claude | StrOutputParser()
            
            try:
                score_str = await chain.ainvoke({
                    "anatomy": ", ".join(context.anatomy),
                    "modality": context.imaging_modality,
                    "features": ", ".join(context.morphology + context.enhancement_pattern),
                    "caption": img.get('caption', ''),
                    "context": img.get('alt_text', '')
                }, config={"run_name": f"{run_name}_image_scoring", "callbacks": [self.langsmith_tracer]})
                
                relevance_score = float(score_str.strip())
                
                if relevance_score > 0.5:  # Only keep relevant images
                    scored_images.append(ExtractedImage(
                        url=img['url'],
                        caption=img.get('caption', ''),
                        figure_number=img.get('figure_number'),
                        relevance_score=relevance_score,
                        alt_text=img.get('alt_text')
                    ))
                    
            except Exception as e:
                logger.error(f"Failed to score image: {e}")
                continue
        
        # Sort by relevance and return top images
        return sorted(scored_images, key=lambda x: x.relevance_score, reverse=True)[:5]
    
    async def _analyze_relevance(self, content: str, context, run_name: str) -> Tuple[float, str]:
        """Analyze document relevance"""
        relevance_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a radiologist evaluating medical literature relevance.
Score the relevance based on:
1. Matching anatomy and imaging modality
2. Similar pathological features
3. Diagnostic value
4. Presence of relevant images or figures

Provide:
1. A relevance score (0.0-1.0)
2. Brief reasoning (one sentence)

Format: SCORE|REASONING"""),
            ("human", """Case context:
Anatomy: {anatomy}
Modality: {modality}
Features: {features}

Document excerpt:
{content}

Evaluate relevance.""")
        ])
        
        chain = relevance_prompt | self.claude | StrOutputParser()
        
        result = await chain.ainvoke({
            "anatomy": ", ".join(context.anatomy),
            "modality": context.imaging_modality,
            "features": ", ".join(context.morphology + context.enhancement_pattern),
            "content": content[:5000]  # Limit content
        }, config={"run_name": f"{run_name}_relevance_analysis", "callbacks": [self.langsmith_tracer]})
        
        parts = result.strip().split('|')
        if len(parts) == 2:
            try:
                score = float(parts[0])
                reasoning = parts[1]
                return score, reasoning
            except:
                pass
        
        return 0.5, "Unable to determine relevance"
    
    def _extract_metadata(self, content: str, result: Dict) -> Dict:
        """Extract paper metadata from content"""
        metadata = {
            'title': result.get('title', 'Unknown Title'),
            'authors': [],
            'journal': 'Unknown Journal',
            'year': datetime.now().year,
            'doi': None,
            'abstract': result.get('snippet', '')
        }
        
        # Extract DOI
        doi_match = re.search(r'10\.\d{4,9}/[-._;()/:\w]+', content)
        if doi_match:
            metadata['doi'] = doi_match.group(0)
        
        # Extract year
        year_match = re.search(r'(19|20)\d{2}', content[:2000])  # Look in beginning
        if year_match:
            metadata['year'] = int(year_match.group(0))
        
        return metadata
    
    def _extract_relevant_sections(self, content: str, context) -> List[str]:
        """Extract relevant text sections"""
        sections = []
        
        # Split content into paragraphs
        paragraphs = content.split('\n\n')
        
        # Keywords to look for
        keywords = (context.anatomy + context.morphology + 
                   context.enhancement_pattern + [context.imaging_modality])
        
        for para in paragraphs:
            if len(para) > 100 and len(sections) < 5:  # Limit sections
                para_lower = para.lower()
                if any(keyword.lower() in para_lower for keyword in keywords):
                    sections.append(para[:500])  # Limit section length
        
        return sections
    
    async def _execute_search(self, query: str) -> List[Dict]:
        """Execute search and return raw results"""
        try:
            logger.info(f"Executing search query: {query}")
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, self.search_tool.run, query)
            
            # Parse results
            parsed = []
            if isinstance(results, str):
                # Try to parse as JSON
                try:
                    import json
                    data = json.loads(results)
                    if isinstance(data, list):
                        parsed = data
                except:
                    # Parse as text
                    lines = results.split('\n')
                    for i in range(0, len(lines), 3):
                        if i + 2 < len(lines):
                            parsed.append({
                                'title': lines[i].strip(),
                                'url': lines[i + 1].strip(),
                                'snippet': lines[i + 2].strip()
                            })
            
            return parsed
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
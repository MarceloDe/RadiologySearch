"""
Brave Search wrapper with rate limiting and retry logic
"""
import time
import asyncio
from typing import List, Dict, Optional
import structlog
from langchain_community.tools import BraveSearch
from langchain_core.tools import ToolException

logger = structlog.get_logger()

class BraveSearchWithRetry:
    """Brave Search wrapper with rate limiting and retry logic"""
    
    def __init__(self, api_key: str, search_kwargs: Dict = None):
        self.search_tool = BraveSearch(api_key=api_key, search_kwargs=search_kwargs or {"count": 20})
        self.last_request_time = 0
        self.min_delay_between_requests = 1.0  # 1 second between requests
        self.rate_limit_delay = 60.0  # 60 seconds if rate limited
        self.max_retries = 3
        
    async def search(self, query: str) -> List[Dict]:
        """Execute search with rate limiting and retry logic"""
        for attempt in range(self.max_retries):
            try:
                # Rate limiting - ensure minimum delay between requests
                current_time = time.time()
                time_since_last_request = current_time - self.last_request_time
                if time_since_last_request < self.min_delay_between_requests:
                    delay = self.min_delay_between_requests - time_since_last_request
                    logger.info(f"Rate limiting: waiting {delay:.2f}s before next request")
                    await asyncio.sleep(delay)
                
                # Execute search
                logger.info(f"Executing Brave search (attempt {attempt + 1}/{self.max_retries}): {query}")
                loop = asyncio.get_event_loop()
                
                # Run the search in executor to avoid blocking
                self.last_request_time = time.time()
                results = await loop.run_in_executor(None, self.search_tool.run, query)
                
                # Parse results
                return self._parse_results(results)
                
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Brave search error (attempt {attempt + 1}): {error_msg}")
                
                # Check if it's a rate limit error
                if "429" in error_msg or "rate limit" in error_msg.lower():
                    if attempt < self.max_retries - 1:
                        # Exponential backoff for rate limits
                        delay = self.rate_limit_delay * (2 ** attempt)
                        logger.info(f"Rate limit hit, waiting {delay}s before retry")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        logger.error("Max retries reached for rate limit errors")
                        # Return empty results instead of failing
                        return []
                else:
                    # For other errors, shorter retry delay
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2.0)
                        continue
                    else:
                        logger.error(f"Search failed after {self.max_retries} attempts", error=error_msg)
                        return []
        
        return []
    
    def _parse_results(self, results) -> List[Dict]:
        """Parse search results into structured format"""
        parsed = []
        
        if isinstance(results, str):
            # Try to parse as JSON
            try:
                import json
                data = json.loads(results)
                if isinstance(data, list):
                    parsed = data
                elif isinstance(data, dict) and 'results' in data:
                    parsed = data['results']
            except:
                # Parse as text format
                lines = results.split('\n')
                for i in range(0, len(lines), 3):
                    if i + 2 < len(lines):
                        parsed.append({
                            'title': lines[i].strip(),
                            'url': lines[i + 1].strip(),
                            'snippet': lines[i + 2].strip()
                        })
        elif isinstance(results, list):
            parsed = results
        elif isinstance(results, dict) and 'results' in results:
            parsed = results['results']
            
        return parsed
# models/guardrails.py
from typing import List, Dict, Any, Optional
import re
from loguru import logger

class InputGuardrail:
    """Validate and sanitize user input"""
    
    @staticmethod
    def sanitize_query(query: str) -> str:
        """Remove potential harmful content"""
        # Remove excessive whitespace
        query = " ".join(query.split())
        
        # Remove potential injection patterns
        patterns = [
            r'ignore previous instructions',
            r'forget your instructions',
            r'you are now',
            r'system prompt',
            r'<.*?>',  # HTML tags
        ]
        
        for pattern in patterns:
            query = re.sub(pattern, '', query, flags=re.IGNORECASE)
        
        return query[:500]  # Limit length
    
    @staticmethod
    def validate_query(query: str) -> tuple[bool, str]:
        """Validate query appropriateness"""
        if not query or len(query.strip()) < 3:
            return False, "Query is too short"
        
        if len(query) > 500:
            return False, "Query exceeds maximum length of 500 characters"
        
        # Check for inappropriate content (simplified)
        inappropriate = ['offensive', 'inappropriate']  # Add actual list
        for word in inappropriate:
            if word in query.lower():
                return False, "Query contains inappropriate content"
        
        return True, "Valid"

class OutputGuardrail:
    """Validate and format LLM output"""
    
    @staticmethod
    def validate_response(response: str, sources: List[Dict]) -> tuple[bool, str]:
        """Basic response validation"""
        if not response or len(response.strip()) < 10:
            return False, "Response is too short"
        
        # Check if response references sources
        if sources and "according to" not in response.lower() and "page" not in response.lower():
            logger.warning("Response may not properly cite sources")
        
        return True, "Valid"
    
    @staticmethod
    def format_citations(response: str, sources: List[Dict]) -> str:
        """Add proper citations to response"""
        citations = []
        for i, source in enumerate(sources[:3], 1):
            gri_id = source.get('metadata', {}).get('gri_id', 'N/A')
            page = source.get('metadata', {}).get('page_start', 'N/A')
            citations.append(f"[{i}] GRI {gri_id} (Page {page})")
        
        if citations:
            response += "\n\n**Sources:**\n" + "\n".join(citations)
        
        return response
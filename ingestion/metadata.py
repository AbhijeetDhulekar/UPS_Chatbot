# ingestion/metadata.py
from typing import Dict, List
import re

class MetadataEnricher:
    """Add rich metadata to chunks"""
    
    # Category mapping based on keywords
    CATEGORY_KEYWORDS = {
        "Environmental": [
            "emission", "ghg", "carbon", "climate", "energy", "water", 
            "waste", "environmental", "305-", "306-"
        ],
        "Social": [
            "employee", "workforce", "labor", "human rights", "safety",
            "diversity", "community", "401-", "403-", "404-", "405-"
        ],
        "Governance": [
            "board", "governance", "compliance", "ethics", "risk",
            "audit", "oversight", "102-", "103-"
        ]
    }
    
    @classmethod
    def enrich(cls, chunks: List[Dict]) -> List[Dict]:
        """Add metadata to each chunk"""
        enriched_chunks = []
        
        for chunk in chunks:
            metadata = chunk["metadata"].copy()
            
            # Add document info
            metadata["doc_version"] = "2024"
            metadata["doc_name"] = "UPS-GRI-Report-2024"
            
            # Add category
            category = cls._determine_category(chunk["text"], metadata)
            metadata["category"] = category
            
            # Add page range
            metadata["page_range"] = f"{metadata['page_start']}-{metadata['page_end']}"
            
            # Add unique ID
            import hashlib
            chunk_id = hashlib.md5(
                f"{metadata['gri_id']}_{metadata['page_start']}_{metadata['chunk_index']}".encode()
            ).hexdigest()[:8]
            metadata["chunk_id"] = chunk_id
            
            enriched_chunks.append({
                "text": chunk["text"],
                "metadata": metadata
            })
        
        return enriched_chunks
    
    @classmethod
    def _determine_category(cls, text: str, metadata: Dict) -> str:
        """Determine category based on content and GRI ID"""
        text_lower = text.lower()
        gri_id = metadata.get("gri_id", "")
        
        # Check by GRI ID prefix
        if gri_id:
            if gri_id.startswith(("305", "306", "301")):
                return "Environmental"
            elif gri_id.startswith(("401", "403", "404", "405")):
                return "Social"
            elif gri_id.startswith(("102", "103")):
                return "Governance"
        
        # Check by keywords
        for category, keywords in cls.CATEGORY_KEYWORDS.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        
        return "General"
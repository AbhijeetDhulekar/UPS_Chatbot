# ingestion/chunker.py
import re
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
from config import Config
from debug.debugger import debugger

# Download NLTK data
nltk.download('punkt', quiet=True)

class HybridChunker:
    """Hybrid chunking: Hierarchical + Semantic"""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.gri_pattern = r'\b(\d{1,3}-\d{1,2}[A-Z]?)\b'
    
    @debugger.timer
    def extract_hierarchical_sections(self, text_blocks: List[Dict]) -> List[Dict]:
        """Level 1: Split by GRI IDs and headers"""
        sections = []
        current_section = {
            "gri_id": None,
            "header": "Introduction",
            "content": [],
            "page_start": None,
            "page_end": None
        }
        
        for block in text_blocks:
            text = block["text"]
            page = block["page"]
            
            # Check for GRI ID
            gri_match = re.search(self.gri_pattern, text)
            is_header = block.get("is_header", False)
            
            # New section if we find a GRI header
            if (gri_match and is_header) or (is_header and "GRI" in text):
                if current_section["content"]:
                    sections.append(current_section)
                
                current_section = {
                    "gri_id": gri_match.group(1) if gri_match else None,
                    "header": text.strip(),
                    "content": [text],
                    "page_start": page,
                    "page_end": page
                }
            else:
                current_section["content"].append(text)
                current_section["page_end"] = page
        
        # Add last section
        if current_section["content"]:
            sections.append(current_section)
        
        debugger.log("HIERARCHICAL", {"sections": len(sections)})
        return sections
    
    def semantic_chunking(self, text: str) -> List[str]:
        """Level 2: Split by topic boundaries using embeddings"""
        sentences = sent_tokenize(text)
        
        if len(sentences) <= 5:
            return [text]
        
        # Get embeddings
        embeddings = self.embedding_model.encode(sentences)
        
        chunks = []
        current_chunk = []
        current_embeddings = []
        
        for i, (sent, emb) in enumerate(zip(sentences, embeddings)):
            current_chunk.append(sent)
            current_embeddings.append(emb)
            
            # Check for topic boundary
            if len(current_chunk) >= 3:
                chunk_emb = np.mean(current_embeddings, axis=0)
                
                if i < len(sentences) - 1:
                    next_emb = embeddings[i + 1]
                    similarity = cosine_similarity([chunk_emb], [next_emb])[0][0]
                    
                    if similarity < Config.SEMANTIC_THRESHOLD:
                        chunks.append(" ".join(current_chunk))
                        current_chunk = []
                        current_embeddings = []
        
        # Add remaining
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    @debugger.timer
    def process_document(self, text_blocks: List[Dict]) -> List[Dict]:
        """Complete hybrid chunking pipeline"""
        final_chunks = []
        
        # Level 1: Hierarchical
        sections = self.extract_hierarchical_sections(text_blocks)
        
        for section in sections:
            section_text = " ".join(section["content"])
            
            # Level 2: Semantic chunking if needed
            if len(section_text) > Config.MAX_CHUNK_SIZE:
                semantic_chunks = self.semantic_chunking(section_text)
                
                for i, chunk_text in enumerate(semantic_chunks):
                    final_chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            "gri_id": section["gri_id"],
                            "section_header": section["header"],
                            "page_start": section["page_start"],
                            "page_end": section["page_end"],
                            "chunk_index": i,
                            "total_chunks": len(semantic_chunks)
                        }
                    })
            else:
                final_chunks.append({
                    "text": section_text,
                    "metadata": {
                        "gri_id": section["gri_id"],
                        "section_header": section["header"],
                        "page_start": section["page_start"],
                        "page_end": section["page_end"],
                        "chunk_index": 0,
                        "total_chunks": 1
                    }
                })
        
        debugger.log("CHUNKING", {"total_chunks": len(final_chunks)})
        return final_chunks
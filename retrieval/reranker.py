# retrieval/reranker.py
from sentence_transformers import CrossEncoder
from typing import List, Dict
from config import Config
from debug.debugger import debugger

class Reranker:
    """Cross-encoder reranking"""
    
    def __init__(self):
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    @debugger.timer
    def rerank(self, query: str, candidates: List[Dict], k: int = Config.TOP_K_FINAL) -> List[Dict]:
        """Rerank candidates using cross-encoder"""
        if not candidates:
            return []
        
        # Prepare pairs
        pairs = [[query, candidate["text"]] for candidate in candidates]
        
        # Get scores
        scores = self.model.predict(pairs)
        
        # Add scores to candidates
        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)
        
        # Sort by score
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        
        debugger.log("RERANKING", {
            "input_candidates": len(candidates),
            "output_candidates": min(len(reranked), k),
            "top_score": reranked[0]["rerank_score"] if reranked else None
        })
        
        return reranked[:k]
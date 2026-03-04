# retrieval/hybrid_search.py
"""
Hybrid Search using FAISS + BM25
Complete FAISS version
"""

import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any
from config import Config
from debug.debugger import debugger

class HybridSearch:
    """Hybrid search combining FAISS vector search and BM25"""
    
    def __init__(self, vector_store, all_chunks: List[Dict]):
        """
        Initialize hybrid search.
        
        Args:
            vector_store: FAISSVectorStore instance
            all_chunks: List of all chunks for BM25
        """
        self.vector_store = vector_store
        self.all_chunks = all_chunks
        
        # Prepare BM25
        self.corpus = [chunk["text"] for chunk in all_chunks]
        self.tokenized_corpus = [self._tokenize(text) for text in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        debugger.log("HYBRID", {
            "bm25_corpus_size": len(self.corpus),
            "vector_store_type": type(vector_store).__name__
        })
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        return text.lower().split()
    
    @debugger.timer
    def search(self, query: str, k: int = Config.TOP_K_INITIAL) -> List[Dict]:
        """
        Perform hybrid search using FAISS and BM25.
        
        Args:
            query: User question
            k: Number of results to return
            
        Returns:
            List of chunks with combined scores
        """
        debugger.log("HYBRID_SEARCH_START", {
            "query": query[:100],
            "k": k,
            "total_chunks": len(self.all_chunks)
        })
        
        try:
            # Vector search with FAISS
            vector_results = self.vector_store.search(query, k=k)
            debugger.log("VECTOR_SEARCH", {
                "results": len(vector_results),
                "top_score": vector_results[0]["score"] if vector_results else 0
            })
            
            # BM25 search
            tokenized_query = self._tokenize(query)
            bm25_scores = self.bm25.get_scores(tokenized_query)
            top_bm25_indices = np.argsort(bm25_scores)[-k:][::-1]
            
            debugger.log("BM25_SEARCH", {
                "top_scores": [float(bm25_scores[i]) for i in top_bm25_indices[:3]]
            })
            
            # RRF (Reciprocal Rank Fusion) to combine scores
            combined_scores = {}
            
            # Add vector results (RRF)
            for rank, result in enumerate(vector_results):
                # Use text hash as key
                text_key = hash(result["text"][:200])
                score = 1 / (rank + 61)  # RRF constant
                
                if text_key in combined_scores:
                    combined_scores[text_key]["score"] += score
                else:
                    combined_scores[text_key] = {
                        "score": score,
                        "chunk": result
                    }
            
            # Add BM25 results
            for rank, idx in enumerate(top_bm25_indices):
                if idx < len(self.all_chunks):
                    chunk = self.all_chunks[idx]
                    text_key = hash(chunk["text"][:200])
                    score = 1 / (rank + 61)
                    
                    if text_key in combined_scores:
                        combined_scores[text_key]["score"] += score
                    else:
                        combined_scores[text_key] = {
                            "score": score,
                            "chunk": chunk
                        }
            
            # Sort by combined score
            sorted_results = sorted(
                combined_scores.values(),
                key=lambda x: x["score"],
                reverse=True
            )[:k]
            
            # Extract chunks
            results = [item["chunk"] for item in sorted_results]
            
            debugger.log("HYBRID_SEARCH_END", {
                "final_results": len(results),
                "top_score": results[0].get("score", 0) if results else 0
            })
            
            return results
            
        except Exception as e:
            debugger.log("HYBRID_SEARCH_ERROR", {
                "error": str(e),
                "query": query[:100]
            }, level="ERROR")
            
            # Fallback to vector-only search
            debugger.log("HYBRID_FALLBACK", "Using vector-only search", level="WARNING")
            return self.vector_store.search(query, k=k)
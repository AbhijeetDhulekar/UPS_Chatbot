# memory/redis_cache.py
import redis
import json
import hashlib
import numpy as np
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from config import Config
from debug.debugger import debugger

class SemanticCache:
    """
    Long-term semantic cache for storing and retrieving query results.
    Supports exact-match and semantic-similarity lookups.
    """

    def __init__(self):
        """Initialize Redis connection and embedding model."""
        self.redis_client = redis.Redis(
            host=Config.REDIS_HOST,
            port=Config.REDIS_PORT,
            db=Config.REDIS_DB + 1,  # Use a separate DB for cache
            decode_responses=True
        )
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.similarity_threshold = 0.95  # Cosine similarity threshold for semantic matches
        self.ttl = Config.CACHE_TTL

    def _compute_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for text."""
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()

    def _compute_hash(self, text: str) -> str:
        """Generate MD5 hash for exact matching."""
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached result for a query if a semantically similar query exists.

        Args:
            query: User question.

        Returns:
            Cached result dictionary or None if not found.
        """
        query_emb = self._compute_embedding(query)

        # 1. Check exact match first (fast)
        exact_key = f"cache:exact:{self._compute_hash(query)}"
        exact_result = self.redis_client.get(exact_key)
        if exact_result:
            debugger.log("CACHE", {"type": "exact_hit", "query": query[:50]})
            return json.loads(exact_result)

        # 2. Semantic search over recent cached entries (simplified)
        # In production, consider using Redisearch with vector similarity.
        recent_keys = self.redis_client.keys("cache:semantic:*")
        best_similarity = 0.0
        best_result = None

        # Limit to last 100 keys for performance
        for key in recent_keys[-100:]:
            cached_data_json = self.redis_client.get(key)
            if not cached_data_json:
                continue

            cached_data = json.loads(cached_data_json)
            cached_emb = cached_data.get("embedding")
            if cached_emb:
                # Compute cosine similarity
                similarity = np.dot(query_emb, cached_emb) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(cached_emb) + 1e-9
                )
                if similarity > best_similarity and similarity > self.similarity_threshold:
                    best_similarity = similarity
                    best_result = cached_data.get("result")

        if best_result:
            debugger.log("CACHE", {"type": "semantic_hit", "similarity": best_similarity})
            return best_result

        debugger.log("CACHE", {"type": "miss", "query": query[:50]})
        return None

    def set(self, query: str, result: Dict[str, Any]):
        """
        Store query result in cache.

        Args:
            query: User question.
            result: Result dictionary to cache.
        """
        # Store exact match
        exact_key = f"cache:exact:{self._compute_hash(query)}"
        self.redis_client.setex(
            exact_key,
            timedelta(seconds=self.ttl),
            json.dumps(result)
        )

        # Store semantic version with embedding
        semantic_key = f"cache:semantic:{self._compute_hash(query)}"
        semantic_data = {
            "query": query,
            "embedding": self._compute_embedding(query),
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        self.redis_client.setex(
            semantic_key,
            timedelta(seconds=self.ttl),
            json.dumps(semantic_data)
        )

        debugger.log("CACHE", {"action": "set", "query": query[:50]})

    def invalidate(self, query: str):
        """Remove a specific query from cache."""
        exact_key = f"cache:exact:{self._compute_hash(query)}"
        semantic_key = f"cache:semantic:{self._compute_hash(query)}"
        self.redis_client.delete(exact_key)
        self.redis_client.delete(semantic_key)
        debugger.log("CACHE", {"action": "invalidate", "query": query[:50]})
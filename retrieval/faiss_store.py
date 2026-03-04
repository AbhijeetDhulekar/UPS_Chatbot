# retrieval/faiss_store.py
"""
FAISS Vector Store Implementation
Complete end-to-end vector store for RAG systems
"""

import faiss
import numpy as np
import pickle
import json
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from sentence_transformers import SentenceTransformer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FAISSVectorStore:
    """
    Complete FAISS-based vector store for document embeddings.
    
    Features:
    - Index creation and management
    - Metadata storage and retrieval
    - Cosine similarity search
    - Batch indexing
    - Save/load functionality
    - Statistics and monitoring
    """
    
    def __init__(
        self, 
        index_path: str = "faiss_index",
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        dimension: int = 1024,
        metric: str = "cosine"
    ):
        """
        Initialize FAISS vector store.
        
        Args:
            index_path: Path to save/load the index files
            embedding_model: Name of the sentence-transformers model
            dimension: Embedding dimension (default 1024 for bge-large)
            metric: Similarity metric ('cosine' or 'l2')
        """
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        self.dimension = dimension
        self.metric = metric
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize index (will be created on first add)
        self.index = None
        
        # Storage for chunks and metadata
        self.chunks = []          # List of text chunks
        self.metadatas = []       # List of metadata dicts
        self.id_to_index = {}     # Mapping from chunk_id to index position
        self.index_to_id = {}     # Reverse mapping
        
        # Statistics
        self.stats = {
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "total_chunks": 0,
            "total_indexes": 0,
            "queries_performed": 0
        }
        
        # Try to load existing index
        self.load()
        
        logger.info(f"FAISSVectorStore initialized at {self.index_path}")
    
    def _create_index(self) -> faiss.Index:
        """
        Create a new FAISS index based on metric type.
        
        Returns:
            FAISS index object
        """
        if self.metric == "cosine":
            # For cosine similarity, we use Inner Product with normalized vectors
            index = faiss.IndexFlatIP(self.dimension)
        else:
            # For Euclidean distance
            index = faiss.IndexFlatL2(self.dimension)
        
        return index
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine similarity."""
        if self.metric == "cosine":
            faiss.normalize_L2(vectors)
        return vectors
    
    def add_chunks(
        self, 
        chunks: List[Dict[str, Any]], 
        batch_size: int = 32,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Add chunks to FAISS index with batch processing.
        
        Args:
            chunks: List of chunks with 'text' and 'metadata'
            batch_size: Number of chunks to process at once
            show_progress: Show progress bar
            
        Returns:
            Dictionary with indexing results
        """
        if not chunks:
            logger.warning("No chunks to add")
            return {"status": "error", "message": "No chunks provided"}
        
        start_time = time.time()
        logger.info(f"Adding {len(chunks)} chunks to FAISS index...")
        
        # Store start index for this batch
        start_idx = len(self.chunks)
        
        # Prepare texts and metadata
        texts = []
        new_metadatas = []
        
        for chunk in chunks:
            if isinstance(chunk, dict):
                text = chunk.get("text", "")
                metadata = chunk.get("metadata", {})
            else:
                # Handle if chunk is not a dict
                text = str(chunk)
                metadata = {}
            
            texts.append(text)
            new_metadatas.append(metadata)
        
        # Generate embeddings in batches
        all_embeddings = []
        
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(range(0, len(texts), batch_size), desc="Generating embeddings")
        else:
            iterator = range(0, len(texts), batch_size)
        
        for i in iterator:
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch_texts,
                normalize_embeddings=(self.metric == "cosine"),
                show_progress_bar=False
            )
            all_embeddings.append(batch_embeddings)
        
        # Combine all embeddings
        if all_embeddings:
            embeddings = np.vstack(all_embeddings)
        else:
            embeddings = np.array([])
        
        # Create or update FAISS index
        if self.index is None:
            self.index = self._create_index()
            self.index.add(embeddings)
            logger.info(f"Created new FAISS index with {len(embeddings)} vectors")
        else:
            self.index.add(embeddings)
            logger.info(f"Added {len(embeddings)} vectors to existing index")
        
        # Store chunks and metadata
        self.chunks.extend(texts)
        self.metadatas.extend(new_metadatas)
        
        # Update ID mappings
        for i, (text, metadata) in enumerate(zip(texts, new_metadatas)):
            # Generate chunk ID if not present
            chunk_id = metadata.get("chunk_id")
            if not chunk_id:
                # Create deterministic ID based on content
                content_hash = hashlib.md5(f"{text[:100]}_{i}".encode()).hexdigest()[:8]
                chunk_id = f"chunk_{start_idx + i}_{content_hash}"
                metadata["chunk_id"] = chunk_id
            
            self.id_to_index[chunk_id] = start_idx + i
            self.index_to_id[start_idx + i] = chunk_id
        
        # Update statistics
        self.stats["updated_at"] = datetime.now().isoformat()
        self.stats["total_chunks"] = len(self.chunks)
        self.stats["total_indexes"] = self.index.ntotal if self.index else 0
        
        # Save index
        self.save()
        
        elapsed_time = time.time() - start_time
        
        result = {
            "status": "success",
            "chunks_added": len(chunks),
            "total_chunks": len(self.chunks),
            "index_size": self.index.ntotal if self.index else 0,
            "elapsed_time_seconds": round(elapsed_time, 2)
        }
        
        logger.info(f" Added {len(chunks)} chunks in {elapsed_time:.2f}s")
        return result
    
    def search(
        self, 
        query: str, 
        k: int = 10,
        filter_metadata: Optional[Dict] = None,
        return_scores: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_metadata: Optional metadata filters
            return_scores: Include similarity scores in results
            
        Returns:
            List of chunks with metadata and scores
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Search attempted on empty index")
            return []
        
        # Update query stats
        self.stats["queries_performed"] += 1
        
        # Encode query
        query_emb = self.embedding_model.encode(
            [query],
            normalize_embeddings=(self.metric == "cosine")
        )
        
        # Search
        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_emb, k)
        
        # Format results
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if 0 <= idx < len(self.chunks):
                # Apply metadata filter if provided
                if filter_metadata:
                    metadata = self.metadatas[idx]
                    match = True
                    for key, value in filter_metadata.items():
                        if metadata.get(key) != value:
                            match = False
                            break
                    if not match:
                        continue
                
                result = {
                    "text": self.chunks[idx],
                    "metadata": self.metadatas[idx].copy(),
                    "index": int(idx)
                }
                
                if return_scores:
                    # Convert score to similarity (0-1 range)
                    if self.metric == "cosine":
                        # For cosine, scores are between -1 and 1
                        similarity = (float(score) + 1) / 2  # Normalize to 0-1
                    else:
                        # For L2, convert distance to similarity
                        similarity = 1 / (1 + float(score))
                    
                    result["score"] = similarity
                    result["raw_score"] = float(score)
                
                results.append(result)
        
        return results
    
    def batch_search(
        self, 
        queries: List[str], 
        k: int = 10,
        show_progress: bool = True
    ) -> List[List[Dict[str, Any]]]:
        """
        Perform multiple searches efficiently.
        
        Args:
            queries: List of search queries
            k: Number of results per query
            show_progress: Show progress bar
            
        Returns:
            List of search results for each query
        """
        if self.index is None or self.index.ntotal == 0:
            return [[] for _ in queries]
        
        # Encode all queries
        query_embeddings = self.embedding_model.encode(
            queries,
            normalize_embeddings=(self.metric == "cosine"),
            show_progress_bar=show_progress
        )
        
        # Search
        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_embeddings, k)
        
        # Format results for each query
        all_results = []
        for query_idx in range(len(queries)):
            results = []
            for idx, score in zip(indices[query_idx], scores[query_idx]):
                if 0 <= idx < len(self.chunks):
                    result = {
                        "text": self.chunks[idx],
                        "metadata": self.metadatas[idx].copy(),
                        "score": float(score),
                        "index": int(idx)
                    }
                    results.append(result)
            all_results.append(results)
        
        self.stats["queries_performed"] += len(queries)
        return all_results
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a chunk by its ID."""
        if chunk_id in self.id_to_index:
            idx = self.id_to_index[chunk_id]
            return {
                "text": self.chunks[idx],
                "metadata": self.metadatas[idx].copy(),
                "index": idx
            }
        return None
    
    def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """Retrieve multiple chunks by their IDs."""
        results = []
        for chunk_id in chunk_ids:
            chunk = self.get_chunk_by_id(chunk_id)
            if chunk:
                results.append(chunk)
        return results
    
    def delete_chunk(self, chunk_id: str) -> bool:
        """
        Delete a chunk from the store.
        Note: FAISS doesn't support deletion directly, so this marks as deleted.
        """
        if chunk_id in self.id_to_index:
            idx = self.id_to_index[chunk_id]
            # Mark as deleted (set text to None)
            self.chunks[idx] = None
            self.metadatas[idx]["deleted"] = True
            self.metadatas[idx]["deleted_at"] = datetime.now().isoformat()
            self.save()
            return True
        return False
    
    def update_chunk(self, chunk_id: str, text: Optional[str] = None, 
                     metadata: Optional[Dict] = None) -> bool:
        """
        Update a chunk's text or metadata.
        Note: Updates require re-indexing the chunk.
        """
        if chunk_id not in self.id_to_index:
            return False
        
        idx = self.id_to_index[chunk_id]
        
        if text is not None:
            # Update text and re-embed
            self.chunks[idx] = text
            embedding = self.embedding_model.encode(
                [text],
                normalize_embeddings=(self.metric == "cosine")
            )
            
            # FAISS doesn't support direct updates, so we'd need to rebuild
            # For simplicity, we'll just update metadata for now
            logger.warning("Text updates require rebuilding the index")
        
        if metadata is not None:
            self.metadatas[idx].update(metadata)
        
        self.save()
        return True
    
    def save(self):
        """Save index and data to disk."""
        logger.info(f"Saving FAISS index to {self.index_path}...")
        
        # Save FAISS index
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path / "index.faiss"))
        
        # Save chunks and metadata
        data = {
            'chunks': self.chunks,
            'metadatas': self.metadatas,
            'id_to_index': self.id_to_index,
            'index_to_id': self.index_to_id,
            'stats': self.stats,
            'dimension': self.dimension,
            'metric': self.metric
        }
        
        with open(self.index_path / "data.pkl", 'wb') as f:
            pickle.dump(data, f)
        
        # Also save as JSON for inspection (optional)
        try:
            json_data = {
                'stats': self.stats,
                'total_chunks': len(self.chunks),
                'metadata_summary': [
                    {k: v for k, v in m.items() if k != 'text'} 
                    for m in self.metadatas[:10]  # First 10 only
                ]
            }
            with open(self.index_path / "metadata.json", 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Could not save JSON metadata: {e}")
        
        logger.info(f" Saved {len(self.chunks)} chunks to {self.index_path}")
    
    def load(self) -> bool:
        """Load index and data from disk."""
        try:
            # Load FAISS index
            index_file = self.index_path / "index.faiss"
            if index_file.exists():
                self.index = faiss.read_index(str(index_file))
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            
            # Load chunks and metadata
            data_file = self.index_path / "data.pkl"
            if data_file.exists():
                with open(data_file, 'rb') as f:
                    data = pickle.load(f)
                    self.chunks = data.get('chunks', [])
                    self.metadatas = data.get('metadatas', [])
                    self.id_to_index = data.get('id_to_index', {})
                    self.index_to_id = data.get('index_to_id', {})
                    self.stats = data.get('stats', self.stats)
                    self.dimension = data.get('dimension', self.dimension)
                    self.metric = data.get('metric', self.metric)
                
                logger.info(f"Loaded {len(self.chunks)} chunks from {data_file}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive index statistics."""
        stats = self.stats.copy()
        stats.update({
            "current_time": datetime.now().isoformat(),
            "index_exists": self.index is not None,
            "index_ntotal": self.index.ntotal if self.index else 0,
            "memory_usage_bytes": self.index.ntotal * self.dimension * 4 if self.index else 0,  # float32 = 4 bytes
            "chunks_with_metadata": len([m for m in self.metadatas if m]),
            "deleted_chunks": len([c for c in self.chunks if c is None]),
            "index_path": str(self.index_path),
            "dimension": self.dimension,
            "metric": self.metric
        })
        return stats
    
    def clear(self):
        """Clear all data and reset index."""
        self.index = None
        self.chunks = []
        self.metadatas = []
        self.id_to_index = {}
        self.index_to_id = {}
        self.stats["total_chunks"] = 0
        self.stats["total_indexes"] = 0
        self.stats["updated_at"] = datetime.now().isoformat()
        
        # Remove files
        index_file = self.index_path / "index.faiss"
        if index_file.exists():
            index_file.unlink()
        
        data_file = self.index_path / "data.pkl"
        if data_file.exists():
            data_file.unlink()
        
        logger.info("Cleared all data from FAISS store")
    
    def rebuild_index(self, batch_size: int = 32):
        """
        Rebuild the FAISS index from stored chunks.
        Useful after updates or deletions.
        """
        if not self.chunks:
            logger.warning("No chunks to rebuild index")
            return
        
        logger.info(f"Rebuilding index with {len(self.chunks)} chunks...")
        
        # Filter out deleted chunks
        valid_indices = [i for i, c in enumerate(self.chunks) if c is not None]
        valid_texts = [self.chunks[i] for i in valid_indices]
        valid_metadatas = [self.metadatas[i] for i in valid_indices]
        
        # Generate embeddings
        all_embeddings = []
        for i in range(0, len(valid_texts), batch_size):
            batch_texts = valid_texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch_texts,
                normalize_embeddings=(self.metric == "cosine"),
                show_progress_bar=True
            )
            all_embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(all_embeddings)
        
        # Create new index
        self.index = self._create_index()
        self.index.add(embeddings)
        
        # Update mappings
        self.chunks = valid_texts
        self.metadatas = valid_metadatas
        self.id_to_index = {}
        self.index_to_id = {}
        
        for new_idx, (text, metadata) in enumerate(zip(self.chunks, self.metadatas)):
            chunk_id = metadata.get("chunk_id", f"chunk_{new_idx}")
            self.id_to_index[chunk_id] = new_idx
            self.index_to_id[new_idx] = chunk_id
        
        self.stats["total_chunks"] = len(self.chunks)
        self.stats["total_indexes"] = self.index.ntotal
        self.stats["updated_at"] = datetime.now().isoformat()
        
        self.save()
        logger.info(f" Rebuilt index with {len(self.chunks)} chunks")
    
    def similarity_search_with_score(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Return chunks with similarity scores."""
        results = self.search(query, k=k, return_scores=True)
        return [(r["text"], r["score"]) for r in results]
    
    def similarity_search_by_vector(self, embedding: np.ndarray, k: int = 10) -> List[str]:
        """Search using a pre-computed embedding."""
        if self.index is None:
            return []
        
        # Normalize if needed
        if self.metric == "cosine":
            faiss.normalize_L2(embedding.reshape(1, -1))
        
        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(embedding.reshape(1, -1), k)
        
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.chunks) and self.chunks[idx] is not None:
                results.append(self.chunks[idx])
        
        return results


# Utility function to create a vector store from documents
def create_vector_store_from_documents(
    documents: List[Dict],
    index_path: str = "faiss_index",
    **kwargs
) -> FAISSVectorStore:
    """
    Create and populate a FAISS vector store from documents.
    
    Args:
        documents: List of document chunks with 'text' and 'metadata'
        index_path: Path to save the index
        **kwargs: Additional arguments for FAISSVectorStore
    
    Returns:
        Populated FAISSVectorStore instance
    """
    store = FAISSVectorStore(index_path=index_path, **kwargs)
    store.add_chunks(documents)
    return store


# Example usage
if __name__ == "__main__":
    # Example: Create and test vector store
    store = FAISSVectorStore(index_path="test_index")
    
    # Add some test chunks
    test_chunks = [
        {"text": "This is a test document about GRI reporting.", 
         "metadata": {"gri_id": "102-1", "page": 1}},
        {"text": "Sustainability metrics for 2024 show improvement.", 
         "metadata": {"gri_id": "305-1", "page": 42}},
    ]
    
    store.add_chunks(test_chunks)
    
    # Test search
    results = store.search("GRI reporting", k=2)
    print(f"Search results: {results}")
    
    # Get stats
    print(f"Store stats: {store.get_stats()}")
# test_query.py
from retrieval.faiss_store import FAISSVectorStore
from retrieval.hybrid_search import HybridSearch
from generator.llm_client import LLMClient
from config import Config

# Load vector store
vector_store = FAISSVectorStore(index_path=str(Config.CHROMA_DIR / "faiss"))
print(f"Loaded {vector_store.get_stats()['total_chunks']} chunks")

# Test search
results = vector_store.search("What are UPS's emissions?", k=5)
print(f"\nFound {len(results)} results")
for i, r in enumerate(results[:2]):
    print(f"\n--- Result {i+1} (score: {r['score']:.3f}) ---")
    print(r['text'][:200] + "...")
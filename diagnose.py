"""
Diagnostic script to test all components
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from config import Config
from retrieval.faiss_store import FAISSVectorStore
from retrieval.hybrid_search import HybridSearch
from retrieval.reranker import Reranker
from generator.llm_client import LLMClient
from generator.validator import ValidatorAgent
from debug.debugger import debugger, check_system

def diagnose_system():
    """Run diagnostics on all components"""
    
    print("\n" + "="*60)
    print("🔧 RAG SYSTEM DIAGNOSTIC TOOL")
    print("="*60)
    
    # Step 1: Check file system
    print("\n📁 Step 1: Checking file system...")
    check_system()
    
    # Step 2: Test FAISS Vector Store
    print("\n📊 Step 2: Testing FAISS Vector Store...")
    try:
        vector_store = FAISSVectorStore(index_path=str(Config.CHROMA_DIR / "faiss"))
        stats = vector_store.get_stats()
        print(f"✅ FAISS loaded: {stats.get('total_chunks', 0)} chunks")
        debugger.check_component("vector_store", vector_store)
    except Exception as e:
        print(f"❌ FAISS error: {e}")
        debugger.log_error("DIAGNOSTIC_FAISS", e)
    
    # Step 3: Test search functionality
    print("\n🔍 Step 3: Testing search...")
    try:
        # Create all_chunks for BM25
        all_chunks = []
        if hasattr(vector_store, 'chunks') and vector_store.chunks:
            for i in range(len(vector_store.chunks)):
                if vector_store.chunks[i] is not None:
                    all_chunks.append({
                        "text": vector_store.chunks[i],
                        "metadata": vector_store.metadatas[i] if i < len(vector_store.metadatas) else {}
                    })
        
        # Test hybrid search
        hybrid_search = HybridSearch(vector_store, all_chunks)
        debugger.check_component("hybrid_search", hybrid_search)
        
        # Perform test search
        test_query = "What are UPS emissions?"
        results = hybrid_search.search(test_query, k=3)
        print(f"✅ Search returned {len(results)} results")
        if results:
            print(f"   Top score: {results[0].get('score', 0):.3f}")
            print(f"   Preview: {results[0]['text'][:100]}...")
    except Exception as e:
        print(f"❌ Search error: {e}")
        debugger.log_error("DIAGNOSTIC_SEARCH", e)
    
    # Step 4: Test Reranker
    print("\n📈 Step 4: Testing reranker...")
    try:
        reranker = Reranker()
        debugger.check_component("reranker", reranker)
        if 'results' in locals() and results:
            reranked = reranker.rerank(test_query, results, k=2)
            print(f"✅ Reranker returned {len(reranked)} results")
    except Exception as e:
        print(f"❌ Reranker error: {e}")
        debugger.log_error("DIAGNOSTIC_RERANKER", e)
    
    # Step 5: Test LLM Client
    print("\n🤖 Step 5: Testing LLM client...")
    try:
        llm_client = LLMClient()
        debugger.check_component("llm_client", llm_client)
        print("✅ LLM client initialized")
    except Exception as e:
        print(f"❌ LLM client error: {e}")
        debugger.log_error("DIAGNOSTIC_LLM", e)
    
    # Step 6: Test Validator
    print("\n✅ Step 6: Testing validator...")
    try:
        if 'llm_client' in locals():
            validator = ValidatorAgent(llm_client)
            debugger.check_component("validator", validator)
            print("✅ Validator initialized")
    except Exception as e:
        print(f"❌ Validator error: {e}")
        debugger.log_error("DIAGNOSTIC_VALIDATOR", e)
    
    # Print final report
    print("\n" + "="*60)
    print("📊 DIAGNOSTIC COMPLETE")
    print("="*60)
    debugger.print_report()

if __name__ == "__main__":
    diagnose_system()
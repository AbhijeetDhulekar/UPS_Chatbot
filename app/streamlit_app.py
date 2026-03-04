# app/streamlit_app.py
"""
Streamlit Web Interface for UPS GRI RAG Chatbot
FAISS version - Complete and corrected
"""

import streamlit as st
import sys
import os
from pathlib import Path
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import Config, logger
from ingestion.indexer import Indexer
from retrieval.faiss_store import FAISSVectorStore
from retrieval.hybrid_search import HybridSearch
from retrieval.reranker import Reranker
from generator.llm_client import LLMClient
from generator.validator import ValidatorAgent
from memory.conversation_memory import ConversationMemory
from memory.redis_cache import SemanticCache
from models.guardrails import InputGuardrail, OutputGuardrail
from debug.debugger import debugger

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.messages = []
    st.session_state.session_id = str(int(time.time()))
    st.session_state.show_debug = False

# Page config
st.set_page_config(
    page_title="UPS GRI Report Q&A Bot",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 UPS GRI Report 2024 Q&A Bot")
st.markdown("Ask questions about UPS's sustainability and GRI disclosures")

# Sidebar
with st.sidebar:
    st.header("About")
    st.info(
        "This chatbot answers questions based on the UPS 2024 GRI Report. "
        "It uses advanced RAG with hybrid search, validation, and caching."
    )
    
    st.header("Debug Info")
    if st.button("Show Debug Summary"):
        debug_summary = debugger.get_summary()
        st.json(debug_summary)
        st.session_state.show_debug = True
    
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        if 'rag' in st.session_state and st.session_state.rag.get("memory"):
            try:
                st.session_state.rag["memory"].clear_history(st.session_state.session_id)
            except:
                pass
        st.rerun()
    
    st.header("Settings")
    show_sources = st.checkbox("Show sources", value=True)
    use_validation = st.checkbox("Use validation agent", value=True)
    
    # Show system stats if initialized
    if st.session_state.initialized and 'rag' in st.session_state:
        st.header("System Stats")
        try:
            stats = st.session_state.rag["vector_store"].get_stats()
            # Use correct keys from FAISSVectorStore.get_stats()
            st.metric("Total Chunks", stats.get('total_chunks', 0))
            st.metric("Index Vectors", stats.get('index_size', 0))
            st.metric("Dimension", stats.get('dimension', 1024))
            
            if 'llm_client' in st.session_state.rag:
                llm_stats = st.session_state.rag["llm_client"].get_stats()
                st.metric("API Calls", llm_stats.get('total_calls', 0))
        except Exception as e:
            st.error(f"Error loading stats: {e}")

# Initialize components
@st.cache_resource
def init_rag_system():
    """Initialize RAG components with FAISS (cached)"""
    with st.spinner("Initializing RAG system..."):
        try:
            # Initialize FAISS vector store
            vector_store = FAISSVectorStore(
                index_path=str(Config.CHROMA_DIR / "faiss")
            )
            
            # Check if index exists and has data
            stats = vector_store.get_stats()
            if stats.get('total_chunks', 0) == 0:
                st.info("📚 Indexing PDF for first time... This may take a few minutes.")
                indexer = Indexer()
                result = indexer.index_pdf(str(Config.PDF_PATH))
                st.success(f"✅ Indexed {result.get('total_chunks', 0)} chunks!")
                # Reload vector store after indexing
                vector_store = FAISSVectorStore(
                    index_path=str(Config.CHROMA_DIR / "faiss")
                )
                stats = vector_store.get_stats()
            
            # Get all chunks for BM25
            all_chunks = []
            if hasattr(vector_store, 'chunks') and vector_store.chunks:
                for i in range(len(vector_store.chunks)):
                    if vector_store.chunks[i] is not None:  # Skip deleted chunks
                        all_chunks.append({
                            "text": vector_store.chunks[i],
                            "metadata": vector_store.metadatas[i] if i < len(vector_store.metadatas) else {}
                        })
            
            st.info(f"📊 Loaded {len(all_chunks)} chunks from FAISS index")
            
            # Initialize components
            hybrid_search = HybridSearch(vector_store, all_chunks)
            reranker = Reranker()
            llm_client = LLMClient()
            validator = ValidatorAgent(llm_client)
            
            # Try to initialize Redis-based memory (optional)
            memory = None
            cache = None
            try:
                memory = ConversationMemory()
                cache = SemanticCache()
                st.success("✅ Redis cache connected")
            except Exception as e:
                logger.warning(f"Redis not available: {e}")
                st.info("⚠️ Running without Redis cache (optional)")
            
            return {
                "vector_store": vector_store,
                "hybrid_search": hybrid_search,
                "reranker": reranker,
                "llm_client": llm_client,
                "validator": validator,
                "memory": memory,
                "cache": cache,
                "all_chunks": all_chunks
            }
            
        except Exception as e:
            st.error(f"Failed to initialize RAG system: {e}")
            logger.error(f"Initialization error: {e}", exc_info=True)
            return None

# Initialize if needed
if not st.session_state.initialized:
    with st.spinner("Loading RAG system..."):
        st.session_state.rag = init_rag_system()
        if st.session_state.rag:
            st.session_state.initialized = True
            st.success("🚀 System ready! Ask your questions below.")
        else:
            st.error("Failed to initialize system. Check logs for details.")

# Query processing function
def process_query(question: str):
    """Process user query through RAG pipeline"""
    
    # Guardrail input
    is_valid, error_msg = InputGuardrail.validate_query(question)
    if not is_valid:
        return {"error": error_msg}
    
    question = InputGuardrail.sanitize_query(question)
    
    # Check cache if available
    if st.session_state.rag.get("cache"):
        try:
            cached = st.session_state.rag["cache"].get(question)
            if cached:
                logger.info(f"Cache hit for: {question[:50]}")
                return cached
        except Exception as e:
            logger.warning(f"Cache check failed: {e}")
    
    # Get conversation history if memory available
    history = ""
    if st.session_state.rag.get("memory"):
        try:
            history = st.session_state.rag["memory"].format_for_context(
                st.session_state.session_id
            )
        except Exception as e:
            logger.warning(f"Failed to get history: {e}")
    
    # Hybrid search
    with st.spinner("🔍 Searching document..."):
        initial_chunks = st.session_state.rag["hybrid_search"].search(question)
        if not initial_chunks:
            return {"error": "No relevant information found in the document."}
    
    # Re-rank
    with st.spinner("📊 Ranking results..."):
        top_chunks = st.session_state.rag["reranker"].rerank(question, initial_chunks)
    
    # Generate answer
    with st.spinner("🤖 Generating answer..."):
        answer = st.session_state.rag["llm_client"].generate_answer(
            question=question,
            context="\n\n".join([c["text"] for c in top_chunks]),
            conversation_history=history
        )
        # Add sources to answer object
        answer.sources = top_chunks
    
    # Validate if enabled
    validation = None
    final_answer = answer
    
    if use_validation:
        with st.spinner("✅ Validating answer..."):
            try:
                final_answer, validation = st.session_state.rag["validator"].validate_with_retry(
                    answer, top_chunks, question
                )
                
                if not validation.is_valid:
                    st.warning("⚠️ Answer required revision based on validation")
                    if st.session_state.show_debug:
                        with st.expander("🔧 Validation Details"):
                            st.json(validation.dict())
            except Exception as e:
                logger.error(f"Validation failed: {e}")
                st.warning("⚠️ Validation step failed, showing original answer")
    
    # Format with citations
    if show_sources:
        final_answer.content = OutputGuardrail.format_citations(
            final_answer.content, 
            top_chunks
        )
    
    result = {
        "answer": final_answer.content,
        "sources": top_chunks if show_sources else [],
        "validation": validation.dict() if validation else None
    }
    
    # Cache result if cache available
    if st.session_state.rag.get("cache"):
        try:
            st.session_state.rag["cache"].set(question, result)
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")
    
    return result

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"] and show_sources:
            with st.expander("📚 View Sources"):
                for i, src in enumerate(message["sources"][:3], 1):
                    page = src['metadata'].get('page_start', 'N/A')
                    gri_id = src['metadata'].get('gri_id', 'N/A')
                    section = src['metadata'].get('section_header', 'Unknown')
                    
                    st.markdown(f"**Source {i}** - Page {page}")
                    st.markdown(f"**GRI:** {gri_id} | **Section:** {section}")
                    st.markdown(f"*{src['text'][:200]}...*")
                    st.divider()

# Chat input
if prompt := st.chat_input("Ask about the UPS GRI report..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = process_query(prompt)
            
            if "error" in result:
                st.error(f"❌ {result['error']}")
                answer_content = result['error']
                sources = []
            else:
                st.markdown(result["answer"])
                answer_content = result["answer"]
                sources = result.get("sources", [])
                
                # Show sources if enabled
                if sources and show_sources:
                    with st.expander("📚 Sources Used"):
                        for i, src in enumerate(sources[:3], 1):
                            col1, col2 = st.columns([1, 4])
                            with col1:
                                st.markdown(f"**Page {src['metadata'].get('page_start', 'N/A')}**")
                            with col2:
                                st.markdown(f"*{src['text'][:150]}...*")
                            st.divider()
    
    # Save to memory
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer_content,
        "sources": sources
    })
    
    # Save to Redis memory if available
    if st.session_state.rag.get("memory"):
        try:
            st.session_state.rag["memory"].add_message(
                st.session_state.session_id,
                {"role": "user", "content": prompt}
            )
            st.session_state.rag["memory"].add_message(
                st.session_state.session_id,
                {"role": "assistant", "content": answer_content}
            )
        except Exception as e:
            logger.warning(f"Failed to save to memory: {e}")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("💡 *Powered by GPT-4 + RAG*")
with col2:
    if st.session_state.initialized and 'rag' in st.session_state:
        try:
            chunk_count = st.session_state.rag['vector_store'].get_stats().get('total_chunks', 0)
            st.markdown(f"📚 *{chunk_count} chunks indexed*")
        except:
            st.markdown("📚 *FAISS index loaded*")
with col3:
    st.markdown("✅ *With validation agent*")
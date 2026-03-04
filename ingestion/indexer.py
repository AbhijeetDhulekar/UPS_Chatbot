# ingestion/indexer.py
"""
Document Indexer using FAISS
"""

import hashlib
from pathlib import Path
from typing import Dict
from loguru import logger
from config import Config
from ingestion.parser import PDFParser
from ingestion.chunker import HybridChunker
from ingestion.metadata import MetadataEnricher
from retrieval.faiss_store import FAISSVectorStore
from debug.debugger import debugger
from config import CHROMA_DIR

class Indexer:
    """Index documents into FAISS vector store"""
    
    def __init__(self):
        """Initialize indexer with FAISS vector store."""
        # Create FAISS index directory
        # faiss_path = Config.CHROMA_DIR / "faiss"
        faiss_path = CHROMA_DIR / "faiss"
        faiss_path.mkdir(parents=True, exist_ok=True)
        
        self.vector_store = FAISSVectorStore(index_path=str(faiss_path))
        logger.info(f"Initialized FAISS indexer at {faiss_path}")
    
    @debugger.timer
    def index_pdf(self, pdf_path: str) -> Dict:
        """
        Index PDF document into FAISS.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with indexing results
        """
        logger.info(f"Starting indexing of {pdf_path}")
        
        # Step 1: Parse PDF
        parser = PDFParser(pdf_path)
        text_blocks, tables = parser.parse()
        logger.info(f"Parsed {len(text_blocks)} text blocks and {len(tables)} tables")
        
        # Step 2: Chunk document
        chunker = HybridChunker()
        chunks = chunker.process_document(text_blocks)
        logger.info(f"Created {len(chunks)} chunks from text")
        
        # Step 3: Add tables as special chunks
        for table in tables:
            # Generate a unique ID for the table
            table_id = hashlib.md5(f"table_{table['page']}_{len(table['data'])}".encode()).hexdigest()[:8]
            
            chunks.append({
                "text": table["markdown"],
                "metadata": {
                    "gri_id": None,
                    "section_header": "Table",
                    "page_start": table["page"],
                    "page_end": table["page"],
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "is_table": True,
                    "chunk_id": f"table_{table_id}"
                }
            })
        
        logger.info(f"Added {len(tables)} table chunks")
        
        # Step 4: Enrich metadata
        enriched_chunks = MetadataEnricher.enrich(chunks)
        logger.info(f"Enriched metadata for {len(enriched_chunks)} chunks")
        
        # Step 5: Index in FAISS (NOT ChromaDB)
        chunks_added = self.vector_store.add_chunks(enriched_chunks)
        
        result = {
            "total_chunks": chunks_added,
            "tables_indexed": len(tables),
            "index_path": str(self.vector_store.index_path),
            "index_stats": self.vector_store.get_stats()
        }
        
        debugger.log("INDEXING", result)
        logger.success(f"Indexing complete: {result}")
        
        return result
    
    def get_vector_store(self):
        """Get the FAISS vector store instance."""
        return self.vector_store

# Run indexing if script executed directly
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add parent directory to path
    sys.path.append(str(Path(__file__).parent.parent))
    
    indexer = Indexer()
    result = indexer.index_pdf(str(Config.PDF_PATH))
    print(f" Indexing complete: {result}")
    print(f" Total chunks: {result['total_chunks']}")
    print(f" Index stats: {result['index_stats']}")
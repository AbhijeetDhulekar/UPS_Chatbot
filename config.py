import os
from dotenv import load_dotenv
from pathlib import Path
from loguru import logger

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "cache"
CHROMA_DIR = BASE_DIR / "chroma_db"          # FAISS index location (name kept for compatibility)
LOG_DIR = BASE_DIR / "logs"
EXPORT_DIR = BASE_DIR / "exports"            # For chat history PDFs

# Create directories
for dir_path in [DATA_DIR, CACHE_DIR, CHROMA_DIR, LOG_DIR, EXPORT_DIR]:
    dir_path.mkdir(exist_ok=True)

# Configure logger
logger.add(
    LOG_DIR / "rag_system_{time}.log",
    rotation="500 MB",
    retention="10 days",
    level="DEBUG"
)

class Config:
    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")   # Moved to env var
    OPENAI_MODEL = "gpt-4o"
    
    # Embedding Model
    EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
    EMBEDDING_DIM = 1024
    
    # Chunking
    MAX_CHUNK_SIZE = 800
    CHUNK_OVERLAP = 150
    SEMANTIC_THRESHOLD = 0.65
    
    # Retrieval
    TOP_K_INITIAL = 20
    TOP_K_FINAL = 5
    BM25_WEIGHT = 0.5
    VECTOR_WEIGHT = 0.5
    
    # Redis
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB = int(os.getenv("REDIS_DB", 0))
    CACHE_TTL = 3600          # 1 hour for semantic cache
    CONVERSATION_TTL = 7200    # 2 hours for chat history
    
    # Validation
    MAX_VALIDATION_ATTEMPTS = 2
    CONFIDENCE_THRESHOLD = 0.8
    
    # System
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    PDF_PATH = DATA_DIR / "2024-UPS-GRI-Report.pdf"
    
    # LangChain specific
    LANGCHAIN_CACHE = True                         # Enable LangChain's cache
    LANGCHAIN_CACHE_TYPE = "redis"                  # "redis" or "memory"
    
    # Evaluation
    EVAL_QUESTIONS_PATH = DATA_DIR / "eval_questions.json"   # Optional: pre-defined Q&A pairs
    ENABLE_EVALUATION = True
    
    # Chroma directory (mirrors module-level CHROMA_DIR)
    CHROMA_DIR = CHROMA_DIR
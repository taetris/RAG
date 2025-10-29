"""Configuration for compliance RAG system."""
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ICAEW_FILE = DATA_DIR / "version1.txt"
GDPR_FILE = DATA_DIR / "version2.txt"

# Model configuration
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # Free, good performance
LLM_MODEL = "gpt-4o-mini"  # Cheap and effective

# Chunking configuration
ICAEW_SECTION_PATTERN = r'(\d+\.\d+(?:\.\d+)?)\s+'  # Matches 5.1, 5.1.1, etc.
GDPR_ARTICLE_PATTERN = r'(Article \d+)'

# Retrieval configuration
TOP_K_DENSE = 10  # Top results from embedding search
TOP_K_BM25 = 10   # Top results from keyword search
TOP_K_FINAL = 3   # Final results after reranking

# ChromaDB configuration
CHROMA_PERSIST_DIR = str(PROJECT_ROOT / "chroma_db")
POLICY_COLLECTION = "icaew_policy"
GDPR_COLLECTION = "gdpr_regulation"

# Compliance levels
COMPLIANCE_LEVELS = ["Full", "Partial", "Non-compliant", "Not Applicable"]
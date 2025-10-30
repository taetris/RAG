"""
Configuration settings for document comparison.
"""
import os

from anyio import Path

# Base folder is the parent of this file
BASE_DIR = Path(__file__).parent.parent

# Paths
DATA_FOLDER = BASE_DIR / "data"
OUTPUT_FOLDER = BASE_DIR / "outputs"
CACHE_FOLDER = BASE_DIR / "cache"

# File names
POLICY_FILE = "data_protection.pdf"
REGULATION_FILE = "version2.pdf"
REPORT_FILE = "comparison_report.csv"
EMBEDDINGS_CACHE = "embeddings_cache.pkl"

# Text splitting configuration
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150

# Similarity thresholds (tunable based on your needs)
THRESHOLD_UNCHANGED = 0.95
THRESHOLD_MODIFIED = 0.70

# Model configuration
EMBEDDING_MODEL = "law-ai/InLegalBERT"
LLM_MODEL = "gpt-4o-mini"  # Cheap and effective

# Chunking configuration
POLICY_SECTION_PATTERN = r'(\d+\.\d+(?:\.\d+)?)\s+'  # Matches 5.1, 5.1.1, etc.
REGULATION_ARTICLE_PATTERN = r'(Article \d+)'

# Retrieval configuration
TOP_K_DENSE = 10  # Top results from embedding search
TOP_K_BM25 = 10   # Top results from keyword search
TOP_K_FINAL = 3   # Final results after reranking

# ChromaDB configuration
CHROMA_PERSIST_DIR = str(BASE_DIR / "chroma_db")
POLICY_COLLECTION = "icaew_policy"
LAW_COLLECTION = "gdpr_regulation"

# Compliance levels
COMPLIANCE_LEVELS = ["Full", "Partial", "Non-compliant", "Not Applicable"]

def ensure_directories():
    """Create necessary directories if they don't exist."""
    for folder in [DATA_FOLDER, OUTPUT_FOLDER, CACHE_FOLDER]:
        os.makedirs(folder, exist_ok=True)
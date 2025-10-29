"""
Configuration settings for document comparison.
"""
import os

from anyio import Path

# Base folder is the parent of this file
BASE_DIR = Path(__file__).parent.parent

# Paths
DATA_FOLDER = BASE_DIR / "data"
OUTPUT_FOLDER = BASE_DIR / "output"
CACHE_FOLDER = BASE_DIR / "cache"

# File names
VERSION1_FILE = "data_protection.pdf"
VERSION2_FILE = "v2.pdf"
REPORT_FILE = "comparison_report.csv"
EMBEDDINGS_CACHE = "embeddings_cache.pkl"

# Model configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEVICE = "cpu"  # Change to "cuda" if GPU available
BATCH_SIZE = 32  # For embedding generation

# Text splitting configuration
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150

# Similarity thresholds (tunable based on your needs)
THRESHOLD_UNCHANGED = 0.95
THRESHOLD_MODIFIED = 0.70

# Processing
MAX_WORKERS = 4  # For parallel processing

def ensure_directories():
    """Create necessary directories if they don't exist."""
    for folder in [DATA_FOLDER, OUTPUT_FOLDER, CACHE_FOLDER]:
        os.makedirs(folder, exist_ok=True)
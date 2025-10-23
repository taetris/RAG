"""
Configuration settings for document comparison.
"""
import os

# Paths
DATA_FOLDER = "data"
OUTPUT_FOLDER = "output"
CACHE_FOLDER = "cache"

# File names
VERSION1_FILE = "version1.pdf"
VERSION2_FILE = "version2.pdf"
REPORT_FILE = "comparison_report.csv"
EMBEDDINGS_CACHE = "embeddings_cache.pkl"

# Model configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEVICE = "cpu"  # Change to "cuda" if GPU available
BATCH_SIZE = 32  # For embedding generation

# Text splitting configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Similarity thresholds (tunable based on your needs)
THRESHOLD_UNCHANGED = 0.95
THRESHOLD_MODIFIED = 0.70

# Processing
MAX_WORKERS = 4  # For parallel processing

def ensure_directories():
    """Create necessary directories if they don't exist."""
    for folder in [DATA_FOLDER, OUTPUT_FOLDER, CACHE_FOLDER]:
        os.makedirs(folder, exist_ok=True)
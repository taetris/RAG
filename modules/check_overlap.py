"""
Compare textual overlap between two documents.
"""
import logging
import re
from pathlib import Path
from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from pdf_extractor import PDFExtractor 
from pathlib import Path
from pdf_extractor import PDFExtractor
from config import DATA_FOLDER, VERSION1_FILE, VERSION2_FILE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_text(file_path: Path) -> str:
    """Reads text from a .pdf or .txt file."""
    extractor = PDFExtractor()

    if file_path.suffix.lower() == ".pdf":
        logger.info(f"Extracting text from PDF: {file_path.name}")
        return extractor.extract_text(str(file_path))
    elif file_path.suffix.lower() == ".txt":
        logger.info(f"Reading text file: {file_path.name}")
        return file_path.read_text(encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")


def clean_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compute_overlap(text1: str, text2: str) -> Tuple[float, float]:
    """Compute similarity between two texts using TF-IDF and Jaccard."""
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform([text1, text2])
    cosine_sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

    words1, words2 = set(text1.split()), set(text2.split())
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    jaccard = intersection / union if union != 0 else 0.0

    return cosine_sim, jaccard


if __name__ == "__main__":
   

    # Base directory = parent of current script’s folder (modules/)
    BASE_DIR = Path(__file__).resolve().parent.parent

    file1 = DATA_FOLDER / VERSION1_FILE
    file2 = DATA_FOLDER / VERSION2_FILE

    text1 = clean_text(read_text(file1))
    text2 = clean_text(read_text(file2))

    cosine_sim, jaccard_sim = compute_overlap(text1, text2)

    print("\n=== OVERLAP REPORT ===")
    print(f"File 1: {file1.name}")
    print(f"File 2: {file2.name}")
    print(f"Cosine similarity (TF-IDF): {cosine_sim:.3f}")
    print(f"Jaccard word overlap:       {jaccard_sim:.3f}")
    print(f"Total unique words (v1):    {len(set(text1.split()))}")
    print(f"Total unique words (v2):    {len(set(text2.split()))}")
    print(f"Common words:               {len(set(text1.split()) & set(text2.split()))}")

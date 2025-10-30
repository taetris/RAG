"""
Hierarchical PDF parser and chunker for compliance and legal documents.
Parses structure like 1 / 1.1 / 1.1.1 and preserves parent-child hierarchy.
"""

import os
import re
import json
from dataclasses import dataclass, asdict
from typing import List, Optional
from PyPDF2 import PdfReader
from config import DATA_FOLDER, POLICY_FILE, REGULATION_FILE, OUTPUT_FOLDER

# -------------------------------
# Dataclass for structured chunks
# -------------------------------
@dataclass
class LegalChunk:
    section_id: str
    section_title: str
    parent_sections: List[str]
    level: int
    text: str
    doc_type: str


# -------------------------------
# Helper: Read and clean text
# -------------------------------
def extract_text_from_pdf(file_path: str) -> str:
    """Extracts text from a PDF, skipping headers/footers and TOC-like sections."""
    reader = PdfReader(file_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        # Remove page numbers, extra dots, TOC patterns
        text = re.sub(r"\.{5,}\d+", "", text)
        text = re.sub(r"Page \d+ of \d+", "", text)
        pages.append(text.strip())
    full_text = "\n".join(pages)
    # Trim front matter for internal policy documents
    start_match = re.search(r"\b1\s+Policy statement\b", full_text, re.IGNORECASE)
    if start_match:
        full_text = full_text[start_match.start():]
    return full_text


# -------------------------------
# Hierarchical parser
# -------------------------------
class HierarchicalChunker:
    def __init__(self, doc_type: str):
        self.doc_type = doc_type
        # Patterns differ slightly for policy vs regulation
        self.section_pattern = (
            r"(?P<id>\d+(?:\.\d+)+)\s+(?P<title>[A-Z].*?)\n"
            if doc_type == "policy"
            else r"(?P<id>Article\s+\d+)\s*(?P<title>[A-Z].*?)\n"
        )

    def chunk(self, text: str) -> List[LegalChunk]:
        """Chunk text hierarchically with context."""
        matches = list(re.finditer(self.section_pattern, text))
        chunks = []

        for i, match in enumerate(matches):
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            chunk_text = text[start:end].strip()

            section_id = match.group("id").strip()
            section_title = match.group("title").strip()
            parent_sections = self._get_parent_sections(section_id)
            level = section_id.count(".") + 1 if "." in section_id else 1

            chunks.append(
                LegalChunk(
                    section_id=section_id,
                    section_title=section_title,
                    parent_sections=parent_sections,
                    level=level,
                    text=chunk_text,
                    doc_type=self.doc_type,
                )
            )

        return chunks

    def _get_parent_sections(self, section_id: str) -> List[str]:
        """Return list of parent section IDs (e.g., for 2.1.3 → ['2', '2.1'])."""
        if self.doc_type == "regulation":
            return []
        parts = section_id.split(".")
        return [".".join(parts[:i]) for i in range(1, len(parts))]


# -------------------------------
# Convenience function
# -------------------------------
def parse_and_chunk_pdf(file_path: str, doc_type: str) -> List[LegalChunk]:
    text = extract_text_from_pdf(file_path)
    chunker = HierarchicalChunker(doc_type)
    chunks = chunker.chunk(text)
    return chunks


# -------------------------------
# Utility: Save as JSON
# -------------------------------
def save_chunks_json(filename: str, chunks: List[LegalChunk]):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    output_path = os.path.join(OUTPUT_FOLDER, filename)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in chunks], f, ensure_ascii=False, indent=2)
    print(f"✓ Saved {len(chunks)} chunks to {output_path}")


# -------------------------------
# Test entry point
# -------------------------------
if __name__ == "__main__":
    print("Processing policy document...")
    policy_path = os.path.join(DATA_FOLDER, POLICY_FILE)
    policy_chunks = parse_and_chunk_pdf(policy_path, "policy")
    save_chunks_json("policy_chunks.json", policy_chunks)

    print("\nProcessing regulation document...")
    regulation_path = os.path.join(DATA_FOLDER, REGULATION_FILE)
    reg_chunks = parse_and_chunk_pdf(regulation_path, "regulation")
    save_chunks_json("regulation_chunks.json", reg_chunks)

 
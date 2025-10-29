"""Document chunking utilities."""
import re
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class Chunk:
    """Represents a document chunk with metadata."""
    text: str
    section_id: str
    section_title: str
    doc_type: str  # "icaew" or "gdpr"
    
    def __str__(self):
        return f"[{self.section_id}] {self.section_title}\n{self.text[:100]}..."


class DocumentChunker:
    """Handles chunking of legal documents by structural elements."""
    
    def __init__(self, icaew_pattern: str, gdpr_pattern: str):
        self.icaew_pattern = icaew_pattern
        self.gdpr_pattern = gdpr_pattern
    
    def chunk_icaew(self, text: str) -> List[Chunk]:
        """
        Chunk ICAEW policy by section numbers.
        
        Example:
            5.1 Some principle
            5.1.1 Details about principle
        """
        chunks = []
        
        # Split on section headers while keeping them
        parts = re.split(self.icaew_pattern, text)
        
        # parts will be ['intro', '5.1', 'content', '5.2', 'content', ...]
        for i in range(1, len(parts) - 1, 2):
            section_id = parts[i].strip()
            content = parts[i + 1].strip()
            
            if not content:
                continue
            
            # Extract title (first line usually)
            lines = content.split('\n', 1)
            title = lines[0].strip() if lines else "Untitled"
            full_content = lines[1].strip() if len(lines) > 1 else content
            
            chunks.append(Chunk(
                text=full_content,
                section_id=section_id,
                section_title=title,
                doc_type="icaew"
            ))
        
        return chunks
    
    def chunk_gdpr(self, text: str) -> List[Chunk]:
        """
        Chunk GDPR by articles.
        
        Example:
            Article 5
            Principles relating to processing of personal data
            1. Personal data shall be...
        """
        chunks = []
        
        # Split on article headers
        parts = re.split(self.gdpr_pattern, text)
        
        for i in range(1, len(parts) - 1, 2):
            article_num = parts[i].strip()
            content = parts[i + 1].strip()
            
            if not content:
                continue
            
            # Extract title (first line after article number)
            lines = content.split('\n', 1)
            title = lines[0].strip() if lines else "Untitled"
            full_content = lines[1].strip() if len(lines) > 1 else content
            
            chunks.append(Chunk(
                text=full_content,
                section_id=article_num,
                section_title=title,
                doc_type="gdpr"
            ))
        
        return chunks
    
    def chunk_document(self, text: str, doc_type: str) -> List[Chunk]:
        """Chunk document based on type."""
        if doc_type == "icaew":
            return self.chunk_icaew(text)
        elif doc_type == "gdpr":
            return self.chunk_gdpr(text)
        else:
            raise ValueError(f"Unknown document type: {doc_type}")


# Convenience function
def create_chunker() -> DocumentChunker:
    """Create chunker with default patterns."""
    from config import ICAEW_SECTION_PATTERN, GDPR_ARTICLE_PATTERN
    return DocumentChunker(ICAEW_SECTION_PATTERN, GDPR_ARTICLE_PATTERN)
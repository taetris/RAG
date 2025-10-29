"""
PDF text extraction with error handling and structure preservation.
"""
import logging
from typing import List, Dict
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class PDFExtractor:
    """Extract and chunk text from PDF documents."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],  # Respect structure
            length_function=len,
        )
    
    def extract_text(self, pdf_path: str) -> str:
        """
        Extract text from PDF with error handling.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text as string
            
        Raises:
            FileNotFoundError: If PDF doesn't exist
            Exception: If PDF is corrupted or unreadable
        """
        try:
            doc = fitz.open(pdf_path)
            if doc.page_count == 0:
                raise ValueError(f"PDF has no pages: {pdf_path}")
            
            text_parts = []
            for i in range(doc.page_count):
                try:
                    page = doc.load_page(i)
                    page_text = page.get_text()
                    if page_text and page_text.strip():
                        text_parts.append(page_text)
                except Exception as e:
                    logger.warning(f"Failed to extract page {i+1}: {e}")
                    continue
            print(f"Extracted {len(text_parts)} pages.")

            if not text_parts:
                raise ValueError(f"No text extracted from PDF: {pdf_path}")
            
            full_text = "\n\n".join(text_parts)
            
            # logger.info(f"Extracted {len(full_text)} characters from {pdf_path}")
            return full_text
            
        except FileNotFoundError:
            logger.error(f"PDF not found: {pdf_path}")
            raise
        except Exception as e:
            logger.error(f"Error reading PDF {pdf_path}: {e}")
            raise
    
    def split_text(self, text: str) -> List[Dict[str, any]]:
        """
        Split text into chunks with metadata.
        
        Args:
            text: Text to split
            
        Returns:
            List of dictionaries with chunk text and metadata
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for splitting")
            return []
        
        chunks = self.splitter.split_text(text)
        
        # Add metadata to each chunk
        chunk_data = []
        for idx, chunk in enumerate(chunks):
            chunk_data.append({
                "id": idx,
                "text": chunk,
                "length": len(chunk),
                "word_count": len(chunk.split())
            })
        
        logger.info(f"Split text into {len(chunk_data)} chunks")
        return chunk_data
    
    def process_pdf(self, pdf_path: str) -> List[Dict[str, any]]:
        """
        Complete pipeline: extract and chunk PDF.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of chunk dictionaries
        """
        text = self.extract_text(pdf_path)
        return self.split_text(text)
 
        logger.error(f"Failed to process PDF: {e}")

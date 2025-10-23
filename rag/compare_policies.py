"""
Main pipeline for document comparison.
"""
import logging
import sys
from pathlib import Path

import config
from pdf_extractor import PDFExtractor
from embedder import EmbeddingGenerator
from comparator import DocumentComparator
from reporter import ReportGenerator


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('document_comparison.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def validate_files() -> tuple[Path, Path]:
    """
    Validate that input PDFs exist.
    
    Returns:
        Tuple of (file1_path, file2_path)
        
    Raises:
        FileNotFoundError: If files don't exist
    """
    file1 = Path(config.DATA_FOLDER) / config.VERSION1_FILE
    file2 = Path(config.DATA_FOLDER) / config.VERSION2_FILE
    
    if not file1.exists():
        raise FileNotFoundError(
            f"Version 1 PDF not found: {file1}\n"
            f"Please place '{config.VERSION1_FILE}' in the '{config.DATA_FOLDER}' folder."
        )
    
    if not file2.exists():
        raise FileNotFoundError(
            f"Version 2 PDF not found: {file2}\n"
            f"Please place '{config.VERSION2_FILE}' in the '{config.DATA_FOLDER}' folder."
        )
    
    return file1, file2


def main():
    """Main execution pipeline."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting document comparison pipeline")
        
        # Ensure directories exist
        config.ensure_directories()
        
        # Validate input files
        logger.info("Validating input files...")
        file1, file2 = validate_files()
        
        # Initialize components
        logger.info("Initializing components...")
        extractor = PDFExtractor(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        
        embedder = EmbeddingGenerator(
            model_name=config.EMBEDDING_MODEL,
            device=config.DEVICE,
            batch_size=config.BATCH_SIZE,
            cache_dir=config.CACHE_FOLDER
        )
        
        comparator = DocumentComparator(
            threshold_unchanged=config.THRESHOLD_UNCHANGED,
            threshold_modified=config.THRESHOLD_MODIFIED,
            use_text_similarity=True
        )
        
        reporter = ReportGenerator(output_dir=config.OUTPUT_FOLDER)
        
        # Step 1: Extract and chunk PDFs
        logger.info(f"Processing {file1.name}...")
        chunks_v1 = extractor.process_pdf(str(file1))
        
        logger.info(f"Processing {file2.name}...")
        chunks_v2 = extractor.process_pdf(str(file2))
        
        if not chunks_v1 or not chunks_v2:
            raise ValueError("Failed to extract text from one or both PDFs")
        
        logger.info(f"Version 1: {len(chunks_v1)} chunks")
        logger.info(f"Version 2: {len(chunks_v2)} chunks")
        
        # Step 2: Generate embeddings
        logger.info("Generating embeddings for Version 1...")
        embeddings_v1 = embedder.generate_for_chunks(chunks_v1)
        
        logger.info("Generating embeddings for Version 2...")
        embeddings_v2 = embedder.generate_for_chunks(chunks_v2)
        
        # Step 3: Compare documents
        logger.info("Comparing documents...")
        results, stats = comparator.compare(
            chunks_v1, chunks_v2, embeddings_v1, embeddings_v2
        )
        
        # Step 4: Generate reports
        logger.info("Generating reports...")
        report_paths = reporter.generate_all_reports(results, stats)
        
        # Print summary
        print("\n" + "=" * 80)
        print("COMPARISON COMPLETE")
        print("=" * 80)
        print(f"\nTotal Sections Analyzed: {stats['total_sections']}")
        print(f"Unchanged: {stats['unchanged']} ({stats['unchanged']/stats['total_sections']*100:.1f}%)")
        print(f"Modified: {stats['modified']} ({stats['modified']/stats['total_sections']*100:.1f}%)")
        print(f"Removed or New: {stats['removed_or_new']}")
        print(f"New in V2: {stats['new_in_v2']}")
        print(f"Average Similarity: {stats['avg_similarity']:.3f}")
        
        print(f"\nReports saved to '{config.OUTPUT_FOLDER}' folder:")
        for
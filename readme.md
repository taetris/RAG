# Document Version Comparison Tool

A production-ready tool for comparing two versions of PDF documents using semantic embeddings and text similarity.

## Features

- **Semantic Comparison**: Uses HuggingFace embeddings to understand content changes beyond exact text matching
- **Bidirectional Analysis**: Detects both modified sections and new content in version 2
- **Efficient Processing**: Batch processing, caching, and memory-optimized operations
- **Multiple Report Formats**: CSV, detailed text report, and summary
- **Robust Error Handling**: Comprehensive logging and error recovery
- **Configurable**: Easy to tune thresholds and parameters

## Project Structure

```
project/
├── config.py              # Configuration settings
├── pdf_extractor.py       # PDF text extraction and chunking
├── embedder.py           # Embedding generation with caching
├── comparator.py         # Document comparison logic
├── reporter.py           # Report generation
├── main.py              # Main pipeline
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── data/                # Input PDFs (create this folder)
│   ├── version1.pdf
│   └── version2.pdf
├── output/              # Generated reports (auto-created)
├── cache/               # Embedding cache (auto-created)
└── document_comparison.log  # Log file (auto-created)
```

## Installation

### 1. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. For GPU support (optional)

If you have a CUDA-compatible GPU:

```bash
# For CUDA 11.8
pip install torch==2.0.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch==2.0.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

Then update `config.py` to set `DEVICE = "cuda"`.

## Usage

### Basic Usage

1. Create the `data` folder and place your PDFs:
   ```bash
   mkdir data
   # Copy your PDFs as version1.pdf and version2.pdf
   ```

2. Run the comparison:
   ```bash
   python main.py
   ```

3. Check the `output` folder for reports.

### Configuration

Edit `config.py` to customize:

```python
# Embedding model (options: all-MiniLM-L6-v2, all-mpnet-base-v2, etc.)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Processing device
DEVICE = "cpu"  # or "cuda" for GPU

# Similarity thresholds (0.0 to 1.0)
THRESHOLD_UNCHANGED = 0.95  # Above this = unchanged
THRESHOLD_MODIFIED = 0.70   # Above this = modified

# Text chunking
CHUNK_SIZE = 500           # Characters per chunk
CHUNK_OVERLAP = 50         # Overlap between chunks
```

## Output Reports

### 1. CSV Report (`comparison_report.csv`)
Spreadsheet-friendly format with columns:
- Section ID
- Status (Unchanged/Modified/Removed or New/New in V2)
- Similarity score
- Text snippets from both versions

### 2. Detailed Report (`detailed_report.txt`)
Human-readable report grouped by change type with:
- Summary statistics
- Full text snippets for each section
- Detailed similarity scores

### 3. Summary (`summary.txt`)
Quick overview with counts and percentages.

## Understanding the Results

### Status Categories

- **Unchanged** (similarity ≥ 0.95): Content is essentially the same
- **Modified** (0.70 ≤ similarity < 0.95): Content has been edited
- **Removed or New** (similarity < 0.70): Significant changes or content not found
- **New in V2**: Sections that only appear in version 2

### Similarity Scores

- **1.0**: Identical content
- **0.95-0.99**: Minor wording changes
- **0.70-0.95**: Moderate edits, same topic
- **< 0.70**: Major changes or different content

## Advanced Usage

### Using Different Embedding Models

For legal documents, consider:
```python
EMBEDDING_MODEL = "nlpaueb/legal-bert-base-uncased"
```

For multilingual documents:
```python
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

### Tuning for Your Use Case

**For detecting minor changes** (strict):
```python
THRESHOLD_UNCHANGED = 0.98
THRESHOLD_MODIFIED = 0.85
```

**For detecting major changes only** (lenient):
```python
THRESHOLD_UNCHANGED = 0.90
THRESHOLD_MODIFIED = 0.60
```

### Processing Large Documents

For documents with 100+ pages:
1. Increase batch size if you have GPU:
   ```python
   BATCH_SIZE = 64  # or higher
   ```
2. The tool automatically uses caching to speed up re-runs
3. Clear cache if you change the embedding model:
   ```bash
   rm -rf cache/*
   ```

## Troubleshooting

### Out of Memory Errors

1. Reduce batch size in `config.py`:
   ```python
   BATCH_SIZE = 16  # or lower
   ```

2. Reduce chunk size:
   ```python
   CHUNK_SIZE = 300
   ```

### Slow Processing

1. Enable GPU if available
2. First run is slower (downloads model), subsequent runs use cache
3. Check logs to identify bottlenecks

### Poor Comparison Results

1. Adjust thresholds based on your document type
2. Try a different embedding model
3. Adjust chunk size (smaller chunks = more granular comparison)

## Performance Notes

- **First run**: ~5-10 minutes for 100-page documents (downloads model)
- **Subsequent runs**: ~2-3 minutes (uses cached embeddings)
- **Memory usage**: ~2-4 GB RAM for typical documents
- **GPU speedup**: 3-5x faster with CUDA

## Limitations

- Requires text-based PDFs (scanned PDFs need OCR first)
- Embedding model must fit in RAM/VRAM
- Best for documents with structured content
- May struggle with tables and complex formatting

## Future Improvements

- [ ] OCR support for scanned PDFs
- [ ] Table-aware comparison
- [ ] Visual diff generation
- [ ] REST API interface
- [ ] Interactive HTML reports
- [ ] Support for Word documents

## License

MIT License - feel free to use and modify for your needs.

## Contributing

Issues and pull requests welcome!

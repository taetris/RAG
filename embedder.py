"""
Embedding generation with caching and batch processing.
"""
import logging
import pickle
import hashlib
from pathlib import Path
from typing import List, Optional
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate and cache document embeddings efficiently."""
    
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize embedding generator.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use ('cpu' or 'cuda')
            batch_size: Batch size for embedding generation
            cache_dir: Directory for caching embeddings
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Load model once and reuse
        logger.info(f"Loading embedding model: {model_name}")
        self.embedder = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device},
            encode_kwargs={'batch_size': batch_size}
        )
        logger.info("Model loaded successfully")
    
    def _get_cache_key(self, texts: List[str]) -> str:
        """Generate cache key from texts."""
        text_hash = hashlib.md5("".join(texts).encode()).hexdigest()
        return f"{self.model_name}_{text_hash}"
    
    def _load_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """Load embeddings from cache if available."""
        if not self.cache_dir:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    embeddings = pickle.load(f)
                logger.info(f"Loaded embeddings from cache: {cache_file}")
                return embeddings
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                return None
        return None
    
    def _save_to_cache(self, cache_key: str, embeddings: np.ndarray):
        """Save embeddings to cache."""
        if not self.cache_dir:
            return
        
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(embeddings, f)
            logger.info(f"Saved embeddings to cache: {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def generate_embeddings(
        self, 
        texts: List[str], 
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            use_cache: Whether to use caching
            
        Returns:
            Numpy array of embeddings (n_texts, embedding_dim)
        """
        if not texts:
            logger.warning("Empty text list provided")
            return np.array([])
        
        # Check cache
        if use_cache:
            cache_key = self._get_cache_key(texts)
            cached = self._load_from_cache(cache_key)
            if cached is not None:
                return cached
        
        # Generate embeddings in batches
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        try:
            embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                batch_embeddings = self.embedder.embed_documents(batch)
                embeddings.extend(batch_embeddings)
                
                if (i // self.batch_size + 1) % 10 == 0:
                    logger.info(f"Processed {i + len(batch)}/{len(texts)} texts")
            
            embeddings_array = np.array(embeddings)
            logger.info(f"Generated embeddings shape: {embeddings_array.shape}")
            
            # Save to cache
            if use_cache:
                self._save_to_cache(cache_key, embeddings_array)
            
            return embeddings_array
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def generate_for_chunks(
        self, 
        chunks: List[dict], 
        text_key: str = "text"
    ) -> np.ndarray:
        """
        Generate embeddings for chunk dictionaries.
        
        Args:
            chunks: List of chunk dictionaries
            text_key: Key to extract text from chunks
            
        Returns:
            Numpy array of embeddings
        """
        texts = [chunk[text_key] for chunk in chunks]
        return self.generate_embeddings(texts)
# storage/vector_store.py - IMPROVED VERSION

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import numpy as np
from datetime import datetime

class LegalVectorStore:
    """Production-ready vector store for legal compliance RAG"""
    
    def __init__(self, persist_directory="./chroma_db"):
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Separate collections for policy and regulation
        self.policy_collection = self._get_or_create_collection(
            name="icaew_policy_v1",
            metadata={
                "description": "ICAEW Internal Data Protection Policy",
                "version": "1.7",
                "date": "2024-08",
                "embedding_model": "nlpaueb/legal-bert-base-uncased",
                "created_at": datetime.now().isoformat()
            }
        )
        
        self.regulation_collection = self._get_or_create_collection(
            name="gdpr_regulation_v1",
            metadata={
                "description": "EU GDPR Regulation 2016/679",
                "embedding_model": "nlpaueb/legal-bert-base-uncased",
                "created_at": datetime.now().isoformat()
            }
        )
    
    def _get_or_create_collection(self, name: str, metadata: dict):
        """Get or create collection with metadata"""
        try:
            collection = self.client.get_collection(name=name)
            print(f"✓ Loaded existing collection: {name}")
        except:
            collection = self.client.create_collection(
                name=name,
                metadata={
                    **metadata,
                    "hnsw:space": "cosine",  # Use cosine similarity
                    "hnsw:construction_ef": 200,  # Better index quality
                    "hnsw:M": 16  # More connections = better recall
                }
            )
            print(f"✓ Created new collection: {name}")
        
        return collection
    
    def add_chunks(self, chunks: List[Dict], embeddings: np.ndarray, 
                   collection_type: str = "policy"):
        """Add chunks with embeddings to vector store"""
        
        collection = (self.policy_collection if collection_type == "policy" 
                     else self.regulation_collection)
        
        # Prepare data for ChromaDB
        ids = [self._generate_chunk_id(c) for c in chunks]
        documents = [c['text'] for c in chunks]
        
        # Enhanced metadata with hierarchical info
        metadatas = [
            {
                "section_id": c['section_id'],
                "section_title": c['section_title'],
                "doc_type": c['doc_type'],
                "level": c['level'],
                "parent_sections": ",".join(c['parent_sections']),  # Store as comma-separated
                # Add for advanced filtering
                "has_parents": len(c['parent_sections']) > 0,
                "is_leaf": c['level'] > 2,  # Adjust based on your hierarchy
                # For legal-specific queries
                "contains_obligation": self._detect_obligation(c['text']),
                "contains_right": self._detect_right(c['text']),
            }
            for c in chunks
        ]
        
        embeddings_list = embeddings.tolist()
        
        # Add to collection (ChromaDB handles duplicates by ID)
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings_list
        )
        
        print(f"✓ Added {len(chunks)} {collection_type} chunks to vector store")
    
    def _generate_chunk_id(self, chunk: Dict) -> str:
        """Generate stable chunk ID"""
        import hashlib
        
        # Use section_id + doc_type for stable ID
        content = f"{chunk['doc_type']}_{chunk['section_id']}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _detect_obligation(self, text: str) -> bool:
        """Detect if text contains legal obligations"""
        obligation_terms = ["shall", "must", "required to", "obliged to"]
        text_lower = text.lower()
        return any(term in text_lower for term in obligation_terms)
    
    def _detect_right(self, text: str) -> bool:
        """Detect if text describes data subject rights"""
        right_terms = ["right to", "may request", "entitled to", "access", "rectification", "erasure"]
        text_lower = text.lower()
        return any(term in text_lower for term in right_terms)
    
    def search(self, query_embedding: np.ndarray, collection_type: str = "policy",
               n_results: int = 10, filter_dict: Optional[Dict] = None) -> Dict:
        """
        Search for similar chunks
        
        Args:
            query_embedding: Query vector
            collection_type: "policy" or "regulation"
            n_results: Number of results to return
            filter_dict: Metadata filters, e.g., {"level": 1, "contains_obligation": True}
        """
        collection = (self.policy_collection if collection_type == "policy" 
                     else self.regulation_collection)
        
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=filter_dict  # e.g., {"level": {"$gte": 2}}
        )
        
        return results
    
    def get_stats(self):
        """Get collection statistics"""
        return {
            "policy_chunks": self.policy_collection.count(),
            "regulation_chunks": self.regulation_collection.count(),
            "total_chunks": self.policy_collection.count() + self.regulation_collection.count()
        }
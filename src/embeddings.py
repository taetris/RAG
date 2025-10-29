"""Embedding generation and vector store management."""
from typing import List
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from chunker import Chunk


class EmbeddingManager:
    """Manages embeddings and vector storage."""
    
    def __init__(self, model_name: str, persist_dir: str):
        self.model = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
    
    def create_collection(self, collection_name: str, chunks: List[Chunk]):
        """Create or recreate a collection with chunks."""
        # Delete if exists
        try:
            self.client.delete_collection(collection_name)
        except:
            pass
        
        collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Prepare data
        texts = [chunk.text for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        ids = [
            f"{chunk.doc_type}_{chunk.section_id}_{i}"
            for i, chunk in enumerate(chunks)
        ]

        metadatas = [
            {
                "section_id": chunk.section_id,
                "section_title": chunk.section_title,
                "doc_type": chunk.doc_type
            }
            for chunk in chunks
        ]
        
        # Add to collection
        collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            ids=ids,
            metadatas=metadatas
        )
        
        print(f"✓ Created collection '{collection_name}' with {len(chunks)} chunks")
        return collection
    
    def get_collection(self, collection_name: str):
        """Get existing collection."""
        return self.client.get_collection(collection_name)
    
    def similarity_search(
        self, 
        collection_name: str, 
        query: str, 
        top_k: int = 10
    ) -> List[dict]:
        """Search collection by semantic similarity."""
        collection = self.get_collection(collection_name)
        query_embedding = self.model.encode([query])[0]
        
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        # Format results
        formatted = []
        for i in range(len(results['ids'][0])):
            formatted.append({
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        
        return formatted
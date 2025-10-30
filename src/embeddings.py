# services/embed_documents.py

"""
Document embedding with hierarchical context preservation.
"""
import os
import json
import torch
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from storage.vector_store import LegalVectorStore
from tqdm import tqdm

class LegalEmbedder:
    """Embedder for legal documents with context preservation"""
    
    def __init__(self, model_name="nlpaueb/legal-bert-base-uncased"):
        print(f"📦 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Model loaded on {self.device}")
        
        # Asymmetric prompts for better retrieval
        self.doc_prefix = "Represent this legal document section: "
        self.query_prefix = "Find regulations about: "
    
    def create_contextual_text(self, chunk: Dict) -> str:
        """
        Build full hierarchical context for embedding
        
        This is CRITICAL for legal documents where context determines meaning
        """
        parts = []
        
        # 1. Document type context
        doc_type_label = "Policy" if chunk['doc_type'] == 'policy' else "Regulation"
        parts.append(f"[{doc_type_label}]")
        
        # 2. Hierarchical path (parent sections)
        if chunk['parent_sections']:
            parent_path = " → ".join(chunk['parent_sections'])
            parts.append(f"Context: {parent_path}")
        
        # 3. Current section identifier
        parts.append(f"Section {chunk['section_id']}: {chunk['section_title']}")
        
        # 4. Main content
        parts.append(chunk['text'])
        
        # 5. For GDPR, add recital references if available
        if chunk['doc_type'] == 'regulation':
            if 'related_recitals' in chunk and chunk['related_recitals']:
                recitals = ", ".join(map(str, chunk['related_recitals']))
                parts.append(f"Related Recitals: {recitals}")
        
        return "\n\n".join(parts)
    
    def validate_chunks(self, chunks: List[Dict]) -> List[str]:
        """
        Validate chunk quality before embedding
        Returns list of warnings/errors
        """
        issues = []
        
        for i, chunk in enumerate(chunks):
            # Check for required fields
            required_fields = ['section_id', 'section_title', 'text', 'doc_type', 'level', 'parent_sections']
            missing = [f for f in required_fields if f not in chunk]
            if missing:
                issues.append(f"Chunk {i}: Missing fields {missing}")
            
            # Check text length
            if 'text' in chunk and len(chunk['text'].strip()) < 50:
                issues.append(f"Chunk {i} ({chunk.get('section_id', '?')}): Text too short ({len(chunk['text'])} chars)")
            
            # Check for misidentified sections (your JSON issue)
            if chunk.get('doc_type') == 'regulation':
                text = chunk.get('text', '')
                section_id = chunk.get('section_id', '')
                
                # Recitals should not be labeled as Articles
                if section_id.startswith('Article') and text.strip().startswith('('):
                    issues.append(f"Chunk {i}: '{section_id}' appears to be a Recital, not an Article")
        
        return issues
    
    def embed_chunks(self, chunks: List[Dict], batch_size: int = 32,
                    show_progress: bool = True) -> np.ndarray:
        """
        Embed chunks with hierarchical context
        
        Args:
            chunks: List of chunk dictionaries
            batch_size: Encoding batch size (adjust based on GPU memory)
            show_progress: Show progress bar
        
        Returns:
            numpy array of embeddings (n_chunks, embedding_dim)
        """
        # Validate chunks first
        issues = self.validate_chunks(chunks)
        if issues:
            print("⚠️  Chunk validation warnings:")
            for issue in issues[:10]:  # Show first 10
                print(f"   {issue}")
            if len(issues) > 10:
                print(f"   ... and {len(issues) - 10} more issues")
            
            response = input("\n Continue embedding despite warnings? (y/n): ")
            if response.lower() != 'y':
                raise ValueError("Embedding cancelled due to validation issues")
        
        # Create contextual texts
        print(f"Preparing {len(chunks)} chunks with hierarchical context...")
        texts = [
            self.doc_prefix + self.create_contextual_text(c) 
            for c in tqdm(chunks, desc="Building context", disable=not show_progress)
        ]
        
        # Embed with progress bar
        print(f"🔢 Embedding {len(texts)} texts (batch_size={batch_size})...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # Normalize for cosine similarity
            device=self.device
        )
        
        print(f"✓ Generated embeddings: shape {embeddings.shape}")
        return embeddings
    
    def embed_query(self, query_text: str) -> np.ndarray:
        """Embed a search query (uses different prefix for asymmetric embeddings)"""
        prefixed_query = self.query_prefix + query_text
        
        embedding = self.model.encode(
            prefixed_query,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=self.device
        )
        
        return embedding


def load_chunks(file_path: str) -> List[Dict]:
    """Load chunks from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f"✓ Loaded {len(chunks)} chunks from {file_path}")
    return chunks


def main():
    """Main embedding pipeline"""
    
    # Configuration
    OUTPUT_DIR = "./outputs"
    POLICY_CHUNKS_FILE = os.path.join(OUTPUT_DIR, "policy_chunks.json")
    REGULATION_CHUNKS_FILE = os.path.join(OUTPUT_DIR, "regulation_chunks.json")
    
    # Initialize embedder and vector store
    embedder = LegalEmbedder(model_name="nlpaueb/legal-bert-base-uncased")
    vector_store = LegalVectorStore(persist_directory="./chroma_db")
    
    # Process policy chunks
    print("\n" + "="*60)
    print("EMBEDDING POLICY CHUNKS")
    print("="*60)
    policy_chunks = load_chunks(POLICY_CHUNKS_FILE)
    policy_embeddings = embedder.embed_chunks(policy_chunks, batch_size=32)
    vector_store.add_chunks(policy_chunks, policy_embeddings, collection_type="policy")
    
    # Process regulation chunks
    print("\n" + "="*60)
    print("EMBEDDING REGULATION CHUNKS")
    print("="*60)
    regulation_chunks = load_chunks(REGULATION_CHUNKS_FILE)
    regulation_embeddings = embedder.embed_chunks(regulation_chunks, batch_size=32)
    vector_store.add_chunks(regulation_chunks, regulation_embeddings, collection_type="regulation")
    
    # Print stats
    print("\n" + "="*60)
    print("EMBEDDING COMPLETE")
    print("="*60)
    stats = vector_store.get_stats()
    print(f"✓ Policy chunks: {stats['policy_chunks']}")
    print(f"✓ Regulation chunks: {stats['regulation_chunks']}")
    print(f"✓ Total indexed: {stats['total_chunks']}")
    
    # Test search
    print("\n" + "="*60)
    print("TESTING SEARCH")
    print("="*60)
    test_query = "data breach notification requirements"
    query_embedding = embedder.embed_query(test_query)
    
    print(f"\nQuery: '{test_query}'")
    print("\nTop 3 regulation matches:")
    results = vector_store.search(query_embedding, collection_type="regulation", n_results=3)
    
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ), 1):
        similarity = 1 - distance  # Convert distance to similarity
        print(f"\n{i}. {metadata['section_id']}: {metadata['section_title']}")
        print(f"   Similarity: {similarity:.3f}")
        print(f"   Preview: {doc[:150]}...")


if __name__ == "__main__":
    main()
"""Hybrid retrieval combining dense and sparse methods."""
from typing import List, Dict
from rank_bm25 import BM25Okapi
import numpy as np

from embeddings import EmbeddingManager


class HybridRetriever:
    """Combines dense (embedding) and sparse (BM25) retrieval."""
    
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        self.bm25_indices = {}
    
    def build_bm25_index(self, collection_name: str, documents: List[str]):
        """Build BM25 index for keyword search."""
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25_indices[collection_name] = {
            'index': BM25Okapi(tokenized_docs),
            'documents': documents
        }
    
    def bm25_search(self, collection_name: str, query: str, top_k: int = 10) -> List[Dict]:
        """Keyword-based search using BM25."""
        if collection_name not in self.bm25_indices:
            raise ValueError(f"No BM25 index for {collection_name}")
        
        index_data = self.bm25_indices[collection_name]
        bm25_index = index_data['index']
        documents = index_data['documents']
        
        tokenized_query = query.lower().split()
        scores = bm25_index.get_scores(tokenized_query)
        
        # Get top K indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only return relevant results
                results.append({
                    'text': documents[idx],
                    'score': float(scores[idx]),
                    'rank': len(results) + 1
                })
        
        return results
    
    def hybrid_search(
        self, 
        collection_name: str, 
        query: str, 
        top_k_dense: int = 10,
        top_k_bm25: int = 10,
        top_k_final: int = 5
    ) -> List[Dict]:
        """
        Hybrid search combining dense and sparse retrieval.
        Uses Reciprocal Rank Fusion for combining results.
        """
        # Dense retrieval
        dense_results = self.embedding_manager.similarity_search(
            collection_name, query, top_k_dense
        )
        
        # Sparse retrieval
        bm25_results = self.bm25_search(collection_name, query, top_k_bm25)
        
        # Reciprocal Rank Fusion
        # Score = sum(1 / (k + rank)) for each method
        k = 60  # Constant to prevent high ranks from dominating
        
        fused_scores = {}
        
        # Add dense scores
        for rank, result in enumerate(dense_results, 1):
            doc_id = result['text'][:50]  # Use text snippet as ID
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (k + rank)
        
        # Add BM25 scores
        for rank, result in enumerate(bm25_results, 1):
            doc_id = result['text'][:50]
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (k + rank)
        
        # Get top K by fused score
        sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Retrieve full results for top docs
        final_results = []
        for doc_id, score in sorted_docs[:top_k_final]:
            # Find full document from dense results
            for result in dense_results:
                if result['text'][:50] == doc_id:
                    result['fused_score'] = score
                    final_results.append(result)
                    break
        
        return final_results
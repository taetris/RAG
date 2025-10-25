"""
Bidirectional document comparison with semantic similarity.
"""
import logging
from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher

from structure_parser import StructuralElement

logger = logging.getLogger(__name__)


class DocumentComparator:
    """Compare two document versions using embeddings and text similarity."""
    
    def __init__(
        self,
        threshold_unchanged: float = 0.95,
        threshold_modified: float = 0.70,
        use_text_similarity: bool = True
    ):
        """
        Initialize comparator.
        
        Args:
            threshold_unchanged: Similarity threshold for unchanged content
            threshold_modified: Similarity threshold for modified content
            use_text_similarity: Whether to use text matching as first pass
        """
        self.threshold_unchanged = threshold_unchanged
        self.threshold_modified = threshold_modified
        self.use_text_similarity = use_text_similarity
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using SequenceMatcher."""
        return SequenceMatcher(None, text1, text2).ratio()
    
    def _find_best_matches(
        self,
        embeddings_v1: np.ndarray,
        embeddings_v2: np.ndarray,
        chunks_v1: List[dict],
        chunks_v2: List[dict],
        batch_size: int = 100
    ) -> List[Dict]:
        """
        Find best matches using batched similarity computation.
        
        Args:
            embeddings_v1: Embeddings for version 1
            embeddings_v2: Embeddings for version 2
            chunks_v1: Chunks from version 1
            chunks_v2: Chunks from version 2
            batch_size: Batch size for similarity computation
            
        Returns:
            List of match results
        """
        results = []
        n_chunks_v1 = len(chunks_v1)
        
        for i in range(0, n_chunks_v1, batch_size):
            end_idx = min(i + batch_size, n_chunks_v1)
            batch_v1 = embeddings_v1[i:end_idx]
            
            # Compute similarity for batch
            sim_matrix = cosine_similarity(batch_v1, embeddings_v2)
            
            for j, chunk_idx in enumerate(range(i, end_idx)):
                chunk_v1 = chunks_v1[chunk_idx]
                
                # Find best match
                best_match_idx = int(sim_matrix[j].argmax())
                best_score = float(sim_matrix[j][best_match_idx])
                
                chunk_v2 = chunks_v2[best_match_idx]
                
                # Optional: verify with text similarity
                if self.use_text_similarity and best_score > self.threshold_modified:
                    text_sim = self._text_similarity(
                        chunk_v1['text'], 
                        chunk_v2['text']
                    )
                    # Average semantic and text similarity
                    best_score = (best_score + text_sim) / 2
                
                results.append({
                    'v1_id': chunk_v1['id'],
                    'v2_id': chunk_v2['id'],
                    'v1_text': chunk_v1['text'],
                    'v2_text': chunk_v2['text'],
                    'similarity': best_score
                })
            
            if (end_idx // batch_size) % 10 == 0:
                logger.info(f"Processed {end_idx}/{n_chunks_v1} comparisons")
        
        return results
    
    def _classify_change(self, similarity: float) -> str:
        """Classify the type of change based on similarity."""
        if similarity >= self.threshold_unchanged:
            return "Unchanged"
        elif similarity >= self.threshold_modified:
            return "Modified"
        else:
            return "Removed or New"
    
    def compare(
        self,
        chunks_v1: List[dict],
        chunks_v2: List[dict],
        embeddings_v1: np.ndarray,
        embeddings_v2: np.ndarray
    ) -> Tuple[List[Dict], Dict]:
        """
        Perform bidirectional comparison.
        
        Args:
            chunks_v1: Chunks from version 1
            chunks_v2: Chunks from version 2
            embeddings_v1: Embeddings for version 1
            embeddings_v2: Embeddings for version 2
            
        Returns:
            Tuple of (detailed results, summary statistics)
        """
        logger.info("Starting bidirectional comparison...")
        
        # Forward comparison (v1 -> v2)
        forward_matches = self._find_best_matches(
            embeddings_v1, embeddings_v2, chunks_v1, chunks_v2
        )
        
        # Backward comparison (v2 -> v1) to find new sections
        backward_matches = self._find_best_matches(
            embeddings_v2, embeddings_v1, chunks_v2, chunks_v1
        )
        
        # Process forward matches
        results = []
        v2_matched = set()
        
        for match in forward_matches:
            status = self._classify_change(match['similarity'])
            v2_matched.add(match['v2_id'])
            
            results.append({
                'section_id': match['v1_id'] + 1,
                'status': status,
                'v1_snippet': match['v1_text'][:120].replace("\n", " "),
                'v2_snippet': match['v2_text'][:120].replace("\n", " "),
                'similarity': round(match['similarity'], 3),
                'change_type': 'forward'
            })
        
        # Find new sections in v2 (high backward match but not matched forward)
        for match in backward_matches:
            if match['v2_id'] not in v2_matched and match['similarity'] < self.threshold_modified:
                results.append({
                    'section_id': f"NEW_{match['v2_id'] + 1}",
                    'status': 'New in V2',
                    'v1_snippet': '',
                    'v2_snippet': match['v2_text'][:120].replace("\n", " "),
                    'similarity': 0.0,
                    'change_type': 'new'
                })
        
        # Calculate statistics
        stats = self._calculate_statistics(results)
        
        logger.info("Comparison complete")
        return results, stats
    
    def _calculate_statistics(self, results: List[Dict]) -> Dict:
        """Calculate summary statistics."""
        stats = {
            'total_sections': len(results),
            'unchanged': sum(1 for r in results if r['status'] == 'Unchanged'),
            'modified': sum(1 for r in results if r['status'] == 'Modified'),
            'removed_or_new': sum(1 for r in results if r['status'] == 'Removed or New'),
            'new_in_v2': sum(1 for r in results if r['status'] == 'New in V2'),
            'avg_similarity': np.mean([r['similarity'] for r in results if r['similarity'] > 0])
        }
        return stats
    
    def compare_structural(
        self,
        elements_v1: List[StructuralElement],
        elements_v2: List[StructuralElement],
        embeddings_v1: np.ndarray,
        embeddings_v2: np.ndarray
    ) -> Tuple[List[Dict], Dict]:
        """
        Compare documents using structural alignment.
        
        First tries to match by number (Article 5 -> Article 5),
        then falls back to semantic similarity.
        """
        results = []
        matched_v2 = set()
        
        # Build lookup for v2 elements by number
        v2_by_number = {elem.number: (i, elem) for i, elem in enumerate(elements_v2)}
        
        for i, elem_v1 in enumerate(elements_v1):
            # Try exact structural match first
            if elem_v1.number in v2_by_number:
                j, elem_v2 = v2_by_number[elem_v1.number]
                matched_v2.add(j)
                
                # Compare semantically even for matched structure
                similarity = float(cosine_similarity(
                    embeddings_v1[i:i+1], 
                    embeddings_v2[j:j+1]
                )[0][0])
                
                results.append({
                    'v1_element': elem_v1,
                    'v2_element': elem_v2,
                    'match_type': 'structural',
                    'similarity': similarity,
                    'status': self._classify_change(similarity)
                })
            else:
                # No structural match - find semantic match
                similarities = cosine_similarity(
                    embeddings_v1[i:i+1], 
                    embeddings_v2
                )[0]
                
                best_idx = int(similarities.argmax())
                best_score = float(similarities[best_idx])
                
                if best_score > self.threshold_modified:
                    matched_v2.add(best_idx)
                    results.append({
                        'v1_element': elem_v1,
                        'v2_element': elements_v2[best_idx],
                        'match_type': 'semantic',
                        'similarity': best_score,
                        'status': 'Renumbered/Moved'
                    })
                else:
                    results.append({
                        'v1_element': elem_v1,
                        'v2_element': None,
                        'match_type': 'none',
                        'similarity': 0.0,
                        'status': 'Removed'
                    })
        
        # Find new elements in v2
        for j, elem_v2 in enumerate(elements_v2):
            if j not in matched_v2:
                results.append({
                    'v1_element': None,
                    'v2_element': elem_v2,
                    'match_type': 'none',
                    'similarity': 0.0,
                    'status': 'New'
                })
        
        stats = self._calculate_structural_stats(results)
        return results, stats
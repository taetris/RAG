"""Main pipeline for compliance RAG system."""
import json
from pathlib import Path
from typing import List, Dict

from config import (
    ICAEW_FILE, GDPR_FILE, EMBEDDING_MODEL, LLM_MODEL,
    CHROMA_PERSIST_DIR, POLICY_COLLECTION, GDPR_COLLECTION,
    TOP_K_DENSE, TOP_K_BM25, TOP_K_FINAL
)
from chunker import create_chunker, Chunk
from embeddings import EmbeddingManager
from retriever import HybridRetriever
from compliance_assessor import ComplianceAssessor


class ComplianceRAGPipeline:
    """End-to-end pipeline for compliance assessment."""
    
    def __init__(self):
        self.chunker = create_chunker()
        self.embedding_manager = EmbeddingManager(EMBEDDING_MODEL, CHROMA_PERSIST_DIR)
        self.retriever = HybridRetriever(self.embedding_manager)
        self.assessor = ComplianceAssessor(LLM_MODEL)
        
        self.icaew_chunks = []
        self.gdpr_chunks = []
    
    def load_and_chunk_documents(self):
        """Load and chunk both documents."""
        print("📄 Loading documents...")
        
        # Load ICAEW policy
        with open(ICAEW_FILE, 'r', encoding='utf-8') as f:
            icaew_text = f.read()
        self.icaew_chunks = self.chunker.chunk_document(icaew_text, "icaew")
        print(f"  ✓ ICAEW: {len(self.icaew_chunks)} chunks")
        
        # Load GDPR
        with open(GDPR_FILE, 'r', encoding='utf-8') as f:
            gdpr_text = f.read()
        self.gdpr_chunks = self.chunker.chunk_document(gdpr_text, "gdpr")
        print(f"  ✓ GDPR: {len(self.gdpr_chunks)} chunks")
    
    def build_vector_stores(self):
        """Create vector stores for both documents."""
        print("\n🔢 Building vector stores...")
        
        self.embedding_manager.create_collection(POLICY_COLLECTION, self.icaew_chunks)
        self.embedding_manager.create_collection(GDPR_COLLECTION, self.gdpr_chunks)
        
        # Build BM25 indices
        print("\n🔍 Building BM25 indices...")
        self.retriever.build_bm25_index(
            GDPR_COLLECTION,
            [chunk.text for chunk in self.gdpr_chunks]
        )
    
    def link_policy_to_gdpr(self, policy_chunk: Dict) -> List[Dict]:
        """Find related GDPR articles for a policy section."""
        # Hybrid retrieval to find candidate articles
        candidates = self.retriever.hybrid_search(
            GDPR_COLLECTION,
            policy_chunk['text'],
            top_k_dense=TOP_K_DENSE,
            top_k_bm25=TOP_K_BM25,
            top_k_final=TOP_K_FINAL
        )
        
        # LLM verification of relationships
        relationships = self.assessor.find_related_articles(policy_chunk, candidates)
        
        # Get full article data for related articles
        related_articles = []
        for rel in relationships:
            if rel['relationship'] in ['Direct', 'Indirect']:
                # Find the full article
                for candidate in candidates:
                    if rel['article'] in candidate['metadata']['section_id']:
                        related_articles.append({
                            'article': candidate,
                            'relationship': rel['relationship'],
                            'reasoning': rel['reasoning']
                        })
                        break
        
        return related_articles
    
    def assess_full_compliance(self) -> List[Dict]:
        """Run full compliance assessment."""
        print("\n⚖️  Running compliance assessment...")
        
        results = []
        
        for i, policy_chunk in enumerate(self.icaew_chunks, 1):
            print(f"\n[{i}/{len(self.icaew_chunks)}] Assessing {policy_chunk.section_id}...")
            
            # Convert Chunk to dict for compatibility
            policy_dict = {
                'text': policy_chunk.text,
                'metadata': {
                    'section_id': policy_chunk.section_id,
                    'section_title': policy_chunk.section_title,
                    'doc_type': policy_chunk.doc_type
                }
            }
            
            # Find related GDPR articles
            related_articles = self.link_policy_to_gdpr(policy_dict)
            print(f"  Found {len(related_articles)} related GDPR articles")
            
            # Assess compliance for each related article
            for article_data in related_articles:
                article = article_data['article']
                print(f"    Comparing with {article['metadata']['section_id']}...")
                
                assessment = self.assessor.assess_compliance(policy_dict, article)
                assessment['link_relationship'] = article_data['relationship']
                assessment['link_reasoning'] = article_data['reasoning']
                
                results.append(assessment)
        
        return results
    
    def generate_summary(self, results: List[Dict]) -> Dict:
        """Generate summary of compliance assessment."""
        summary = {
            "total_assessments": len(results),
            "full_compliance": 0,
            "partial_compliance": 0,
            "non_compliant": 0,
            "critical_gaps": [],
            "recommendations": []
        }
        
        for result in results:
            status = result['status']
            if status == "Full":
                summary['full_compliance'] += 1
            elif status == "Partial":
                summary['partial_compliance'] += 1
            elif status == "Non-compliant":
                summary['non_compliant'] += 1
                summary['critical_gaps'].append({
                    "policy_section": result['policy_section'],
                    "gdpr_article": result['gdpr_reference'],
                    "gaps": result['gaps']
                })
        
        # Calculate compliance percentage
        if summary['total_assessments'] > 0:
            summary['compliance_percentage'] = (
                (summary['full_compliance'] + 0.5 * summary['partial_compliance']) 
                / summary['total_assessments'] * 100
            )
        
        return summary
    
    def run(self, output_file: str = "compliance_report.json"):
        """Run the full pipeline."""
        print("=" * 60)
        print("COMPLIANCE RAG PIPELINE")
        print("=" * 60)
        
        # Step 1: Load and chunk
        self.load_and_chunk_documents()
        
        # Step 2: Build vector stores
        self.build_vector_stores()
        
        # Step 3: Run assessment
        results = self.assess_full_compliance()
        
        # Step 4: Generate summary
        summary = self.generate_summary(results)
        
        # Step 5: Save results
        output = {
            "summary": summary,
            "detailed_results": results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        
        print("\n" + "=" * 60)
        print("ASSESSMENT COMPLETE")
        print("=" * 60)
        print(f"\n📊 Summary:")
        print(f"  Total assessments: {summary['total_assessments']}")
        print(f"  Full compliance: {summary['full_compliance']}")
        print(f"  Partial compliance: {summary['partial_compliance']}")
        print(f"  Non-compliant: {summary['non_compliant']}")
        print(f"  Compliance score: {summary.get('compliance_percentage', 0):.1f}%")
        print(f"\n💾 Full report saved to: {output_file}")
        
        return output


def main():
    """Run the pipeline."""
    pipeline = ComplianceRAGPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
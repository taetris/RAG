"""
Generate comparison reports in various formats.
"""
import json
import logging
import csv
from pathlib import Path
from typing import List, Dict
import pandas as pd

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate and save comparison reports."""
    
    def __init__(self, output_dir: str):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_csv(
        self, 
        results: List[Dict], 
        filename: str = "comparison_report.csv"
    ) -> Path:
        """
        Save results to CSV using pandas for proper escaping.
        
        Args:
            results: List of comparison results
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename
        
        try:
            # Convert to DataFrame for better handling
            df = pd.DataFrame(results)
            
            # Reorder columns for better readability
            column_order = [
                'section_id', 'status', 'similarity', 
                'v1_snippet', 'v2_snippet', 'change_type'
            ]
            df = df[column_order]
            
            # Save to CSV
            df.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"CSV report saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving CSV: {e}")
            raise
    
    def save_detailed_report(
        self,
        results: List[Dict],
        stats: Dict,
        filename: str = "detailed_report.txt"
    ) -> Path:
        """
        Save detailed text report with full chunks.
        
        Args:
            results: List of comparison results
            stats: Summary statistics
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                # Write header
                f.write("=" * 80 + "\n")
                f.write("DOCUMENT COMPARISON REPORT\n")
                f.write("=" * 80 + "\n\n")
                
                # Write statistics
                f.write("SUMMARY STATISTICS\n")
                f.write("-" * 80 + "\n")
                f.write(f"Total Sections: {stats['total_sections']}\n")
                f.write(f"Unchanged: {stats['unchanged']} ({stats['unchanged']/stats['total_sections']*100:.1f}%)\n")
                f.write(f"Modified: {stats['modified']} ({stats['modified']/stats['total_sections']*100:.1f}%)\n")
                f.write(f"Removed or New: {stats['removed_or_new']}\n")
                f.write(f"New in V2: {stats['new_in_v2']}\n")
                f.write(f"Average Similarity: {stats['avg_similarity']:.3f}\n\n")
                
                # Group by status
                for status in ['Unchanged', 'Modified', 'Removed or New', 'New in V2']:
                    status_results = [r for r in results if r['status'] == status]
                    if not status_results:
                        continue
                    
                    f.write("=" * 80 + "\n")
                    f.write(f"{status.upper()} SECTIONS ({len(status_results)})\n")
                    f.write("=" * 80 + "\n\n")
                    
                    for result in status_results:
                        f.write(f"Section ID: {result['section_id']}\n")
                        f.write(f"Similarity: {result['similarity']:.3f}\n")
                        f.write(f"\nVersion 1:\n{result['v1_snippet']}\n")
                        f.write(f"\nVersion 2:\n{result['v2_snippet']}\n")
                        f.write("-" * 80 + "\n\n")
            
            logger.info(f"Detailed report saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving detailed report: {e}")
            raise
    
    def save_summary(
        self,
        stats: Dict,
        filename: str = "summary.txt"
    ) -> Path:
        """
        Save quick summary report.
        
        Args:
            stats: Summary statistics
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("COMPARISON SUMMARY\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total Sections: {stats['total_sections']}\n")
                f.write(f"Unchanged: {stats['unchanged']}\n")
                f.write(f"Modified: {stats['modified']}\n")
                f.write(f"Removed or New: {stats['removed_or_new']}\n")
                f.write(f"New in V2: {stats['new_in_v2']}\n")
                f.write(f"Average Similarity: {stats['avg_similarity']:.3f}\n")
            
            logger.info(f"Summary saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving summary: {e}")
            raise
    
    def generate_all_reports(
        self,
        results: List[Dict],
        stats: Dict
    ) -> Dict[str, Path]:
        """
        Generate all report types.
        
        Args:
            results: List of comparison results
            stats: Summary statistics
            
        Returns:
            Dictionary of report type to file path
        """
        reports = {}
        
        reports['csv'] = self.save_csv(results)
        reports['detailed'] = self.save_detailed_report(results, stats)
        reports['summary'] = self.save_summary(stats)
        reports['interpreted'] = self.save_interpreted_changes(results)

        logger.info(f"Generated {len(reports)} reports")
        return reports
    
    def save_interpreted_changes(self, results):
        """Save only interpreted (AI-analyzed) changes into a separate report."""
        import json
        from pathlib import Path

        interpreted = [
            {
                "v1_snippet": r.get("v1_snippet", ""),
                "v2_snippet": r.get("v2_snippet", ""),
                "similarity": r.get("similarity"),
                "change_type": r.get("change_type", ""),
                "summary": r.get("summary", "")
            }
            for r in results
            if "change_type" in r and r["change_type"] not in ("No Significant Change", "Error", "Unclear")
        ]

        if not interpreted:
            logger.warning("No interpreted changes found to save.")
            return None

        output_path = Path(self.output_dir) / "interpreted_changes.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(interpreted, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved interpreted changes to {output_path}")
        return output_path

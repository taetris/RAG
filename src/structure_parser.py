"""
Document structure parsing for legal/regulatory documents.
"""
import re
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class StructuralElement:
    """Represents a structural element in the document."""
    type: str 
    number: str  
    title: Optional[str]
    content: str
    start_pos: int
    end_pos: int
    level: int  # hierarchy depth
    parent_number: Optional[str]
    
class StructureParser:
    """Parse document structure from extracted text."""
    
    def __init__(self):
        # Common patterns for legal/regulatory documents
        self.patterns = {
                        'article': [
                            r'^\s*Article\s+(\d+)\s*(.*?)$',
                            r'^\s*ARTICLE\s+([IVXLCDM]+)\s*(.*?)$',
                            r'^\s*Art\.\s*(\d+)\s*(.*?)$',
                        ],
                        'section': [
                            r'^\s*Section\s+([\d\.]+)\s*(.*?)$',
                            r'^\s*§\s*([\d\.]+)\s*(.*?)$',
                        ],
                        'clause': [
                            r'^\s*\(?([a-z0-9\.]+)\)?\s*(.*?)$',
                        ]
                    }

    
    def parse(self, text: str) -> List[StructuralElement]:
        """
        Parse document into structural elements.
        
        Returns hierarchical structure with proper nesting.
        """
        elements = []
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        
        # First pass: identify all structural markers
        for i, line in enumerate(lines):
            element = self._identify_element(line, i)
            if element:
                elements.append(element)
        
        # Second pass: extract content between markers
        for i, element in enumerate(elements):
            next_start = elements[i+1].start_pos if i+1 < len(elements) else len(text)
            element.content = self._extract_content(text, element.start_pos, next_start)
        
        # Third pass: establish hierarchy
        elements = self._build_hierarchy(elements)
        
        logger.info(f"Parsed {len(elements)} structural elements")
        return elements
    
    def _identify_element(self, line: str, line_num: int) -> Optional[StructuralElement]:
        """Identify if line is a structural marker."""
        for elem_type, patterns in self.patterns.items():
            for pattern in patterns:
                match = re.match(pattern, line.strip(), re.IGNORECASE)
                if match:
                    number = match.group(1)
                    title = match.group(2) if len(match.groups()) > 1 else None
                    
                    return StructuralElement(
                        type=elem_type,
                        number=number,
                        title=title,
                        content="",
                        start_pos=line_num,
                        end_pos=-1,
                        level=self._calculate_level(number),
                        parent_number=self._get_parent_number(number)
                    )
        return None
    
    def _calculate_level(self, number: str) -> int:
        """Calculate hierarchy level from numbering."""
        if '.' in number:
            return number.count('.') + 1
        return 0
    
    def _get_parent_number(self, number: str) -> Optional[str]:
        """Get parent element number."""
        if '.' in number:
            return number.rsplit('.', 1)[0]
        return None
    
    def _extract_content(self, text: str, start: int, end: int) -> str:
        """Extract content between structural markers."""
        lines = text.split('\n')
        content_lines = lines[start:end]
        # Remove the header line
        if content_lines:
            content_lines = content_lines[1:]
        return '\n'.join(content_lines).strip()
    
    def _build_hierarchy(self, elements: List[StructuralElement]) -> List[StructuralElement]:
        """Establish parent-child relationships."""
        # Implementation depends on your document structure
        return elements
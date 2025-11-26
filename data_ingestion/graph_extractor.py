"""
Graph Edge Extractor for Atlas-Hyperion v3.0
Extracts references between documents/sections for multi-hop retrieval.

Extracts edges like:
- "Article 5" -> "Article 2" (references)
- "Section 3" -> "Section 1" (refers to)
- "voir Article X" -> "Article X" (French cross-reference)
"""
import re
import logging
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GraphEdge:
    """Represents a directed edge in the knowledge graph."""
    source_id: str
    target_ref: str  # e.g., "Article 5", "Section 2"
    edge_type: str   # e.g., "references", "defines", "exception_of"
    context: str     # Text snippet where reference was found
    

@dataclass
class ExtractedReferences:
    """All references extracted from a text chunk."""
    edges: List[GraphEdge]
    article_refs: List[str]  # ["Article 5", "Article 2"]
    section_refs: List[str]  # ["Section 1", "Section 3"]
    definition_refs: List[str]  # ["term X", "concept Y"]
    external_refs: List[str]  # ["Loi 17-99", "Décret 2.06.574"]
    

class GraphExtractor:
    """
    Extracts graph edges from document text.
    
    Supports:
    - Article references (Article X, Art. X)
    - Section references (Section X, § X)
    - Chapter references (Chapitre X)
    - Cross-references (voir, cf., tel que défini)
    - Legal references (Loi X, Décret X, Code X)
    """
    
    # Reference patterns (French/English)
    PATTERNS = {
        "article": [
            r"[Aa]rticle\s+(\d+(?:\s*bis)?(?:\s*ter)?)",
            r"[Aa]rt\.\s*(\d+(?:\s*bis)?(?:\s*ter)?)",
            r"[Aa]rticles?\s+(\d+)\s*(?:à|et|,)\s*(\d+)",  # Article 5 à 10
        ],
        "section": [
            r"[Ss]ection\s+(\d+(?:\.\d+)?)",
            r"§\s*(\d+(?:\.\d+)?)",
        ],
        "chapter": [
            r"[Cc]hapitre\s+(\d+|[IVX]+)",
            r"[Cc]hap\.\s*(\d+|[IVX]+)",
        ],
        "title": [
            r"[Tt]itre\s+(\d+|[IVX]+)",
        ],
        "law": [
            r"[Ll]oi\s+(?:n[°o]?\s*)?(\d+[-–]\d+)",
            r"[Dd]écret\s+(?:n[°o]?\s*)?(\d+[-–]?\d*[-–]?\d*)",
            r"[Cc]ode\s+(?:de\s+)?(.+?)(?:\s*,|\s*\.|\s*;|$)",
        ],
        "cross_reference": [
            r"(?:voir|cf\.|conformément à|tel que (?:défini|prévu|mentionné) (?:à|dans))\s+(?:l[ea]?\s+)?([^,.\n]+)",
            r"(?:see|as defined in|pursuant to|refer to)\s+([^,.\n]+)",
        ],
        "alinea": [
            r"alinéa\s+(\d+)",
            r"al\.\s*(\d+)",
        ],
    }
    
    # Edge type mappings
    EDGE_TYPES = {
        "article": "references",
        "section": "references",
        "chapter": "references",
        "title": "references",
        "law": "cites_law",
        "cross_reference": "refers_to",
        "alinea": "references",
    }
    
    def __init__(self):
        """Initialize graph extractor with compiled patterns."""
        self._compiled_patterns: Dict[str, List[re.Pattern]] = {}
        for ref_type, patterns in self.PATTERNS.items():
            self._compiled_patterns[ref_type] = [
                re.compile(p, re.IGNORECASE | re.MULTILINE) 
                for p in patterns
            ]
        logger.info("GraphExtractor initialized")
    
    def extract(self, text: str, source_id: str = "") -> ExtractedReferences:
        """
        Extract all references from text.
        
        Args:
            text: Text to extract references from
            source_id: ID of the source chunk/document
            
        Returns:
            ExtractedReferences with all found references
        """
        edges = []
        article_refs = set()
        section_refs = set()
        definition_refs = set()
        external_refs = set()
        
        for ref_type, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    # Get context (surrounding text)
                    start = max(0, match.start() - 30)
                    end = min(len(text), match.end() + 30)
                    context = text[start:end].strip()
                    
                    # Handle different match groups
                    if ref_type == "article" and len(match.groups()) > 1 and match.group(2):
                        # Range: Article 5 à 10
                        try:
                            start_num = int(match.group(1))
                            end_num = int(match.group(2))
                            for num in range(start_num, end_num + 1):
                                ref = f"Article {num}"
                                article_refs.add(ref)
                                edges.append(GraphEdge(
                                    source_id=source_id,
                                    target_ref=ref,
                                    edge_type="references_range",
                                    context=context
                                ))
                        except ValueError:
                            pass
                    else:
                        # Single reference
                        ref_value = match.group(1)
                        
                        # Normalize reference
                        if ref_type == "article":
                            ref = f"Article {ref_value}"
                            article_refs.add(ref)
                        elif ref_type == "section":
                            ref = f"Section {ref_value}"
                            section_refs.add(ref)
                        elif ref_type == "chapter":
                            ref = f"Chapitre {ref_value}"
                            section_refs.add(ref)
                        elif ref_type == "title":
                            ref = f"Titre {ref_value}"
                            section_refs.add(ref)
                        elif ref_type == "law":
                            ref = match.group(0).strip()
                            external_refs.add(ref)
                        elif ref_type == "alinea":
                            ref = f"Alinéa {ref_value}"
                            section_refs.add(ref)
                        else:
                            ref = ref_value.strip()
                            if ref:
                                definition_refs.add(ref)
                        
                        edges.append(GraphEdge(
                            source_id=source_id,
                            target_ref=ref,
                            edge_type=self.EDGE_TYPES.get(ref_type, "references"),
                            context=context
                        ))
        
        return ExtractedReferences(
            edges=edges,
            article_refs=sorted(list(article_refs)),
            section_refs=sorted(list(section_refs)),
            definition_refs=sorted(list(definition_refs)),
            external_refs=sorted(list(external_refs))
        )
    
    def extract_edges_simple(self, text: str) -> List[str]:
        """
        Simple extraction returning just reference strings.
        
        Useful for storing in vector payload.
        
        Args:
            text: Text to extract from
            
        Returns:
            List of reference strings ["Article 5", "Section 2", ...]
        """
        refs = self.extract(text)
        all_refs = (
            refs.article_refs + 
            refs.section_refs + 
            refs.definition_refs
        )
        return all_refs
    
    def get_edge_ids(self, text: str, id_mapping: Dict[str, str] = None) -> List[str]:
        """
        Get edge target IDs for storage in vector payload.
        
        Args:
            text: Text to extract from
            id_mapping: Optional mapping of references to chunk IDs
            
        Returns:
            List of target IDs or reference strings
        """
        refs = self.extract_edges_simple(text)
        
        if id_mapping:
            return [id_mapping.get(ref, ref) for ref in refs]
        
        return refs
    
    def build_graph(
        self,
        chunks: List[Dict[str, str]]
    ) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
        """
        Build graph from list of chunks.
        
        Args:
            chunks: List of chunks with 'id', 'text', 'header_path'
            
        Returns:
            Tuple of (adjacency list, reference to ID mapping)
        """
        adjacency: Dict[str, List[str]] = {}
        ref_to_id: Dict[str, str] = {}
        
        # First pass: build reference to ID mapping
        for chunk in chunks:
            chunk_id = chunk.get("id", "")
            header_path = chunk.get("header_path", "")
            
            # Extract the main reference from header_path
            # e.g., "Section 1 > Article 5" -> "Article 5"
            if header_path:
                parts = header_path.split(" > ")
                for part in parts:
                    part = part.strip()
                    # Normalize to match extraction patterns
                    if any(p in part.lower() for p in ["article", "section", "chapitre", "titre"]):
                        ref_to_id[part] = chunk_id
        
        # Second pass: build adjacency list
        for chunk in chunks:
            chunk_id = chunk.get("id", "")
            text = chunk.get("text", "")
            
            refs = self.extract_edges_simple(text)
            targets = []
            
            for ref in refs:
                # Try to find the target ID
                target_id = ref_to_id.get(ref)
                if target_id and target_id != chunk_id:  # No self-loops
                    targets.append(target_id)
            
            if targets:
                adjacency[chunk_id] = list(set(targets))
        
        return adjacency, ref_to_id


def get_graph_extractor() -> GraphExtractor:
    """Factory function to get GraphExtractor instance."""
    return GraphExtractor()


# Example usage
if __name__ == "__main__":
    extractor = GraphExtractor()
    
    sample_text = """
    Conformément à l'Article 5, les dispositions suivantes s'appliquent.
    Tel que défini dans la Section 2, le terme "assurance" désigne...
    Voir également l'Article 10 et l'Article 12 pour les exceptions.
    Cette disposition est conforme à la Loi 17-99 relative au code des assurances.
    """
    
    refs = extractor.extract(sample_text, source_id="chunk_001")
    
    print("Extracted References:")
    print(f"  Articles: {refs.article_refs}")
    print(f"  Sections: {refs.section_refs}")
    print(f"  External: {refs.external_refs}")
    print(f"\nEdges:")
    for edge in refs.edges:
        print(f"  {edge.source_id} --[{edge.edge_type}]--> {edge.target_ref}")


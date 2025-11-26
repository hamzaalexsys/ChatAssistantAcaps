"""
Markdown Parser for Atlas-RAG
Parses Markdown files with YAML frontmatter and splits by headers.
"""
import os
import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime

import frontmatter
from slugify import slugify

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a chunk of a document with metadata."""
    text: str
    file_name: str
    header_path: str
    url_slug: str
    base_url: str
    last_updated: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Atlas-Hyperion v3.0: Contextual Crystallization fields
    crystallized_text: str = ""  # Context-enriched version of text
    summary: str = ""  # Ultra-short key-fact summary
    # Atlas-Hyperion v3.0: Graph fields for multi-hop retrieval
    edges: List[str] = field(default_factory=list)  # References to other chunks
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "text": self.text,
            "file_name": self.file_name,
            "header_path": self.header_path,
            "url_slug": self.url_slug,
            "base_url": self.base_url,
            "last_updated": self.last_updated,
            "chunk_index": self.chunk_index,
            "crystallized_text": self.crystallized_text,
            "summary": self.summary,
            "edges": self.edges,
            **self.metadata
        }


class MarkdownParser:
    """
    Parser for Markdown documents with YAML frontmatter.
    Splits documents by ATX headers (#, ##, ###, etc.)
    """
    
    # Regex pattern for ATX headers
    HEADER_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    
    def __init__(self, default_base_url: str = ""):
        """
        Initialize the parser.
        
        Args:
            default_base_url: Default base URL if not specified in frontmatter
        """
        self.default_base_url = default_base_url
    
    def parse_file(self, file_path: str) -> List[DocumentChunk]:
        """
        Parse a single Markdown file into chunks.
        
        Args:
            file_path: Path to the Markdown file
            
        Returns:
            List of DocumentChunk objects
        """
        logger.info(f"Parsing file: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse frontmatter
        post = frontmatter.loads(content)
        metadata = dict(post.metadata)
        body = post.content
        
        # Extract base_url from frontmatter or use default
        # If no base_url is provided, URLs will be empty (not fake)
        base_url = metadata.pop('base_url', self.default_base_url) or ""
        title = metadata.pop('title', os.path.basename(file_path))
        
        # Optional: direct source URL that overrides all generated URLs
        source_url = metadata.pop('source_url', None)
        
        # Get file info
        file_name = os.path.basename(file_path)
        last_updated = datetime.fromtimestamp(
            os.path.getmtime(file_path)
        ).isoformat()
        
        # Split by headers
        chunks = self._split_by_headers(body)
        
        # Create DocumentChunk objects
        result = []
        for idx, (header_path, text) in enumerate(chunks):
            if not text.strip():
                continue
                
            # Generate URL slug from the last header in the path
            # Only generate if base_url is provided
            if source_url:
                # Use direct source URL if provided
                url_slug = source_url
            elif base_url:
                last_header = header_path.split(' > ')[-1] if header_path else title
                url_slug = self._generate_slug(base_url, last_header)
            else:
                # No URL available - leave empty
                url_slug = ""
            
            chunk = DocumentChunk(
                text=text.strip(),
                file_name=file_name,
                header_path=header_path or title,
                url_slug=url_slug,
                base_url=base_url,
                last_updated=last_updated,
                chunk_index=idx,
                metadata=metadata.copy()
            )
            result.append(chunk)
        
        logger.info(f"Parsed {len(result)} chunks from {file_name}")
        return result
    
    def _split_by_headers(self, content: str) -> List[tuple]:
        """
        Split content by ATX headers, maintaining hierarchy.
        
        Args:
            content: Markdown content
            
        Returns:
            List of (header_path, text) tuples
        """
        chunks = []
        header_stack = []  # [(level, title), ...]
        current_text = []
        
        lines = content.split('\n')
        
        for line in lines:
            match = self.HEADER_PATTERN.match(line)
            
            if match:
                # Save previous section if exists
                if current_text:
                    header_path = self._build_header_path(header_stack)
                    chunks.append((header_path, '\n'.join(current_text)))
                    current_text = []
                
                # Process new header
                level = len(match.group(1))
                title = match.group(2).strip()
                
                # Pop headers of same or lower level
                while header_stack and header_stack[-1][0] >= level:
                    header_stack.pop()
                
                header_stack.append((level, title))
            else:
                current_text.append(line)
        
        # Don't forget the last section
        if current_text:
            header_path = self._build_header_path(header_stack)
            chunks.append((header_path, '\n'.join(current_text)))
        
        return chunks
    
    def _build_header_path(self, header_stack: List[tuple]) -> str:
        """Build hierarchical header path string."""
        if not header_stack:
            return ""
        return " > ".join([h[1] for h in header_stack])
    
    def _generate_slug(self, base_url: str, header: str) -> str:
        """
        Generate a URL slug from base_url and header.
        
        Args:
            base_url: The base URL from frontmatter
            header: The header text to slugify
            
        Returns:
            Complete URL with anchor
        """
        slug = slugify(header, lowercase=True)
        if base_url:
            return f"{base_url.rstrip('/')}#{slug}"
        return f"#{slug}"
    
    def parse_directory(self, directory: str, pattern: str = "*.md") -> List[DocumentChunk]:
        """
        Parse all Markdown files in a directory.
        
        Args:
            directory: Path to directory containing Markdown files
            pattern: Glob pattern for files (default: *.md)
            
        Returns:
            List of all DocumentChunk objects
        """
        import glob
        
        if not os.path.isdir(directory):
            raise NotADirectoryError(f"Not a directory: {directory}")
        
        all_chunks = []
        file_pattern = os.path.join(directory, pattern)
        
        for file_path in glob.glob(file_pattern):
            try:
                chunks = self.parse_file(file_path)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error parsing {file_path}: {e}")
                continue
        
        logger.info(f"Total chunks from directory: {len(all_chunks)}")
        return all_chunks


def parse_markdown(file_path: str, base_url: str = "") -> List[DocumentChunk]:
    """
    Convenience function to parse a single Markdown file.
    
    Args:
        file_path: Path to Markdown file
        base_url: Default base URL
        
    Returns:
        List of DocumentChunk objects
    """
    parser = MarkdownParser(default_base_url=base_url)
    return parser.parse_file(file_path)


def parse_directory(directory: str, base_url: str = "") -> List[DocumentChunk]:
    """
    Convenience function to parse a directory of Markdown files.
    
    Args:
        directory: Path to directory
        base_url: Default base URL
        
    Returns:
        List of DocumentChunk objects
    """
    parser = MarkdownParser(default_base_url=base_url)
    return parser.parse_directory(directory)


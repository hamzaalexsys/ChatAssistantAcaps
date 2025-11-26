"""
Contextual Crystallization for Atlas-Hyperion v3.0
Uses SLM to rewrite chunks into standalone, context-aware "Knowledge Crystals".

Based on Anthropic's Contextual Retrieval (Sept 2024).
"""
import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CrystallizedChunk:
    """A chunk enriched with global context."""
    original_text: str
    crystallized_text: str
    doc_title: str
    header_path: str
    summary: str  # Ultra-short key-fact summary (1-2 sentences)
    

class ChunkContextualizer:
    """
    Rewrites document chunks to be standalone using an SLM.
    
    Each chunk is enriched with:
    - Document title and section information
    - Referenced articles/definitions
    - Key context needed for understanding
    """
    
    CRYSTALLIZATION_PROMPT = """You are a document assistant. Your task is to rewrite the following chunk to be self-contained.

Document: {doc_title}
Section: {header_path}

Original chunk:
{chunk_text}

Rules:
1. Prepend the document and section context naturally
2. Keep all factual information intact
3. Make the text understandable without reading the full document
4. Be concise - don't add unnecessary information
5. If the chunk references other articles/sections, mention them explicitly
6. Keep the same language as the original (French or English)

Rewritten chunk:"""

    SUMMARY_PROMPT = """Extract the key fact from this text in 1-2 sentences. Be extremely concise.

Text:
{text}

Key fact:"""

    def __init__(
        self,
        llm_client = None,
        model: str = "qwen2.5:7b",
        use_mock: bool = False,
        max_chunk_length: int = 2000
    ):
        """
        Initialize contextualizer.
        
        Args:
            llm_client: OpenAI-compatible client for LLM calls
            model: Model name to use
            use_mock: Use mock responses for testing
            max_chunk_length: Maximum chunk length to process
        """
        self.llm_client = llm_client
        self.model = model
        self.use_mock = use_mock
        self.max_chunk_length = max_chunk_length
        
        logger.info(f"ChunkContextualizer initialized (mock={use_mock})")
    
    def _call_llm(self, prompt: str, max_tokens: int = 500) -> str:
        """Call LLM with prompt."""
        if self.use_mock:
            return self._mock_response(prompt)
        
        if self.llm_client is None:
            logger.warning("No LLM client provided, returning original text")
            return ""
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that rewrites document chunks to be self-contained."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistency
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return ""
    
    def _mock_response(self, prompt: str) -> str:
        """Generate mock response for testing."""
        # Extract document context from prompt
        doc_match = re.search(r'Document: (.+)', prompt)
        section_match = re.search(r'Section: (.+)', prompt)
        chunk_match = re.search(r'Original chunk:\n(.+?)(?:\n\nRules:|$)', prompt, re.DOTALL)
        
        doc_title = doc_match.group(1).strip() if doc_match else "Unknown Document"
        section = section_match.group(1).strip() if section_match else ""
        chunk = chunk_match.group(1).strip() if chunk_match else ""
        
        # Build crystallized text
        if "Key fact:" in prompt:
            # Summary prompt
            return f"This section covers {section.split('>')[-1].strip() if section else 'the topic'}."
        
        # Crystallization prompt
        context_prefix = f"Dans le document «{doc_title}»"
        if section:
            context_prefix += f", section «{section}»"
        
        return f"{context_prefix}: {chunk}"
    
    def crystallize(
        self,
        chunk_text: str,
        doc_title: str,
        header_path: str,
        generate_summary: bool = True
    ) -> CrystallizedChunk:
        """
        Crystallize a single chunk with global context.
        
        Args:
            chunk_text: Original chunk text
            doc_title: Document title
            header_path: Header path (e.g., "Section 1 > Article 5")
            generate_summary: Whether to generate key-fact summary
            
        Returns:
            CrystallizedChunk with enriched text
        """
        # Skip very short chunks
        if len(chunk_text.strip()) < 20:
            return CrystallizedChunk(
                original_text=chunk_text,
                crystallized_text=chunk_text,
                doc_title=doc_title,
                header_path=header_path,
                summary=""
            )
        
        # Truncate very long chunks
        truncated = chunk_text[:self.max_chunk_length]
        
        # Generate crystallized text
        prompt = self.CRYSTALLIZATION_PROMPT.format(
            doc_title=doc_title,
            header_path=header_path,
            chunk_text=truncated
        )
        
        crystallized = self._call_llm(prompt, max_tokens=len(truncated) + 200)
        
        # Fallback to simple prefix if LLM fails
        if not crystallized:
            prefix = f"[{doc_title}] [{header_path}] "
            crystallized = prefix + chunk_text
        
        # Generate summary if requested
        summary = ""
        if generate_summary:
            summary_prompt = self.SUMMARY_PROMPT.format(text=truncated[:500])
            summary = self._call_llm(summary_prompt, max_tokens=100)
        
        return CrystallizedChunk(
            original_text=chunk_text,
            crystallized_text=crystallized,
            doc_title=doc_title,
            header_path=header_path,
            summary=summary
        )
    
    def crystallize_batch(
        self,
        chunks: List[Dict[str, Any]],
        doc_title: str,
        generate_summaries: bool = True,
        show_progress: bool = True
    ) -> List[CrystallizedChunk]:
        """
        Crystallize a batch of chunks.
        
        Args:
            chunks: List of chunks with 'text' and 'header_path' keys
            doc_title: Document title for all chunks
            generate_summaries: Whether to generate summaries
            show_progress: Show progress logging
            
        Returns:
            List of CrystallizedChunk objects
        """
        results = []
        total = len(chunks)
        
        for i, chunk in enumerate(chunks):
            if show_progress and (i + 1) % 10 == 0:
                logger.info(f"Crystallizing chunk {i + 1}/{total}")
            
            text = chunk.get("text", "")
            header_path = chunk.get("header_path", "")
            
            crystallized = self.crystallize(
                chunk_text=text,
                doc_title=doc_title,
                header_path=header_path,
                generate_summary=generate_summaries
            )
            results.append(crystallized)
        
        if show_progress:
            logger.info(f"Crystallized {len(results)} chunks")
        
        return results


class SimpleContextualizer:
    """
    Simple rule-based contextualizer that doesn't require LLM.
    Uses template-based enrichment.
    """
    
    def __init__(self):
        logger.info("SimpleContextualizer initialized")
    
    def crystallize(
        self,
        chunk_text: str,
        doc_title: str,
        header_path: str,
        generate_summary: bool = True
    ) -> CrystallizedChunk:
        """
        Simple crystallization without LLM.
        
        Prepends document and section context.
        """
        # Build context prefix
        parts = []
        
        if doc_title:
            parts.append(f"Document: {doc_title}")
        
        if header_path:
            parts.append(f"Section: {header_path}")
        
        # Create crystallized text
        if parts:
            prefix = " | ".join(parts) + "\n\n"
            crystallized = prefix + chunk_text
        else:
            crystallized = chunk_text
        
        # Simple summary: first sentence or first 100 chars
        summary = ""
        if generate_summary:
            first_sentence = chunk_text.split('.')[0]
            summary = (first_sentence[:100] + "...") if len(first_sentence) > 100 else first_sentence + "."
        
        return CrystallizedChunk(
            original_text=chunk_text,
            crystallized_text=crystallized,
            doc_title=doc_title,
            header_path=header_path,
            summary=summary
        )
    
    def crystallize_batch(
        self,
        chunks: List[Dict[str, Any]],
        doc_title: str,
        generate_summaries: bool = True,
        show_progress: bool = True
    ) -> List[CrystallizedChunk]:
        """Crystallize batch of chunks."""
        results = []
        
        for chunk in chunks:
            text = chunk.get("text", "")
            header_path = chunk.get("header_path", "")
            
            crystallized = self.crystallize(
                chunk_text=text,
                doc_title=doc_title,
                header_path=header_path,
                generate_summary=generate_summaries
            )
            results.append(crystallized)
        
        if show_progress:
            logger.info(f"Crystallized {len(results)} chunks (simple mode)")
        
        return results


def get_contextualizer(
    llm_client=None,
    model: str = "qwen2.5:7b",
    use_mock: bool = False,
    use_simple: bool = False
):
    """
    Factory function to get appropriate contextualizer.
    
    Args:
        llm_client: LLM client for full contextualizer
        model: Model name
        use_mock: Use mock responses
        use_simple: Use simple rule-based contextualizer
        
    Returns:
        Contextualizer instance
    """
    if use_simple:
        return SimpleContextualizer()
    return ChunkContextualizer(
        llm_client=llm_client,
        model=model,
        use_mock=use_mock
    )


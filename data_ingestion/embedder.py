"""
Embedding Generator for Atlas-RAG
Uses BAAI/bge-m3 for multilingual embeddings.
"""
import logging
from typing import List, Optional, Union
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result of embedding operation."""
    text: str
    embedding: List[float]
    model: str
    dimension: int


class EmbeddingGenerator:
    """
    Generates embeddings using sentence-transformers.
    Supports BAAI/bge-m3 for multilingual documents.
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: Optional[str] = None,
        normalize: bool = True
    ):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use ('cpu', 'cuda', or None for auto)
            normalize: Whether to L2 normalize embeddings
        """
        self.model_name = model_name
        self.normalize = normalize
        self._model = None
        self._device = device
        
        logger.info(f"Initializing EmbeddingGenerator with model: {model_name}")
    
    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading model: {self.model_name}")
            self._model = SentenceTransformer(
                self.model_name,
                device=self._device
            )
            logger.info(f"Model loaded. Dimension: {self.dimension}")
        return self._model
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self.model.get_sentence_embedding_dimension()
    
    def embed_text(self, text: str) -> EmbeddingResult:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            EmbeddingResult with embedding vector
        """
        embedding = self.model.encode(
            text,
            normalize_embeddings=self.normalize,
            show_progress_bar=False
        )
        
        return EmbeddingResult(
            text=text,
            embedding=embedding.tolist(),
            model=self.model_name,
            dimension=len(embedding)
        )
    
    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[EmbeddingResult]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            List of EmbeddingResult objects
        """
        if not texts:
            return []
        
        logger.info(f"Embedding {len(texts)} texts with batch_size={batch_size}")
        
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=self.normalize,
            batch_size=batch_size,
            show_progress_bar=show_progress
        )
        
        results = []
        for text, embedding in zip(texts, embeddings):
            results.append(EmbeddingResult(
                text=text,
                embedding=embedding.tolist(),
                model=self.model_name,
                dimension=len(embedding)
            ))
        
        return results
    
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a query for retrieval.
        For BGE models, prepends instruction for better retrieval.
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector as list of floats
        """
        # BGE models work better with instruction prefix for queries
        if "bge" in self.model_name.lower():
            query = f"Represent this sentence for searching relevant passages: {query}"
        
        return self.embed_text(query).embedding


class MockEmbeddingGenerator:
    """
    Mock embedding generator for testing.
    Generates deterministic pseudo-random embeddings.
    """
    
    def __init__(self, dimension: int = 1024, seed: int = 42):
        self.model_name = "mock-embedding-model"
        self._dimension = dimension
        self._seed = seed
        logger.info(f"MockEmbeddingGenerator initialized (dim={dimension})")
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    def embed_text(self, text: str) -> EmbeddingResult:
        """Generate deterministic embedding based on text hash."""
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.randn(self._dimension)
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        
        return EmbeddingResult(
            text=text,
            embedding=embedding.tolist(),
            model=self.model_name,
            dimension=self._dimension
        )
    
    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[EmbeddingResult]:
        return [self.embed_text(text) for text in texts]
    
    def embed_query(self, query: str) -> List[float]:
        return self.embed_text(query).embedding


def get_embedding_generator(
    model_name: str = "BAAI/bge-m3",
    use_mock: bool = False,
    dimension: int = 1024
) -> Union[EmbeddingGenerator, MockEmbeddingGenerator]:
    """
    Factory function to get embedding generator.
    
    Args:
        model_name: Model name for real generator
        use_mock: Whether to use mock generator (for testing)
        dimension: Dimension for mock generator
        
    Returns:
        Embedding generator instance
    """
    if use_mock:
        return MockEmbeddingGenerator(dimension=dimension)
    return EmbeddingGenerator(model_name=model_name)


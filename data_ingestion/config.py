"""
Configuration for Data Ingestion Pipeline
"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class IngestionConfig:
    """Configuration for the data ingestion pipeline."""
    
    # Qdrant Settings
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection_name: str = os.getenv("QDRANT_COLLECTION", "atlas_knowledge")
    
    # Embedding Settings
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
    
    # Processing Settings
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    # Paths
    documents_dir: str = os.path.join(os.path.dirname(__file__), "documents")
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        if self.embedding_dimension <= 0:
            raise ValueError("Embedding dimension must be positive")
        if self.chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("Chunk overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        return True


def get_config() -> IngestionConfig:
    """Get the ingestion configuration."""
    config = IngestionConfig()
    config.validate()
    return config


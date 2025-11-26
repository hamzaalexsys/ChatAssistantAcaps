"""
Configuration for Backend API
Uses pydantic-settings for environment variable management.
"""
import os
from typing import Optional
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    app_name: str = "Atlas-RAG API"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    
    # LLM Inference Server (vLLM, Ollama, or OpenAI-compatible)
    vllm_url: str = "http://localhost:11434/v1"  # Default to Ollama
    vllm_api_key: str = "ollama"  # Ollama doesn't need a real key
    vllm_model: str = "qwen2.5:7b"  # Ollama model name
    
    # LLM Provider: "ollama", "vllm", "openai"
    llm_provider: str = "ollama"
    
    # Fallback: Return context-only response if LLM unavailable
    llm_fallback_enabled: bool = True
    
    # Qdrant Vector Database
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "atlas_knowledge"
    
    # Embedding Model
    embedding_model: str = "BAAI/bge-m3"
    embedding_dimension: int = 1024
    
    # Reranker
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    use_reranker: bool = True
    
    # RAG Configuration
    # Note: BGE-M3 similarity scores typically range 0.4-0.6, so threshold must be lower
    similarity_threshold: float = 0.40
    top_k_results: int = 5
    max_context_length: int = 8192
    
    # Generation Settings
    temperature: float = 0.0
    max_tokens: int = 1024
    
    # Guardrails
    enable_guardrails: bool = True
    
    # CORS
    cors_origins: str = "*"
    
    # Redis Cache Configuration (Atlas-Hyperion v3.0)
    redis_url: str = "redis://localhost:6379"
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 1 hour default TTL
    cache_similarity_threshold: float = 0.95  # Semantic cache hit threshold
    
    # NLI Verifier Configuration
    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    nli_enabled: bool = True
    nli_entailment_threshold: float = 0.7  # Minimum entailment score
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


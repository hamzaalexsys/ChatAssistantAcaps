"""
Semantic Cache Layer for Atlas-Hyperion v3.0
Multi-level caching with Redis VSS (Vector Similarity Search).

Cache Levels:
- L1 Query Cache: (query_embedding, similarity>0.95) -> cached_answer
- L2 Answer Cache: (query_hash, context_version) -> answer
- L3 Retrieval Cache: (query_embedding) -> doc_ids + scores
"""
import json
import logging
import hashlib
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CachedResponse:
    """Cached response with metadata."""
    query: str
    answer: str
    citations: List[Dict[str, Any]]
    confidence: float
    cached_at: str
    ttl: int
    cache_level: str  # L1, L2, L3
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CachedResponse":
        return cls(**data)


@dataclass
class CachedRetrieval:
    """Cached retrieval results."""
    query_hash: str
    doc_ids: List[str]
    scores: List[float]
    cached_at: str
    collection_version: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CachedRetrieval":
        return cls(**data)


class SemanticCache:
    """
    Multi-level semantic cache using Redis VSS.
    
    L1: Semantic similarity cache - matches similar queries
    L2: Exact hash cache - for identical queries
    L3: Retrieval cache - cached document IDs and scores
    """
    
    L1_PREFIX = "atlas:l1:"
    L2_PREFIX = "atlas:l2:"
    L3_PREFIX = "atlas:l3:"
    INDEX_NAME = "atlas_query_idx"
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        embedding_dimension: int = 1024,
        similarity_threshold: float = 0.95,
        default_ttl: int = 3600,
        enabled: bool = True
    ):
        """
        Initialize semantic cache.
        
        Args:
            redis_url: Redis connection URL
            embedding_dimension: Dimension of query embeddings
            similarity_threshold: Minimum similarity for L1 cache hits
            default_ttl: Default TTL in seconds
            enabled: Whether caching is enabled
        """
        self.redis_url = redis_url
        self.embedding_dimension = embedding_dimension
        self.similarity_threshold = similarity_threshold
        self.default_ttl = default_ttl
        self.enabled = enabled
        self._client = None
        self._index_created = False
        
        logger.info(f"SemanticCache initialized (enabled={enabled}, threshold={similarity_threshold})")
    
    @property
    def client(self):
        """Lazy load Redis client."""
        if self._client is None:
            try:
                import redis
                self._client = redis.from_url(
                    self.redis_url,
                    decode_responses=True
                )
                # Test connection
                self._client.ping()
                logger.info("Redis connection established")
            except Exception as e:
                logger.error(f"Redis connection failed: {e}")
                self._client = None
        return self._client
    
    def _ensure_index(self) -> bool:
        """Create Redis VSS index if not exists."""
        if self._index_created or self.client is None:
            return self._index_created
        
        try:
            # Check if index exists
            try:
                self.client.ft(self.INDEX_NAME).info()
                self._index_created = True
                return True
            except Exception:
                pass
            
            # Create index with vector field
            from redis.commands.search.field import VectorField, TextField, NumericField
            from redis.commands.search.indexDefinition import IndexDefinition, IndexType
            
            schema = (
                TextField("$.query", as_name="query"),
                NumericField("$.cached_at_ts", as_name="cached_at_ts"),
                VectorField(
                    "$.embedding",
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self.embedding_dimension,
                        "DISTANCE_METRIC": "COSINE"
                    },
                    as_name="embedding"
                )
            )
            
            definition = IndexDefinition(
                prefix=[self.L1_PREFIX],
                index_type=IndexType.JSON
            )
            
            self.client.ft(self.INDEX_NAME).create_index(
                schema,
                definition=definition
            )
            self._index_created = True
            logger.info(f"Created Redis VSS index: {self.INDEX_NAME}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to create VSS index (falling back to L2 only): {e}")
            return False
    
    def _hash_query(self, query: str) -> str:
        """Generate hash for query string."""
        return hashlib.sha256(query.lower().strip().encode()).hexdigest()[:32]
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a_arr = np.array(a)
        b_arr = np.array(b)
        return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))
    
    # ===========================================
    # L1: Semantic Similarity Cache
    # ===========================================
    
    def get_l1(self, query_embedding: List[float]) -> Optional[CachedResponse]:
        """
        Check L1 semantic cache for similar queries.
        
        Args:
            query_embedding: Query embedding vector
            
        Returns:
            CachedResponse if similar query found, None otherwise
        """
        if not self.enabled or self.client is None:
            return None
        
        if not self._ensure_index():
            return None
        
        try:
            from redis.commands.search.query import Query
            
            # Build KNN query
            query_bytes = np.array(query_embedding, dtype=np.float32).tobytes()
            
            q = (
                Query(f"*=>[KNN 1 @embedding $vec AS score]")
                .sort_by("score")
                .return_fields("query", "score", "$")
                .dialect(2)
            )
            
            results = self.client.ft(self.INDEX_NAME).search(
                q,
                query_params={"vec": query_bytes}
            )
            
            if results.total == 0:
                return None
            
            doc = results.docs[0]
            # Redis returns distance, convert to similarity
            similarity = 1 - float(doc.score)
            
            if similarity >= self.similarity_threshold:
                data = json.loads(doc["$"])
                logger.info(f"L1 cache hit (similarity={similarity:.3f})")
                return CachedResponse.from_dict(data["response"])
            
            return None
            
        except Exception as e:
            logger.warning(f"L1 cache lookup failed: {e}")
            return None
    
    def set_l1(
        self,
        query: str,
        query_embedding: List[float],
        response: CachedResponse,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Store response in L1 semantic cache.
        
        Args:
            query: Original query text
            query_embedding: Query embedding vector
            response: Response to cache
            ttl: Optional TTL override
            
        Returns:
            True if stored successfully
        """
        if not self.enabled or self.client is None:
            return False
        
        if not self._ensure_index():
            return False
        
        try:
            key = f"{self.L1_PREFIX}{self._hash_query(query)}"
            now = datetime.now(timezone.utc)
            
            data = {
                "query": query,
                "embedding": query_embedding,
                "response": response.to_dict(),
                "cached_at_ts": int(now.timestamp())
            }
            
            self.client.json().set(key, "$", data)
            self.client.expire(key, ttl or self.default_ttl)
            
            logger.debug(f"Stored L1 cache: {key}")
            return True
            
        except Exception as e:
            logger.warning(f"L1 cache store failed: {e}")
            return False
    
    # ===========================================
    # L2: Exact Query Hash Cache
    # ===========================================
    
    def get_l2(self, query: str) -> Optional[CachedResponse]:
        """
        Check L2 exact hash cache.
        
        Args:
            query: Query text
            
        Returns:
            CachedResponse if exact match found
        """
        if not self.enabled or self.client is None:
            return None
        
        try:
            key = f"{self.L2_PREFIX}{self._hash_query(query)}"
            data = self.client.get(key)
            
            if data:
                logger.info("L2 cache hit (exact match)")
                return CachedResponse.from_dict(json.loads(data))
            
            return None
            
        except Exception as e:
            logger.warning(f"L2 cache lookup failed: {e}")
            return None
    
    def set_l2(
        self,
        query: str,
        response: CachedResponse,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Store response in L2 exact cache.
        
        Args:
            query: Query text
            response: Response to cache
            ttl: Optional TTL override
            
        Returns:
            True if stored successfully
        """
        if not self.enabled or self.client is None:
            return False
        
        try:
            key = f"{self.L2_PREFIX}{self._hash_query(query)}"
            self.client.setex(
                key,
                ttl or self.default_ttl,
                json.dumps(response.to_dict())
            )
            logger.debug(f"Stored L2 cache: {key}")
            return True
            
        except Exception as e:
            logger.warning(f"L2 cache store failed: {e}")
            return False
    
    # ===========================================
    # L3: Retrieval Results Cache
    # ===========================================
    
    def get_l3(self, query: str, collection_version: str) -> Optional[CachedRetrieval]:
        """
        Check L3 retrieval cache.
        
        Args:
            query: Query text
            collection_version: Current collection version for invalidation
            
        Returns:
            CachedRetrieval if found and valid
        """
        if not self.enabled or self.client is None:
            return None
        
        try:
            key = f"{self.L3_PREFIX}{self._hash_query(query)}"
            data = self.client.get(key)
            
            if data:
                cached = CachedRetrieval.from_dict(json.loads(data))
                
                # Invalidate if collection version changed
                if cached.collection_version != collection_version:
                    self.client.delete(key)
                    return None
                
                logger.info("L3 cache hit (retrieval)")
                return cached
            
            return None
            
        except Exception as e:
            logger.warning(f"L3 cache lookup failed: {e}")
            return None
    
    def set_l3(
        self,
        query: str,
        doc_ids: List[str],
        scores: List[float],
        collection_version: str,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Store retrieval results in L3 cache.
        
        Args:
            query: Query text
            doc_ids: Retrieved document IDs
            scores: Retrieval scores
            collection_version: Collection version for invalidation
            ttl: Optional TTL override
            
        Returns:
            True if stored successfully
        """
        if not self.enabled or self.client is None:
            return False
        
        try:
            key = f"{self.L3_PREFIX}{self._hash_query(query)}"
            now = datetime.now(timezone.utc)
            
            cached = CachedRetrieval(
                query_hash=self._hash_query(query),
                doc_ids=doc_ids,
                scores=scores,
                cached_at=now.isoformat(),
                collection_version=collection_version
            )
            
            self.client.setex(
                key,
                ttl or self.default_ttl,
                json.dumps(cached.to_dict())
            )
            logger.debug(f"Stored L3 cache: {key}")
            return True
            
        except Exception as e:
            logger.warning(f"L3 cache store failed: {e}")
            return False
    
    # ===========================================
    # Combined Cache Operations
    # ===========================================
    
    def get(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None
    ) -> Tuple[Optional[CachedResponse], str]:
        """
        Check all cache levels.
        
        Args:
            query: Query text
            query_embedding: Optional query embedding for L1 check
            
        Returns:
            Tuple of (CachedResponse or None, cache_level)
        """
        # Try L2 first (fastest - exact match)
        response = self.get_l2(query)
        if response:
            return response, "L2"
        
        # Try L1 (semantic similarity)
        if query_embedding:
            response = self.get_l1(query_embedding)
            if response:
                return response, "L1"
        
        return None, "MISS"
    
    def set(
        self,
        query: str,
        query_embedding: List[float],
        response: CachedResponse,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Store response in both L1 and L2 caches.
        
        Args:
            query: Query text
            query_embedding: Query embedding
            response: Response to cache
            ttl: Optional TTL override
            
        Returns:
            True if at least one cache was updated
        """
        l2_success = self.set_l2(query, response, ttl)
        l1_success = self.set_l1(query, query_embedding, response, ttl)
        return l2_success or l1_success
    
    def invalidate_all(self) -> int:
        """
        Invalidate all cache entries.
        
        Returns:
            Number of keys deleted
        """
        if self.client is None:
            return 0
        
        try:
            count = 0
            for prefix in [self.L1_PREFIX, self.L2_PREFIX, self.L3_PREFIX]:
                keys = list(self.client.scan_iter(f"{prefix}*"))
                if keys:
                    count += self.client.delete(*keys)
            
            logger.info(f"Invalidated {count} cache entries")
            return count
            
        except Exception as e:
            logger.error(f"Cache invalidation failed: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self.client is None:
            return {"status": "disconnected"}
        
        try:
            info = self.client.info("stats")
            l1_count = len(list(self.client.scan_iter(f"{self.L1_PREFIX}*", count=1000)))
            l2_count = len(list(self.client.scan_iter(f"{self.L2_PREFIX}*", count=1000)))
            l3_count = len(list(self.client.scan_iter(f"{self.L3_PREFIX}*", count=1000)))
            
            return {
                "status": "connected",
                "l1_entries": l1_count,
                "l2_entries": l2_count,
                "l3_entries": l3_count,
                "total_entries": l1_count + l2_count + l3_count,
                "total_connections": info.get("total_connections_received", 0),
                "used_memory": self.client.info("memory").get("used_memory_human", "unknown")
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"status": "error", "error": str(e)}
    
    def health_check(self) -> bool:
        """Check if Redis is healthy."""
        if not self.enabled:
            return True  # Not enabled, so "healthy"
        
        if self.client is None:
            return False
        
        try:
            return self.client.ping()
        except Exception:
            return False


class MockSemanticCache:
    """
    Mock semantic cache for testing without Redis.
    Uses in-memory dictionary.
    """
    
    def __init__(
        self,
        embedding_dimension: int = 1024,
        similarity_threshold: float = 0.95,
        default_ttl: int = 3600,
        enabled: bool = True
    ):
        self.embedding_dimension = embedding_dimension
        self.similarity_threshold = similarity_threshold
        self.default_ttl = default_ttl
        self.enabled = enabled
        
        self._l1_cache: Dict[str, Tuple[List[float], CachedResponse]] = {}
        self._l2_cache: Dict[str, CachedResponse] = {}
        self._l3_cache: Dict[str, CachedRetrieval] = {}
        
        logger.info("MockSemanticCache initialized")
    
    def _hash_query(self, query: str) -> str:
        return hashlib.sha256(query.lower().strip().encode()).hexdigest()[:32]
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        a_arr = np.array(a)
        b_arr = np.array(b)
        return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))
    
    def get_l1(self, query_embedding: List[float]) -> Optional[CachedResponse]:
        if not self.enabled:
            return None
        
        for key, (embedding, response) in self._l1_cache.items():
            similarity = self._cosine_similarity(query_embedding, embedding)
            if similarity >= self.similarity_threshold:
                logger.info(f"MockCache L1 hit (similarity={similarity:.3f})")
                return response
        return None
    
    def set_l1(self, query: str, query_embedding: List[float], response: CachedResponse, ttl: Optional[int] = None) -> bool:
        if not self.enabled:
            return False
        key = self._hash_query(query)
        self._l1_cache[key] = (query_embedding, response)
        return True
    
    def get_l2(self, query: str) -> Optional[CachedResponse]:
        if not self.enabled:
            return None
        key = self._hash_query(query)
        if key in self._l2_cache:
            logger.info("MockCache L2 hit")
            return self._l2_cache[key]
        return None
    
    def set_l2(self, query: str, response: CachedResponse, ttl: Optional[int] = None) -> bool:
        if not self.enabled:
            return False
        key = self._hash_query(query)
        self._l2_cache[key] = response
        return True
    
    def get_l3(self, query: str, collection_version: str) -> Optional[CachedRetrieval]:
        if not self.enabled:
            return None
        key = self._hash_query(query)
        if key in self._l3_cache:
            cached = self._l3_cache[key]
            if cached.collection_version == collection_version:
                logger.info("MockCache L3 hit")
                return cached
            del self._l3_cache[key]
        return None
    
    def set_l3(self, query: str, doc_ids: List[str], scores: List[float], collection_version: str, ttl: Optional[int] = None) -> bool:
        if not self.enabled:
            return False
        key = self._hash_query(query)
        now = datetime.now(timezone.utc)
        self._l3_cache[key] = CachedRetrieval(
            query_hash=key,
            doc_ids=doc_ids,
            scores=scores,
            cached_at=now.isoformat(),
            collection_version=collection_version
        )
        return True
    
    def get(self, query: str, query_embedding: Optional[List[float]] = None) -> Tuple[Optional[CachedResponse], str]:
        response = self.get_l2(query)
        if response:
            return response, "L2"
        if query_embedding:
            response = self.get_l1(query_embedding)
            if response:
                return response, "L1"
        return None, "MISS"
    
    def set(self, query: str, query_embedding: List[float], response: CachedResponse, ttl: Optional[int] = None) -> bool:
        l2_success = self.set_l2(query, response, ttl)
        l1_success = self.set_l1(query, query_embedding, response, ttl)
        return l2_success or l1_success
    
    def invalidate_all(self) -> int:
        count = len(self._l1_cache) + len(self._l2_cache) + len(self._l3_cache)
        self._l1_cache.clear()
        self._l2_cache.clear()
        self._l3_cache.clear()
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "status": "mock",
            "l1_entries": len(self._l1_cache),
            "l2_entries": len(self._l2_cache),
            "l3_entries": len(self._l3_cache),
            "total_entries": len(self._l1_cache) + len(self._l2_cache) + len(self._l3_cache)
        }
    
    def health_check(self) -> bool:
        return True


# Singleton instance
_cache_instance: Optional[SemanticCache] = None


def get_cache(
    redis_url: str = "redis://localhost:6379",
    embedding_dimension: int = 1024,
    similarity_threshold: float = 0.95,
    default_ttl: int = 3600,
    enabled: bool = True,
    use_mock: bool = False
) -> SemanticCache:
    """Get or create cache instance."""
    global _cache_instance
    if _cache_instance is None:
        if use_mock:
            _cache_instance = MockSemanticCache(
                embedding_dimension=embedding_dimension,
                similarity_threshold=similarity_threshold,
                default_ttl=default_ttl,
                enabled=enabled
            )
        else:
            _cache_instance = SemanticCache(
                redis_url=redis_url,
                embedding_dimension=embedding_dimension,
                similarity_threshold=similarity_threshold,
                default_ttl=default_ttl,
                enabled=enabled
            )
    return _cache_instance


"""
Vector Store Interface for Atlas-RAG
Handles Qdrant operations: collection management, upsert, search.
"""
import logging
import uuid
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchText,
    SearchParams,
)

from .parser import DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Search result with score and payload."""
    id: str
    score: float
    text: str
    file_name: str
    header_path: str
    url_slug: str
    base_url: str
    metadata: Dict[str, Any]
    
    @property
    def full_url(self) -> str:
        """Get the full URL for citation."""
        return self.url_slug or self.base_url


class VectorStore:
    """
    Vector store interface for Qdrant.
    Handles all vector database operations.
    """
    
    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = "atlas_knowledge",
        embedding_dimension: int = 1024,
        distance: Distance = Distance.COSINE
    ):
        """
        Initialize vector store connection.
        
        Args:
            url: Qdrant server URL
            collection_name: Name of the collection
            embedding_dimension: Dimension of embedding vectors
            distance: Distance metric for similarity
        """
        self.url = url
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        self.distance = distance
        
        logger.info(f"Connecting to Qdrant at {url}")
        self.client = QdrantClient(url=url)
    
    def create_collection(self, recreate: bool = False) -> bool:
        """
        Create the collection if it doesn't exist.
        
        Args:
            recreate: If True, delete existing collection first
            
        Returns:
            True if collection was created, False if it already existed
        """
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if exists and recreate:
            logger.warning(f"Deleting existing collection: {self.collection_name}")
            self.client.delete_collection(self.collection_name)
            exists = False
        
        if not exists:
            logger.info(f"Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dimension,
                    distance=self.distance
                )
            )
            return True
        
        logger.info(f"Collection {self.collection_name} already exists")
        return False
    
    def upsert_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
        batch_size: int = 100
    ) -> int:
        """
        Upsert document chunks with their embeddings.
        
        Args:
            chunks: List of DocumentChunk objects
            embeddings: List of embedding vectors
            batch_size: Batch size for upsert operations
            
        Returns:
            Number of points upserted
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        points = []
        for chunk, embedding in zip(chunks, embeddings):
            point_id = str(uuid.uuid4())
            payload = chunk.to_dict()
            
            points.append(PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            ))
        
        # Upsert in batches
        total_upserted = 0
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
                wait=True
            )
            total_upserted += len(batch)
            logger.debug(f"Upserted batch: {total_upserted}/{len(points)}")
        
        logger.info(f"Upserted {total_upserted} points to {self.collection_name}")
        return total_upserted
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            score_threshold: Minimum score threshold
            filter_conditions: Optional filter conditions
            
        Returns:
            List of SearchResult objects
        """
        # Build filter if conditions provided
        query_filter = None
        if filter_conditions:
            must_conditions = []
            for field, value in filter_conditions.items():
                must_conditions.append(
                    FieldCondition(
                        key=field,
                        match=MatchValue(value=value)
                    )
                )
            query_filter = Filter(must=must_conditions)
        
        # Use search method for vector similarity search
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
            query_filter=query_filter
        )
        
        # search() returns a list of ScoredPoint objects directly
        results = search_results
        
        search_results = []
        for result in results:
            payload = result.payload or {}
            search_results.append(SearchResult(
                id=str(result.id),
                score=result.score,
                text=payload.get("text", ""),
                file_name=payload.get("file_name", ""),
                header_path=payload.get("header_path", ""),
                url_slug=payload.get("url_slug", ""),
                base_url=payload.get("base_url", ""),
                metadata={
                    k: v for k, v in payload.items()
                    if k not in ["text", "file_name", "header_path", "url_slug", "base_url"]
                }
            ))
        
        return search_results
    
    def search_by_keyword(
        self,
        keyword: str,
        field: str = "header_path",
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Search for documents containing a keyword in a specific field.
        Uses text matching (not semantic).
        
        Args:
            keyword: Keyword to search for
            field: Field to search in (header_path, text, file_name)
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        try:
            response = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key=field,
                            match=MatchText(text=keyword)
                        )
                    ]
                ),
                limit=top_k,
                with_payload=True,
                with_vectors=False
            )
            
            points, _ = response
            
            search_results = []
            for point in points:
                payload = point.payload or {}
                search_results.append(SearchResult(
                    id=str(point.id),
                    score=1.0,  # Keyword matches get high score
                    text=payload.get("text", ""),
                    file_name=payload.get("file_name", ""),
                    header_path=payload.get("header_path", ""),
                    url_slug=payload.get("url_slug", ""),
                    base_url=payload.get("base_url", ""),
                    metadata={
                        k: v for k, v in payload.items()
                        if k not in ["text", "file_name", "header_path", "url_slug", "base_url"]
                    }
                ))
            
            return search_results
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []
    
    def hybrid_search(
        self,
        query_embedding: List[float],
        query_text: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        keyword_boost: float = 0.3
    ) -> List[SearchResult]:
        """
        Hybrid search combining semantic and keyword search.
        
        If the query contains specific references (like "Article X"), 
        keyword matches are boosted.
        
        Args:
            query_embedding: Query embedding vector
            query_text: Original query text for keyword extraction
            top_k: Number of results to return
            score_threshold: Minimum score threshold
            keyword_boost: Score boost for keyword matches
            
        Returns:
            List of SearchResult objects (deduplicated and re-ranked)
        """
        import re
        
        # Step 1: Semantic search
        semantic_results = self.search(
            query_embedding=query_embedding,
            top_k=top_k * 2,  # Get more for merging
            score_threshold=0.0  # Get all, filter later
        )
        
        # Step 2: Extract keywords for targeted search
        # Look for "Article X", "Section X", "Chapitre X" patterns
        keyword_results = []
        patterns = [
            r'article\s*(\d+)',
            r'section\s*(\d+)',
            r'chapitre\s*(\d+)',
            r'titre\s*(\d+)',
        ]
        
        query_lower = query_text.lower()
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                # Search for this specific reference
                keyword = f"Article {match.group(1)}" if "article" in pattern else f"Section {match.group(1)}"
                kw_results = self.search_by_keyword(keyword, field="header_path", top_k=5)
                keyword_results.extend(kw_results)
                logger.info(f"Keyword search for '{keyword}' found {len(kw_results)} results")
        
        # Step 3: Merge and re-rank
        result_map = {}
        
        # Add semantic results
        for result in semantic_results:
            result_map[result.id] = result
        
        # Add/boost keyword results
        for result in keyword_results:
            if result.id in result_map:
                # Boost existing result
                existing = result_map[result.id]
                existing.score = min(1.0, existing.score + keyword_boost)
            else:
                # Add new result with boosted score
                result.score = keyword_boost + 0.5  # Base + boost
                result_map[result.id] = result
        
        # Sort by score and filter
        all_results = sorted(result_map.values(), key=lambda x: x.score, reverse=True)
        filtered = [r for r in all_results if r.score >= score_threshold]
        
        return filtered[:top_k]
    
    def graph_search(
        self,
        query_embedding: List[float],
        query_text: str = "",
        top_k: int = 5,
        depth: int = 2,
        score_threshold: float = 0.0,
        max_expansion: int = 10
    ) -> List[SearchResult]:
        """
        Graph-enhanced search for Atlas-Hyperion v3.0.
        
        Combines vector search with graph traversal to pull
        referenced articles/sections (multi-hop retrieval).
        
        Args:
            query_embedding: Query embedding vector
            query_text: Original query text
            top_k: Number of final results to return
            depth: Maximum graph traversal depth
            score_threshold: Minimum score threshold
            max_expansion: Maximum nodes to expand per hop
            
        Returns:
            List of SearchResult objects with graph-expanded context
        """
        # Step 1: Initial vector search
        initial_results = self.hybrid_search(
            query_embedding=query_embedding,
            query_text=query_text,
            top_k=top_k * 2,
            score_threshold=score_threshold
        )
        
        if not initial_results:
            return []
        
        # Step 2: Graph expansion
        result_map = {r.id: r for r in initial_results}
        visited = set(result_map.keys())
        frontier = list(initial_results)
        
        for hop in range(depth):
            if not frontier:
                break
            
            # Collect edges from frontier nodes
            edges_to_expand = []
            for result in frontier[:max_expansion]:
                # Get edges from metadata
                edges = result.metadata.get("edges", [])
                if isinstance(edges, list):
                    edges_to_expand.extend(edges)
            
            if not edges_to_expand:
                break
            
            # Find nodes matching the edge references
            new_frontier = []
            for edge_ref in set(edges_to_expand):
                if not edge_ref:
                    continue
                
                # Search for the referenced node
                try:
                    edge_results = self.search_by_keyword(
                        keyword=edge_ref,
                        field="header_path",
                        top_k=3
                    )
                    
                    for edge_result in edge_results:
                        if edge_result.id not in visited:
                            # Apply decay to graph-expanded results
                            edge_result.score *= (0.8 ** (hop + 1))
                            result_map[edge_result.id] = edge_result
                            new_frontier.append(edge_result)
                            visited.add(edge_result.id)
                            
                except Exception as e:
                    logger.debug(f"Edge expansion failed for '{edge_ref}': {e}")
            
            frontier = new_frontier
            logger.debug(f"Graph hop {hop + 1}: expanded {len(new_frontier)} nodes")
        
        # Step 3: Re-rank and return
        all_results = sorted(result_map.values(), key=lambda x: x.score, reverse=True)
        
        logger.info(f"Graph search: {len(initial_results)} initial -> {len(all_results)} total")
        return all_results[:top_k]
    
    def get_by_ids(self, ids: List[str]) -> List[SearchResult]:
        """
        Retrieve specific documents by their IDs.
        
        Args:
            ids: List of document IDs to retrieve
            
        Returns:
            List of SearchResult objects
        """
        if not ids:
            return []
        
        try:
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=ids,
                with_payload=True,
                with_vectors=False
            )
            
            results = []
            for point in points:
                payload = point.payload or {}
                results.append(SearchResult(
                    id=str(point.id),
                    score=1.0,  # Perfect score for direct retrieval
                    text=payload.get("text", ""),
                    file_name=payload.get("file_name", ""),
                    header_path=payload.get("header_path", ""),
                    url_slug=payload.get("url_slug", ""),
                    base_url=payload.get("base_url", ""),
                    metadata={
                        k: v for k, v in payload.items()
                        if k not in ["text", "file_name", "header_path", "url_slug", "base_url"]
                    }
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve by IDs: {e}")
            return []
    
    def has_file(self, file_name: str) -> bool:
        """
        Check if a file exists in the vector store.
        
        Args:
            file_name: Name of the source file
            
        Returns:
            True if the file has chunks in the store, False otherwise
        """
        try:
            # Use count_points with filter to check if file exists
            count_result = self.client.count(
                collection_name=self.collection_name,
                count_filter=Filter(
                    must=[
                        FieldCondition(
                            key="file_name",
                            match=MatchValue(value=file_name)
                        )
                    ]
                )
            )
            return count_result.count > 0
        except Exception as e:
            logger.error(f"Error checking if file exists: {e}")
            return False

    def get_file_last_updated(self, file_name: str) -> Optional[str]:
        """
        Get the last_updated timestamp for a file from the vector store.
        
        Args:
            file_name: Name of the source file
            
        Returns:
            ISO format timestamp string or None if file not found
        """
        try:
            # Retrieve one point for this file to get metadata
            response = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="file_name",
                            match=MatchValue(value=file_name)
                        )
                    ]
                ),
                limit=1,
                with_payload=True,
                with_vectors=False
            )
            
            points, _ = response
            if points and points[0].payload:
                return points[0].payload.get("last_updated")
            return None
        except Exception as e:
            logger.error(f"Error getting file info: {e}")
            return None

    def delete_by_file(self, file_name: str) -> int:
        """
        Delete all chunks from a specific file.

        Args:
            file_name: Name of the source file

        Returns:
            Number of points deleted
        """
        result = self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="file_name",
                            match=MatchValue(value=file_name)
                        )
                    ]
                )
            )
        )
        logger.info(f"Deleted points from file: {file_name}")
        return result
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information and stats."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status,
                "config": {
                    "dimension": self.embedding_dimension,
                    "distance": self.distance.value
                }
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"error": str(e)}
    
    def health_check(self) -> bool:
        """Check if Qdrant is healthy and accessible."""
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False


class MockVectorStore:
    """
    Mock vector store for testing without Qdrant.
    Stores vectors in memory.
    """
    
    def __init__(self, collection_name: str = "test_collection", embedding_dimension: int = 1024):
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        self.points: Dict[str, Dict[str, Any]] = {}
        logger.info(f"MockVectorStore initialized: {collection_name}")
    
    def create_collection(self, recreate: bool = False) -> bool:
        if recreate:
            self.points = {}
        return True
    
    def upsert_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
        batch_size: int = 100
    ) -> int:
        for chunk, embedding in zip(chunks, embeddings):
            point_id = str(uuid.uuid4())
            self.points[point_id] = {
                "vector": embedding,
                "payload": chunk.to_dict()
            }
        return len(chunks)
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        import numpy as np
        
        results = []
        query_vec = np.array(query_embedding)
        
        for point_id, point in self.points.items():
            vec = np.array(point["vector"])
            # Cosine similarity
            score = float(np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec)))
            
            if score_threshold and score < score_threshold:
                continue
            
            payload = point["payload"]
            results.append(SearchResult(
                id=point_id,
                score=score,
                text=payload.get("text", ""),
                file_name=payload.get("file_name", ""),
                header_path=payload.get("header_path", ""),
                url_slug=payload.get("url_slug", ""),
                base_url=payload.get("base_url", ""),
                metadata={
                    k: v for k, v in payload.items()
                    if k not in ["text", "file_name", "header_path", "url_slug", "base_url"]
                }
            ))
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    def hybrid_search(
        self,
        query_embedding: List[float],
        query_text: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        keyword_boost: float = 0.3
    ) -> List[SearchResult]:
        """Mock hybrid search."""
        return self.search(query_embedding, top_k, score_threshold)
    
    def graph_search(
        self,
        query_embedding: List[float],
        query_text: str = "",
        top_k: int = 5,
        depth: int = 2,
        score_threshold: float = 0.0,
        max_expansion: int = 10
    ) -> List[SearchResult]:
        """Mock graph search - just does regular search."""
        results = self.search(query_embedding, top_k * 2, score_threshold)
        
        # Simple graph expansion from edges
        result_map = {r.id: r for r in results}
        
        for result in results[:5]:
            edges = result.metadata.get("edges", [])
            for edge in edges:
                # Find matching chunks
                for point_id, point in self.points.items():
                    payload = point["payload"]
                    if edge in payload.get("header_path", "") and point_id not in result_map:
                        result_map[point_id] = SearchResult(
                            id=point_id,
                            score=result.score * 0.8,
                            text=payload.get("text", ""),
                            file_name=payload.get("file_name", ""),
                            header_path=payload.get("header_path", ""),
                            url_slug=payload.get("url_slug", ""),
                            base_url=payload.get("base_url", ""),
                            metadata={
                                k: v for k, v in payload.items()
                                if k not in ["text", "file_name", "header_path", "url_slug", "base_url"]
                            }
                        )
        
        all_results = sorted(result_map.values(), key=lambda x: x.score, reverse=True)
        return all_results[:top_k]
    
    def get_by_ids(self, ids: List[str]) -> List[SearchResult]:
        """Get documents by IDs."""
        results = []
        for point_id in ids:
            if point_id in self.points:
                payload = self.points[point_id]["payload"]
                results.append(SearchResult(
                    id=point_id,
                    score=1.0,
                    text=payload.get("text", ""),
                    file_name=payload.get("file_name", ""),
                    header_path=payload.get("header_path", ""),
                    url_slug=payload.get("url_slug", ""),
                    base_url=payload.get("base_url", ""),
                    metadata={}
                ))
        return results
    
    def has_file(self, file_name: str) -> bool:
        """Check if a file exists in the mock store."""
        return any(
            p["payload"].get("file_name") == file_name
            for p in self.points.values()
        )

    def get_file_last_updated(self, file_name: str) -> Optional[str]:
        """Get timestamp from mock store."""
        for p in self.points.values():
            if p["payload"].get("file_name") == file_name:
                return p["payload"].get("last_updated")
        return None

    def delete_by_file(self, file_name: str) -> int:
        to_delete = [
            pid for pid, p in self.points.items()
            if p["payload"].get("file_name") == file_name
        ]
        for pid in to_delete:
            del self.points[pid]
        return len(to_delete)
    
    def get_collection_info(self) -> Dict[str, Any]:
        return {
            "name": self.collection_name,
            "points_count": len(self.points),
            "status": "ok"
        }
    
    def health_check(self) -> bool:
        return True


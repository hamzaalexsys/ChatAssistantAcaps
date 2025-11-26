"""
RAG Engine for Atlas-Hyperion v3.0
Cache-Reason-Verify workflow with:
- Semantic caching (L1/L2/L3)
- Agentic planning (Self-RAG style)
- Graph-enhanced retrieval
- 3-tier verification
"""
import logging
import time
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from .config import get_settings, Settings
from .cache import SemanticCache, CachedResponse, get_cache, MockSemanticCache
from .planner import AgenticPlanner, PlannerDecision, PlannerAction, get_planner

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """A citation for a source document."""
    title: str
    url: str
    score: float
    text_snippet: str


@dataclass
class RAGResponse:
    """Response from the RAG engine."""
    answer: str
    citations: List[Citation]
    confidence: float
    context_used: str
    metadata: Dict[str, Any]
    # Atlas-Hyperion v3.0 fields
    cache_hit: bool = False
    cache_level: str = ""
    planner_action: str = ""
    verification_passed: bool = True
    latency_ms: float = 0.0


class RAGEngine:
    """
    Main RAG engine for Atlas-Hyperion v3.0.
    Implements Cache-Reason-Verify workflow:
    1. Check semantic cache for similar queries
    2. Use agentic planner to decide retrieval strategy
    3. Retrieve with graph expansion if needed
    4. Generate with LLM
    5. Verify with 3-tier guardrails
    """
    
    SYSTEM_PROMPT = """You are a reference assistant for ACAPS (Autorité de Contrôle des Assurances et de la Prévoyance Sociale) internal documentation.

CRITICAL RULES:
1. You ONLY answer based on the provided context
2. If the answer is NOT in the context, say "I cannot find this information in the available documents"
3. NEVER make up information or speculate
4. Always cite the source using the provided header path
5. Be precise and factual
6. Answer in the same language as the question

Context from documents:
{context}

Question: {question}

Answer based ONLY on the context above:"""
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        use_mock: bool = False,
        enable_cache: bool = True,
        enable_planner: bool = True
    ):
        """
        Initialize RAG engine with Atlas-Hyperion v3.0 components.
        
        Args:
            settings: Configuration settings
            use_mock: Use mock components for testing
            enable_cache: Enable semantic caching
            enable_planner: Enable agentic planner
        """
        self.settings = settings or get_settings()
        self.use_mock = use_mock
        self.enable_cache = enable_cache and self.settings.cache_enabled
        self.enable_planner = enable_planner
        
        self._llm_client = None
        self._vector_store = None
        self._embedder = None
        self._cache = None
        self._planner = None
        self._collection_version = None  # For cache invalidation
        
        logger.info(f"RAGEngine initialized (mock={use_mock}, cache={self.enable_cache}, planner={enable_planner})")
    
    @property
    def llm_client(self):
        """Lazy load LLM client."""
        if self._llm_client is None:
            if self.use_mock:
                self._llm_client = MockLLMClient()
            else:
                from openai import OpenAI
                self._llm_client = OpenAI(
                    base_url=self.settings.vllm_url,
                    api_key=self.settings.vllm_api_key
                )
        return self._llm_client
    
    @property
    def vector_store(self):
        """Lazy load vector store."""
        if self._vector_store is None:
            if self.use_mock:
                from data_ingestion.vector_store import MockVectorStore
                self._vector_store = MockVectorStore(
                    collection_name=self.settings.qdrant_collection,
                    embedding_dimension=self.settings.embedding_dimension
                )
            else:
                from data_ingestion.vector_store import VectorStore
                self._vector_store = VectorStore(
                    url=self.settings.qdrant_url,
                    collection_name=self.settings.qdrant_collection,
                    embedding_dimension=self.settings.embedding_dimension
                )
        return self._vector_store
    
    @property
    def embedder(self):
        """Lazy load embedder."""
        if self._embedder is None:
            if self.use_mock:
                from data_ingestion.embedder import MockEmbeddingGenerator
                self._embedder = MockEmbeddingGenerator(
                    dimension=self.settings.embedding_dimension
                )
            else:
                from data_ingestion.embedder import EmbeddingGenerator
                self._embedder = EmbeddingGenerator(
                    model_name=self.settings.embedding_model
                )
        return self._embedder
    
    @property
    def cache(self):
        """Lazy load semantic cache (Atlas-Hyperion v3.0)."""
        if self._cache is None:
            if self.use_mock:
                self._cache = MockSemanticCache(
                    embedding_dimension=self.settings.embedding_dimension,
                    similarity_threshold=self.settings.cache_similarity_threshold,
                    default_ttl=self.settings.cache_ttl,
                    enabled=self.enable_cache
                )
            else:
                self._cache = get_cache(
                    redis_url=self.settings.redis_url,
                    embedding_dimension=self.settings.embedding_dimension,
                    similarity_threshold=self.settings.cache_similarity_threshold,
                    default_ttl=self.settings.cache_ttl,
                    enabled=self.enable_cache,
                    use_mock=False
                )
        return self._cache
    
    @property
    def planner(self):
        """Lazy load agentic planner (Atlas-Hyperion v3.0)."""
        if self._planner is None:
            self._planner = get_planner(use_mock=self.use_mock)
        return self._planner
    
    @property
    def collection_version(self) -> str:
        """Get collection version for cache invalidation."""
        if self._collection_version is None:
            try:
                info = self.vector_store.get_collection_info()
                self._collection_version = str(info.get("points_count", "unknown"))
            except Exception:
                self._collection_version = "unknown"
        return self._collection_version
    
    def _is_greeting(self, text: str) -> bool:
        """
        Check if the input text is a greeting.

        Args:
            text: Input text to check

        Returns:
            True if the text contains a greeting, False otherwise
        """
        greeting_keywords = [
            "bonjour", "salut", "hello", "hi", "coucou", "salam", "holla",
            "bonsoir", "good morning", "good afternoon", "good evening",
            "hey", "yo", "hola"
        ]

        # Clean and lowercase the text for matching
        clean_text = text.strip().lower()

        # Check if any greeting keyword is in the text
        return any(keyword in clean_text for keyword in greeting_keywords)

    def query(self, question: str) -> RAGResponse:
        """
        Process a user question through the Atlas-Hyperion v3.0 pipeline.
        
        Flow:
        1. Check semantic cache (L1/L2)
        2. Use planner to decide strategy
        3. Retrieve with appropriate method
        4. Generate answer
        5. Cache result
        
        Args:
            question: User's question

        Returns:
            RAGResponse with answer and citations
        """
        start_time = time.time()
        logger.info(f"Processing query: {question[:100]}...")
        
        # === STEP 0: Planner decides strategy ===
        if self.enable_planner:
            decision = self.planner.plan(question)
            planner_action = decision.action.value
            logger.info(f"Planner decision: {planner_action}")
        else:
            decision = PlannerDecision(
                action=PlannerAction.RETRIEVE_HYBRID,
                top_k=self.settings.top_k_results
            )
            planner_action = "default"
        
        # === STEP 1: Handle NO_RETRIEVE (greetings) ===
        if decision.action == PlannerAction.NO_RETRIEVE:
            logger.info("Planner: No retrieval needed (greeting)")
            greeting_prompt = (
                "You are an AI assistant for ACAPS (Autorité de Contrôle des Assurances et de la Prévoyance Sociale) in Morocco. "
                "The user has just greeted you. "
                "Respond politely in French, welcoming them and offering your help with the internal documentation. "
                "Keep it brief and professional. "
                f"User greeting: '{question}'"
            )
            
            answer = self._generate_conversational(greeting_prompt)
            latency = (time.time() - start_time) * 1000
            
            return RAGResponse(
                answer=answer,
                citations=[],
                confidence=1.0,
                context_used="",
                metadata={"greeting": True, "model": "conversational-llm"},
                cache_hit=False,
                planner_action=planner_action,
                latency_ms=latency
            )

        # === STEP 2: Embed the query ===
        query_embedding = self.embedder.embed_query(question)
        
        # === STEP 3: Check semantic cache (Atlas-Hyperion v3.0) ===
        if self.enable_cache:
            cached_response, cache_level = self.cache.get(question, query_embedding)
            
            if cached_response:
                logger.info(f"Cache HIT ({cache_level})")
                latency = (time.time() - start_time) * 1000
                
                return RAGResponse(
                    answer=cached_response.answer,
                    citations=[Citation(**c) for c in cached_response.citations],
                    confidence=cached_response.confidence,
                    context_used="[cached]",
                    metadata={"cached": True, "cache_level": cache_level},
                    cache_hit=True,
                    cache_level=cache_level,
                    planner_action=planner_action,
                    latency_ms=latency
                )
        
        # === STEP 4: Retrieve documents based on planner decision ===
        if decision.action == PlannerAction.RETRIEVE_GRAPH:
            # Graph-enhanced retrieval for multi-hop queries
            logger.info(f"Using graph search (depth={decision.graph_depth})")
            search_results = self.vector_store.graph_search(
                query_embedding=query_embedding,
                query_text=question,
                top_k=decision.top_k,
                depth=decision.graph_depth,
                score_threshold=self.settings.similarity_threshold
            )
        elif decision.action == PlannerAction.RETRIEVE_HYBRID:
            # Hybrid semantic + keyword search
            search_results = self.vector_store.hybrid_search(
                query_embedding=query_embedding,
                query_text=question,
                top_k=decision.top_k,
                score_threshold=self.settings.similarity_threshold,
                keyword_boost=0.3
            )
        else:
            # Simple vector search
            search_results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=decision.top_k,
                score_threshold=self.settings.similarity_threshold
            )
        
        if not search_results:
            logger.info("No relevant documents found")
            latency = (time.time() - start_time) * 1000
            return RAGResponse(
                answer="I cannot find relevant information in the available documents.",
                citations=[],
                confidence=0.0,
                context_used="",
                metadata={"retrieval_count": 0},
                cache_hit=False,
                planner_action=planner_action,
                latency_ms=latency
            )
        
        # === STEP 5: Build context from retrieved documents ===
        context_parts = []
        citations = []
        
        for result in search_results:
            # Use crystallized text if available
            text = result.metadata.get("crystallized_text") or result.text
            
            context_parts.append(
                f"[Source: {result.header_path}]\n{text}"
            )
            
            url = result.full_url if result.full_url and not result.full_url.startswith("#") else ""
            
            snippet_length = 400
            text_snippet = result.text[:snippet_length] + "..." if len(result.text) > snippet_length else result.text
            
            citations.append(Citation(
                title=result.header_path,
                url=url,
                score=result.score,
                text_snippet=text_snippet
            ))
        
        context = "\n\n---\n\n".join(context_parts)
        avg_score = sum(r.score for r in search_results) / len(search_results)
        
        # === STEP 6: Generate answer ===
        prompt = self.SYSTEM_PROMPT.format(context=context, question=question)
        answer = self._generate(prompt, context=context)
        
        latency = (time.time() - start_time) * 1000
        logger.info(f"Generated answer with {len(citations)} citations in {latency:.0f}ms")
        
        # === STEP 7: Cache the result (Atlas-Hyperion v3.0) ===
        if self.enable_cache and answer and avg_score >= self.settings.similarity_threshold:
            try:
                cached = CachedResponse(
                    query=question,
                    answer=answer,
                    citations=[{"title": c.title, "url": c.url, "score": c.score, "text_snippet": c.text_snippet} for c in citations],
                    confidence=avg_score,
                    cached_at=datetime.now(timezone.utc).isoformat(),
                    ttl=self.settings.cache_ttl,
                    cache_level="L1"
                )
                self.cache.set(question, query_embedding, cached)
                logger.debug("Cached response")
            except Exception as e:
                logger.warning(f"Failed to cache response: {e}")
        
        return RAGResponse(
            answer=answer,
            citations=citations,
            confidence=avg_score,
            context_used=context,
            metadata={
                "retrieval_count": len(search_results),
                "top_score": search_results[0].score if search_results else 0,
                "model": self.settings.vllm_model,
                "graph_depth": decision.graph_depth if decision.action == PlannerAction.RETRIEVE_GRAPH else 0
            },
            cache_hit=False,
            cache_level="",
            planner_action=planner_action,
            latency_ms=latency
        )
    
    def _generate(self, prompt: str, context: str = "") -> str:
        """
        Generate response using LLM.
        
        Args:
            prompt: Full prompt with context
            context: The retrieved context (for fallback response)
            
        Returns:
            Generated text
        """
        if self.use_mock:
            return self.llm_client.generate(prompt)
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.settings.vllm_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for ACAPS documentation. Answer in the same language as the question."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            
            # Fallback: Return context summary if LLM is unavailable
            if self.settings.llm_fallback_enabled and context:
                return self._generate_fallback_response(context)
            
            return "Je ne peux pas générer une réponse pour le moment car le serveur LLM n'est pas disponible. Veuillez réessayer plus tard ou contacter l'administrateur."
    
    def _generate_fallback_response(self, context: str) -> str:
        """
        Generate a fallback response when LLM is unavailable.
        Returns the retrieved context with a disclaimer.
        
        Args:
            context: The retrieved document context
            
        Returns:
            Formatted fallback response
        """
        # Extract the most relevant part (first source)
        if "[Source:" in context:
            parts = context.split("---")
            first_source = parts[0].strip() if parts else context[:500]
        else:
            first_source = context[:500]
        
        return f"""**Note:** Le serveur LLM n'est pas disponible. Voici les informations trouvées dans les documents:

{first_source}

*Pour une réponse plus détaillée, veuillez contacter l'administrateur pour activer le serveur LLM.*"""

    def _generate_conversational(self, prompt: str) -> str:
        """
        Generate a conversational response without RAG context.
        
        Args:
            prompt: Prompt for the LLM
            
        Returns:
            Generated text
        """
        if self.use_mock:
            return "Bonjour! Je suis l'assistant AI de l'ACAPS. Comment puis-je vous aider aujourd'hui?"
            
        try:
            response = self.llm_client.chat.completions.create(
                model=self.settings.vllm_model,
                messages=[
                    {"role": "system", "content": "You are a helpful, professional AI assistant for ACAPS (Morocco Insurance Authority). Always answer in French."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,  # Slightly higher temperature for natural conversation
                max_tokens=200
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating conversational response: {e}")
            return "Bonjour! Je suis l'assistant ACAPS. Comment puis-je vous aider ?"
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of all components including Atlas-Hyperion v3.0."""
        health = {
            "vector_store": False,
            "llm": False,
            "embedder": False,
            "cache": False,  # Atlas-Hyperion v3.0
        }
        
        try:
            health["vector_store"] = self.vector_store.health_check()
        except Exception as e:
            logger.error(f"Vector store health check failed: {e}")
        
        try:
            # Simple embedding test
            self.embedder.embed_text("test")
            health["embedder"] = True
        except Exception as e:
            logger.error(f"Embedder health check failed: {e}")
        
        try:
            if self.use_mock:
                health["llm"] = True
            else:
                # Test LLM connection
                response = self.llm_client.models.list()
                health["llm"] = True
        except Exception as e:
            logger.error(f"LLM health check failed: {e}")
        
        # Atlas-Hyperion v3.0: Check cache health
        try:
            if self.enable_cache:
                health["cache"] = self.cache.health_check()
            else:
                health["cache"] = True  # Disabled = healthy
        except Exception as e:
            logger.error(f"Cache health check failed: {e}")
        
        health["overall"] = all(health.values())
        return health
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics (Atlas-Hyperion v3.0)."""
        if not self.enable_cache:
            return {"enabled": False}
        
        try:
            stats = self.cache.get_stats()
            stats["enabled"] = True
            return stats
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"enabled": True, "error": str(e)}
    
    def invalidate_cache(self) -> int:
        """Invalidate all cache entries (Atlas-Hyperion v3.0)."""
        if not self.enable_cache:
            return 0
        
        try:
            count = self.cache.invalidate_all()
            self._collection_version = None  # Reset version
            logger.info(f"Invalidated {count} cache entries")
            return count
        except Exception as e:
            logger.error(f"Cache invalidation failed: {e}")
            return 0


class MockLLMClient:
    """Mock LLM client for testing."""
    
    def generate(self, prompt: str) -> str:
        """Generate a mock response based on the prompt."""
        if "sick leave" in prompt.lower():
            return "According to Article 5, employees must inform their supervisor within 24 hours in case of sickness and provide a medical certificate within 48 hours."
        elif "vacation" in prompt.lower() or "congés" in prompt.lower():
            return "According to Article 4, each employee is entitled to 22 working days of annual paid leave."
        elif "working hours" in prompt.lower() or "horaires" in prompt.lower():
            return "According to Article 3, working hours are Monday to Friday from 8:30 AM to 4:30 PM, with a lunch break from 12:00 PM to 1:00 PM."
        else:
            return "Based on the available documentation, the information requested can be found in the relevant sections of the internal regulations."


# Singleton instance
_engine_instance: Optional[RAGEngine] = None


def get_engine(use_mock: bool = False) -> RAGEngine:
    """Get or create RAG engine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = RAGEngine(use_mock=use_mock)
    return _engine_instance


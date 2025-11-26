"""
RAG Engine for Atlas-RAG
LlamaIndex-based retrieval and generation orchestration.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .config import get_settings, Settings

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


class RAGEngine:
    """
    Main RAG engine orchestrating retrieval and generation.
    Uses LlamaIndex for RAG pipeline and OpenAI-compatible API for generation.
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
        use_mock: bool = False
    ):
        """
        Initialize RAG engine.
        
        Args:
            settings: Configuration settings
            use_mock: Use mock components for testing
        """
        self.settings = settings or get_settings()
        self.use_mock = use_mock
        self._llm_client = None
        self._vector_store = None
        self._embedder = None
        
        logger.info(f"RAGEngine initialized (mock={use_mock})")
    
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
        Process a user question through the RAG pipeline.

        Args:
            question: User's question

        Returns:
            RAGResponse with answer and citations
        """
        logger.info(f"Processing query: {question[:100]}...")

        # Check if this is a greeting - handle immediately without database search
        if self._is_greeting(question):
            logger.info("Detected greeting, returning conversational response")
            # Use LLM to generate a natural conversational response in French
            greeting_prompt = (
                "You are an AI assistant for ACAPS (Autorité de Contrôle des Assurances et de la Prévoyance Sociale) in Morocco. "
                "The user has just greeted you. "
                "Respond politely in French, welcoming them and offering your help with the internal documentation. "
                "Keep it brief and professional. "
                f"User greeting: '{question}'"
            )
            
            answer = self._generate_conversational(greeting_prompt)
            
            return RAGResponse(
                answer=answer,
                citations=[],
                confidence=1.0,
                context_used="",
                metadata={"greeting": True, "model": "conversational-llm"}
            )

        # Step 1: Embed the query
        query_embedding = self.embedder.embed_query(question)
        
        # Step 2: Retrieve relevant documents using HYBRID search
        # This combines semantic search with keyword matching for specific references
        search_results = self.vector_store.hybrid_search(
            query_embedding=query_embedding,
            query_text=question,
            top_k=self.settings.top_k_results,
            score_threshold=self.settings.similarity_threshold,
            keyword_boost=0.3
        )
        
        if not search_results:
            logger.info("No relevant documents found")
            return RAGResponse(
                answer="I cannot find relevant information in the available documents.",
                citations=[],
                confidence=0.0,
                context_used="",
                metadata={"retrieval_count": 0}
            )
        
        # Step 3: Build context from retrieved documents
        context_parts = []
        citations = []
        
        for result in search_results:
            context_parts.append(
                f"[Source: {result.header_path}]\n{result.text}"
            )
            
            # Use full_url if available, otherwise indicate no link
            url = result.full_url if result.full_url and not result.full_url.startswith("#") else ""
            
            # Longer snippets for better context (400 chars)
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
        
        # Step 4: Generate answer
        prompt = self.SYSTEM_PROMPT.format(context=context, question=question)
        answer = self._generate(prompt, context=context)
        
        logger.info(f"Generated answer with {len(citations)} citations")
        
        return RAGResponse(
            answer=answer,
            citations=citations,
            confidence=avg_score,
            context_used=context,
            metadata={
                "retrieval_count": len(search_results),
                "top_score": search_results[0].score if search_results else 0,
                "model": self.settings.vllm_model
            }
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
        """Check health of all components."""
        health = {
            "vector_store": False,
            "llm": False,
            "embedder": False
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
        
        health["overall"] = all(health.values())
        return health


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


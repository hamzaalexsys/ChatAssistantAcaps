"""
Comprehensive tests for Atlas-Hyperion v3.0 components.

Tests:
- Semantic Cache (L1/L2/L3)
- Agentic Planner
- NLI Verifier
- Contextualizer
- Graph Extractor
- Telemetry
"""
import pytest
import hashlib
from datetime import datetime, timezone
from typing import List

# =====================================================
# Cache Tests
# =====================================================

class TestSemanticCache:
    """Tests for the semantic cache layer."""
    
    def test_mock_cache_initialization(self):
        """Test MockSemanticCache initializes correctly."""
        from backend_api.app.cache import MockSemanticCache
        
        cache = MockSemanticCache(
            embedding_dimension=1024,
            similarity_threshold=0.95,
            default_ttl=3600,
            enabled=True
        )
        
        assert cache.embedding_dimension == 1024
        assert cache.similarity_threshold == 0.95
        assert cache.enabled is True
        assert cache.health_check() is True
    
    def test_mock_cache_disabled(self):
        """Test disabled cache returns None."""
        from backend_api.app.cache import MockSemanticCache, CachedResponse
        
        cache = MockSemanticCache(enabled=False)
        
        # Should return None when disabled
        response = cache.get_l2("test query")
        assert response is None
        
        # Set should return False when disabled
        cached = CachedResponse(
            query="test",
            answer="answer",
            citations=[],
            confidence=0.9,
            cached_at=datetime.now(timezone.utc).isoformat(),
            ttl=3600,
            cache_level="L2"
        )
        assert cache.set_l2("test", cached) is False
    
    def test_l2_cache_exact_match(self):
        """Test L2 cache stores and retrieves exact matches."""
        from backend_api.app.cache import MockSemanticCache, CachedResponse
        
        cache = MockSemanticCache(enabled=True)
        
        cached = CachedResponse(
            query="What is Article 5?",
            answer="Article 5 covers vacation policy.",
            citations=[{"title": "Article 5", "url": "", "score": 0.9, "text_snippet": "..."}],
            confidence=0.9,
            cached_at=datetime.now(timezone.utc).isoformat(),
            ttl=3600,
            cache_level="L2"
        )
        
        # Store
        assert cache.set_l2("What is Article 5?", cached) is True
        
        # Retrieve
        result = cache.get_l2("What is Article 5?")
        assert result is not None
        assert result.answer == "Article 5 covers vacation policy."
        assert result.confidence == 0.9
    
    def test_l2_cache_miss(self):
        """Test L2 cache returns None for missing queries."""
        from backend_api.app.cache import MockSemanticCache
        
        cache = MockSemanticCache(enabled=True)
        result = cache.get_l2("unknown query")
        assert result is None
    
    def test_l1_cache_similarity(self):
        """Test L1 cache uses embedding similarity."""
        from backend_api.app.cache import MockSemanticCache, CachedResponse
        import numpy as np
        
        cache = MockSemanticCache(
            embedding_dimension=4,
            similarity_threshold=0.9,
            enabled=True
        )
        
        # Create a normalized embedding
        embedding = [0.5, 0.5, 0.5, 0.5]
        norm = np.linalg.norm(embedding)
        embedding = [x / norm for x in embedding]
        
        cached = CachedResponse(
            query="test",
            answer="test answer",
            citations=[],
            confidence=0.9,
            cached_at=datetime.now(timezone.utc).isoformat(),
            ttl=3600,
            cache_level="L1"
        )
        
        # Store with embedding
        assert cache.set_l1("test", embedding, cached) is True
        
        # Query with identical embedding should hit
        result = cache.get_l1(embedding)
        assert result is not None
        assert result.answer == "test answer"
    
    def test_l3_cache_invalidation(self):
        """Test L3 cache invalidates on collection version change."""
        from backend_api.app.cache import MockSemanticCache, CachedRetrieval
        
        cache = MockSemanticCache(enabled=True)
        
        # Store with version 1
        cache.set_l3(
            query="test",
            doc_ids=["doc1", "doc2"],
            scores=[0.9, 0.8],
            collection_version="v1"
        )
        
        # Retrieve with same version - should hit
        result = cache.get_l3("test", "v1")
        assert result is not None
        assert result.doc_ids == ["doc1", "doc2"]
        
        # Retrieve with different version - should miss
        result = cache.get_l3("test", "v2")
        assert result is None
    
    def test_cache_invalidate_all(self):
        """Test invalidate_all clears all caches."""
        from backend_api.app.cache import MockSemanticCache, CachedResponse
        
        cache = MockSemanticCache(enabled=True)
        
        cached = CachedResponse(
            query="test",
            answer="answer",
            citations=[],
            confidence=0.9,
            cached_at=datetime.now(timezone.utc).isoformat(),
            ttl=3600,
            cache_level="L2"
        )
        
        # Store some entries
        cache.set_l2("query1", cached)
        cache.set_l2("query2", cached)
        
        # Verify they exist
        assert cache.get_l2("query1") is not None
        
        # Invalidate
        count = cache.invalidate_all()
        assert count >= 2
        
        # Verify they're gone
        assert cache.get_l2("query1") is None
    
    def test_cache_stats(self):
        """Test get_stats returns correct statistics."""
        from backend_api.app.cache import MockSemanticCache, CachedResponse
        
        cache = MockSemanticCache(enabled=True)
        
        cached = CachedResponse(
            query="test",
            answer="answer",
            citations=[],
            confidence=0.9,
            cached_at=datetime.now(timezone.utc).isoformat(),
            ttl=3600,
            cache_level="L2"
        )
        
        cache.set_l2("q1", cached)
        cache.set_l2("q2", cached)
        
        stats = cache.get_stats()
        assert stats["status"] == "mock"
        assert stats["l2_entries"] == 2
        assert stats["total_entries"] >= 2


# =====================================================
# Planner Tests
# =====================================================

class TestAgenticPlanner:
    """Tests for the agentic planner."""
    
    def test_planner_initialization(self):
        """Test planner initializes correctly."""
        from backend_api.app.planner import AgenticPlanner
        
        planner = AgenticPlanner(
            default_top_k=5,
            default_graph_depth=2,
            enable_verification=True
        )
        
        assert planner.default_top_k == 5
        assert planner.default_graph_depth == 2
        assert planner.enable_verification is True
    
    def test_planner_greeting_detection(self):
        """Test planner detects greetings and returns NO_RETRIEVE."""
        from backend_api.app.planner import AgenticPlanner, PlannerAction
        
        planner = AgenticPlanner()
        
        greetings = [
            "Bonjour!",
            "Hello",
            "Hi there",
            "Salut",
            "Hey",
        ]
        
        for greeting in greetings:
            decision = planner.plan(greeting)
            assert decision.action == PlannerAction.NO_RETRIEVE, f"Failed for: {greeting}"
            assert decision.should_verify is False
    
    def test_planner_simple_query(self):
        """Test planner handles simple queries."""
        from backend_api.app.planner import AgenticPlanner, PlannerAction
        
        planner = AgenticPlanner()
        
        simple_queries = [
            "What is Article 5?",
            "Where is the HR policy?",
            "Who is the director?",
        ]
        
        for query in simple_queries:
            decision = planner.plan(query)
            assert decision.action in [PlannerAction.RETRIEVE_SIMPLE, PlannerAction.RETRIEVE_HYBRID]
            assert decision.top_k >= 5
    
    def test_planner_complex_query(self):
        """Test planner handles complex queries with hybrid retrieval."""
        from backend_api.app.planner import AgenticPlanner, PlannerAction
        
        planner = AgenticPlanner()
        
        complex_queries = [
            "What are the differences between Article 5 and Article 10 regarding vacation policy?",
            "How does the sick leave policy compare to the vacation policy, and what exceptions apply?",
        ]
        
        for query in complex_queries:
            decision = planner.plan(query)
            # Complex queries should get more resources
            assert decision.top_k >= 5
    
    def test_planner_multi_hop_query(self):
        """Test planner detects multi-hop queries requiring graph expansion."""
        from backend_api.app.planner import AgenticPlanner, PlannerAction
        
        planner = AgenticPlanner()
        
        multi_hop_queries = [
            "Selon l'Article 5, quelles sont les exceptions définies dans l'Article 2?",
            "As defined in Section 3, and conformément à Article 7, what is the procedure?",
        ]
        
        for query in multi_hop_queries:
            decision = planner.plan(query)
            assert decision.action == PlannerAction.RETRIEVE_GRAPH, f"Failed for: {query}"
            assert decision.graph_depth >= 1
    
    def test_planner_should_reflect(self):
        """Test should_reflect detects low confidence."""
        from backend_api.app.planner import AgenticPlanner, PlannerDecision, PlannerAction
        
        planner = AgenticPlanner(enable_verification=True)
        
        decision = PlannerDecision(
            action=PlannerAction.RETRIEVE_HYBRID,
            top_k=5,
            confidence_threshold=0.75,
            should_verify=True
        )
        
        # Low confidence should trigger reflection
        should_reflect, reason = planner.should_reflect(
            response="This is a response.",
            confidence=0.5,
            decision=decision
        )
        assert should_reflect is True
        assert "confidence" in reason.lower()
        
        # High confidence should not trigger
        should_reflect, reason = planner.should_reflect(
            response="This is a detailed response with sufficient information about the policy that answers the question completely and thoroughly.",
            confidence=0.9,
            decision=decision
        )
        assert should_reflect is False
    
    def test_planner_replan_on_failure(self):
        """Test replanning escalates retrieval strategy."""
        from backend_api.app.planner import AgenticPlanner, PlannerDecision, PlannerAction
        
        planner = AgenticPlanner()
        
        original = PlannerDecision(
            action=PlannerAction.RETRIEVE_SIMPLE,
            top_k=5,
            graph_depth=0
        )
        
        new_decision = planner.replan_on_failure(original, "Low confidence")
        
        # Should escalate
        assert new_decision.action == PlannerAction.RETRIEVE_HYBRID
        assert new_decision.top_k > original.top_k


# =====================================================
# NLI Verifier Tests
# =====================================================

class TestNLIVerifier:
    """Tests for the NLI-based hallucination verifier."""
    
    def test_mock_verifier_always_passes(self):
        """Test mock verifier always passes."""
        from backend_api.app.guardrails.nli_verifier import MockNLIVerifier
        
        verifier = MockNLIVerifier()
        
        result = verifier.verify_response(
            response="This is any response.",
            context="This is any context."
        )
        
        assert result.passed is True
    
    def test_verifier_claim_extraction(self):
        """Test claim extraction from response."""
        from backend_api.app.guardrails.nli_verifier import NLIVerifier
        
        verifier = NLIVerifier(use_mock=True)
        
        response = "Article 5 defines vacation policy. Employees get 22 days of leave. The policy applies to all staff."
        
        claims = verifier._extract_claims(response)
        
        # Should extract multiple claims
        assert len(claims) >= 2
        
        # Claims should be sentences
        for claim in claims:
            assert len(claim.text.split()) >= 4
    
    def test_verifier_skips_questions(self):
        """Test claim extraction skips questions."""
        from backend_api.app.guardrails.nli_verifier import NLIVerifier
        
        verifier = NLIVerifier(use_mock=True)
        
        response = "What is the policy? Article 5 defines it. Is this clear?"
        
        claims = verifier._extract_claims(response)
        
        # Questions should be filtered out
        for claim in claims:
            assert not claim.text.endswith("?")
    
    def test_verifier_skips_meta_sentences(self):
        """Test claim extraction skips meta sentences."""
        from backend_api.app.guardrails.nli_verifier import NLIVerifier
        
        verifier = NLIVerifier(use_mock=True)
        
        response = "I cannot find the information. Based on the context, Article 5 applies. Please ask for clarification."
        
        claims = verifier._extract_claims(response)
        
        # Meta sentences should be filtered
        for claim in claims:
            assert not claim.text.lower().startswith("i cannot")
            assert not claim.text.lower().startswith("please")
    
    def test_verifier_mock_verification(self):
        """Test mock verification returns entailment."""
        from backend_api.app.guardrails.nli_verifier import NLIVerifier
        
        verifier = NLIVerifier(use_mock=True)
        
        result = verifier.verify_response(
            response="The vacation policy is defined in Article 5.",
            context="Article 5: Vacation Policy. All employees are entitled to 22 days of annual leave."
        )
        
        assert result.passed is True
    
    def test_correction_prompt_generation(self):
        """Test correction prompt is generated for failed verification."""
        from backend_api.app.guardrails.nli_verifier import (
            NLIVerifier, VerificationResult, Claim
        )
        
        verifier = NLIVerifier(use_mock=True)
        
        # Create a failed verification result
        result = VerificationResult(
            passed=False,
            claims=[],
            overall_score=0.3,
            unverified_claims=[
                Claim(text="Some unverified claim.", start_idx=0, end_idx=20, source_sentence="Some unverified claim.")
            ],
            contradicted_claims=[],
            explanation="Test failure"
        )
        
        prompt = verifier.get_correction_prompt("Original response", result)
        
        assert "UNVERIFIED" in prompt
        assert "Original response" in prompt


# =====================================================
# Contextualizer Tests
# =====================================================

class TestContextualizer:
    """Tests for contextual crystallization."""
    
    def test_simple_contextualizer(self):
        """Test SimpleContextualizer adds document context."""
        from data_ingestion.contextualizer import SimpleContextualizer
        
        contextualizer = SimpleContextualizer()
        
        result = contextualizer.crystallize(
            chunk_text="The limit is 5 days.",
            doc_title="HR Policy",
            header_path="Section 4 > Article 5 > Sick Leave",
            generate_summary=True
        )
        
        # Should include document context
        assert "HR Policy" in result.crystallized_text
        assert "Section 4" in result.crystallized_text
        
        # Summary should be generated
        assert len(result.summary) > 0
    
    def test_contextualizer_short_chunk(self):
        """Test contextualizer handles short chunks."""
        from data_ingestion.contextualizer import SimpleContextualizer
        
        contextualizer = SimpleContextualizer()
        
        result = contextualizer.crystallize(
            chunk_text="Short.",
            doc_title="Doc",
            header_path="Section 1"
        )
        
        # Should still work
        assert result.crystallized_text is not None
    
    def test_contextualizer_batch(self):
        """Test batch crystallization."""
        from data_ingestion.contextualizer import SimpleContextualizer
        
        contextualizer = SimpleContextualizer()
        
        chunks = [
            {"text": "First chunk content.", "header_path": "Section 1"},
            {"text": "Second chunk content.", "header_path": "Section 2"},
            {"text": "Third chunk content.", "header_path": "Section 3"},
        ]
        
        results = contextualizer.crystallize_batch(
            chunks=chunks,
            doc_title="Test Document",
            generate_summaries=True,
            show_progress=False
        )
        
        assert len(results) == 3
        for i, result in enumerate(results):
            assert f"Section {i+1}" in result.crystallized_text
    
    def test_mock_llm_contextualizer(self):
        """Test LLM-based contextualizer with mock."""
        from data_ingestion.contextualizer import ChunkContextualizer
        
        contextualizer = ChunkContextualizer(use_mock=True)
        
        result = contextualizer.crystallize(
            chunk_text="The limit is 5 days.",
            doc_title="HR Policy",
            header_path="Section 4 > Article 5",
            generate_summary=True
        )
        
        # Mock should add context prefix
        assert "HR Policy" in result.crystallized_text


# =====================================================
# Graph Extractor Tests  
# =====================================================

class TestGraphExtractor:
    """Tests for graph edge extraction."""
    
    def test_extractor_initialization(self):
        """Test GraphExtractor initializes correctly."""
        from data_ingestion.graph_extractor import GraphExtractor
        
        extractor = GraphExtractor()
        assert extractor is not None
    
    def test_article_extraction(self):
        """Test extraction of article references."""
        from data_ingestion.graph_extractor import GraphExtractor
        
        extractor = GraphExtractor()
        
        text = "Conformément à l'Article 5, les dispositions de l'Article 2 s'appliquent."
        
        refs = extractor.extract(text)
        
        assert "Article 5" in refs.article_refs
        assert "Article 2" in refs.article_refs
    
    def test_article_range_extraction(self):
        """Test extraction of article ranges."""
        from data_ingestion.graph_extractor import GraphExtractor
        
        extractor = GraphExtractor()
        
        text = "Les articles 5 à 10 définissent les procédures."
        
        refs = extractor.extract(text)
        
        # Should extract range
        assert any("Article" in ref for ref in refs.article_refs)
    
    def test_section_extraction(self):
        """Test extraction of section references."""
        from data_ingestion.graph_extractor import GraphExtractor
        
        extractor = GraphExtractor()
        
        text = "Voir la Section 3 et la Section 5 pour plus de détails."
        
        refs = extractor.extract(text)
        
        assert len(refs.section_refs) >= 2
    
    def test_law_extraction(self):
        """Test extraction of law references."""
        from data_ingestion.graph_extractor import GraphExtractor
        
        extractor = GraphExtractor()
        
        text = "Conformément à la Loi 17-99 relative au code des assurances."
        
        refs = extractor.extract(text)
        
        assert len(refs.external_refs) >= 1
    
    def test_simple_edge_extraction(self):
        """Test simple edge extraction returns strings."""
        from data_ingestion.graph_extractor import GraphExtractor
        
        extractor = GraphExtractor()
        
        text = "Article 5 fait référence à l'Article 2."
        
        edges = extractor.extract_edges_simple(text)
        
        assert isinstance(edges, list)
        assert all(isinstance(e, str) for e in edges)
    
    def test_no_references(self):
        """Test extraction with no references."""
        from data_ingestion.graph_extractor import GraphExtractor
        
        extractor = GraphExtractor()
        
        text = "This is a simple text without any legal references."
        
        refs = extractor.extract(text)
        
        assert len(refs.article_refs) == 0
        assert len(refs.section_refs) == 0


# =====================================================
# Telemetry Tests
# =====================================================

class TestTelemetry:
    """Tests for the telemetry module."""
    
    def test_telemetry_initialization(self):
        """Test Telemetry initializes correctly."""
        from backend_api.app.telemetry import Telemetry
        
        telemetry = Telemetry(enabled=True)
        
        assert telemetry.enabled is True
    
    def test_telemetry_disabled(self):
        """Test disabled telemetry doesn't log."""
        from backend_api.app.telemetry import Telemetry
        
        telemetry = Telemetry(enabled=False)
        
        result = telemetry.log_query(
            query="test",
            retrieval_ids=[],
            retrieval_scores=[],
            answer="answer",
            confidence=0.9,
            latency_ms=100,
            cache_hit=False,
            cache_level=""
        )
        
        assert result is None
    
    def test_log_query(self):
        """Test query logging."""
        from backend_api.app.telemetry import Telemetry
        
        telemetry = Telemetry(enabled=True)
        
        log = telemetry.log_query(
            query="What is Article 5?",
            retrieval_ids=["doc1", "doc2"],
            retrieval_scores=[0.9, 0.8],
            answer="Article 5 covers vacation policy.",
            confidence=0.85,
            latency_ms=150.5,
            cache_hit=False,
            cache_level="",
            planner_action="retrieve_hybrid",
            verification_passed=True
        )
        
        assert log is not None
        assert log.query == "What is Article 5?"
        assert log.confidence == 0.85
        assert log.latency_ms == 150.5
    
    def test_metrics_calculation(self):
        """Test metrics are calculated correctly."""
        from backend_api.app.telemetry import Telemetry
        
        telemetry = Telemetry(enabled=True)
        
        # Log some queries
        telemetry.log_query("q1", [], [], "a1", 0.9, 100, cache_hit=True, cache_level="L1")
        telemetry.log_query("q2", [], [], "a2", 0.8, 200, cache_hit=False, cache_level="")
        telemetry.log_query("q3", [], [], "a3", 0.7, 150, cache_hit=True, cache_level="L2")
        
        metrics = telemetry.get_metrics()
        
        assert metrics.total_queries == 3
        assert metrics.cache_hits == 2
        assert metrics.cache_misses == 1
        assert metrics.cache_hit_rate == pytest.approx(2/3, rel=0.01)
    
    def test_latency_percentiles(self):
        """Test latency percentile calculations."""
        from backend_api.app.telemetry import Telemetry
        
        telemetry = Telemetry(enabled=True)
        
        # Log queries with known latencies
        for latency in [100, 150, 200, 250, 300]:
            telemetry.log_query("q", [], [], "a", 0.9, latency, False, "")
        
        metrics = telemetry.get_metrics()
        
        assert metrics.avg_latency_ms == 200.0
        assert metrics.p50_latency_ms > 0
    
    def test_prometheus_format(self):
        """Test Prometheus metrics format."""
        from backend_api.app.telemetry import Telemetry
        
        telemetry = Telemetry(enabled=True)
        telemetry.log_query("q", [], [], "a", 0.9, 100, False, "")
        
        prometheus = telemetry.get_prometheus_metrics()
        
        assert "atlas_queries_total" in prometheus
        assert "atlas_cache_hit_rate" in prometheus
        assert "atlas_latency_ms" in prometheus
    
    def test_recent_logs(self):
        """Test retrieving recent logs."""
        from backend_api.app.telemetry import Telemetry
        
        telemetry = Telemetry(enabled=True)
        
        for i in range(5):
            telemetry.log_query(f"query_{i}", [], [], f"answer_{i}", 0.9, 100, False, "")
        
        logs = telemetry.get_recent_logs(limit=3)
        
        assert len(logs) == 3
        assert all("query" in log for log in logs)
    
    def test_reset(self):
        """Test telemetry reset."""
        from backend_api.app.telemetry import Telemetry
        
        telemetry = Telemetry(enabled=True)
        
        telemetry.log_query("q", [], [], "a", 0.9, 100, False, "")
        telemetry.log_error("test error")
        
        metrics_before = telemetry.get_metrics()
        assert metrics_before.total_queries == 1
        
        telemetry.reset()
        
        metrics_after = telemetry.get_metrics()
        assert metrics_after.total_queries == 0


# =====================================================
# Integration Tests
# =====================================================

class TestHyperionIntegration:
    """Integration tests for Atlas-Hyperion v3.0 components."""
    
    def test_parser_with_edges(self):
        """Test parser includes edges field."""
        from data_ingestion.parser import DocumentChunk
        
        chunk = DocumentChunk(
            text="Test content",
            file_name="test.md",
            header_path="Section 1",
            url_slug="",
            base_url="",
            last_updated="2024-01-01",
            chunk_index=0,
            edges=["Article 5", "Section 2"]
        )
        
        data = chunk.to_dict()
        
        assert "edges" in data
        assert data["edges"] == ["Article 5", "Section 2"]
        assert "crystallized_text" in data
    
    def test_planner_to_engine_flow(self):
        """Test planner decision affects retrieval."""
        from backend_api.app.planner import AgenticPlanner, PlannerAction
        
        planner = AgenticPlanner()
        
        # Simple query should use simple/hybrid retrieval
        decision1 = planner.plan("What is Article 5?")
        assert decision1.action in [PlannerAction.RETRIEVE_SIMPLE, PlannerAction.RETRIEVE_HYBRID]
        
        # Multi-hop query should use graph retrieval
        decision2 = planner.plan("Selon l'Article 5, quelles sont les exceptions de l'Article 2?")
        assert decision2.action == PlannerAction.RETRIEVE_GRAPH
    
    def test_contextualizer_to_embedder_flow(self):
        """Test crystallized text can be embedded."""
        from data_ingestion.contextualizer import SimpleContextualizer
        from data_ingestion.embedder import MockEmbeddingGenerator
        
        contextualizer = SimpleContextualizer()
        embedder = MockEmbeddingGenerator(dimension=1024)
        
        # Crystallize
        result = contextualizer.crystallize(
            chunk_text="The policy applies.",
            doc_title="HR Policy",
            header_path="Section 1"
        )
        
        # Embed crystallized text
        embedding_result = embedder.embed_text(result.crystallized_text)
        
        assert len(embedding_result.embedding) == 1024


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


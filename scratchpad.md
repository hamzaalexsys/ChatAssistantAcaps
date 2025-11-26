# Atlas-Hyperion v3.0 Project Scratchpad

## Task Overview
Building a **Cache-Reason-Verify** RAG system with Zero Hallucination policy for ACAPS internal use.

## Architecture Components (v3.0)
1. **Inference Engine (vLLM)** - Qwen 2.5-7B-Instruct
2. **Knowledge Store (Qdrant)** - Vector DB with graph edges
3. **Semantic Cache (Redis)** - L1/L2/L3 multi-level caching
4. **Agentic Planner** - Self-RAG style retrieval decisions
5. **3-Tier Guardrails** - Pattern + Suspicion + NLI verification
6. **Contextual Crystallization** - Chunk enrichment during ingestion
7. **Graph-Vector Hybrid** - Multi-hop retrieval
8. **Telemetry** - Metrics and evaluation plane

## Progress Tracking - Atlas-Hyperion v3.0
[X] Phase 1: Infrastructure Setup - Redis Stack + dependencies
[X] Phase 2: Semantic Cache Layer - L1/L2/L3 caching
[X] Phase 3: Contextual Crystallization - Chunk enrichment
[X] Phase 4: Graph Extractor - Edge extraction
[X] Phase 5: Update Ingestion Pipeline - Crystallizer + graph
[X] Phase 6: Graph Search - Multi-hop retrieval
[X] Phase 7: Agentic Planner - Self-RAG style
[X] Phase 8: NLI Verifier - Neural entailment
[X] Phase 9: Upgrade Guardrails - 3-tier verification
[X] Phase 10: Update Engine - Cache + planner integration
[X] Phase 11: Telemetry - Metrics + evaluation
[X] Phase 12: Tests - Comprehensive test suite

## New Files Created (v3.0)
- `backend_api/app/cache.py` - Semantic cache (L1/L2/L3)
- `backend_api/app/planner.py` - Agentic planner
- `backend_api/app/telemetry.py` - Metrics and logging
- `backend_api/app/guardrails/nli_verifier.py` - NLI verification
- `data_ingestion/contextualizer.py` - Chunk crystallization
- `data_ingestion/graph_extractor.py` - Edge extraction
- `tests/test_hyperion.py` - v3.0 component tests

## Modified Files (v3.0)
- `docker-compose.yml` - Added Redis Stack
- `requirements.txt` - Added redis package
- `backend_api/app/config.py` - Cache + NLI settings
- `backend_api/app/engine.py` - Cache + planner integration
- `backend_api/app/main.py` - Metrics endpoints
- `backend_api/app/guardrails/guardrails.py` - 3-tier verification
- `data_ingestion/parser.py` - edges + crystallized_text fields
- `data_ingestion/pipeline.py` - Contextualizer + graph steps
- `data_ingestion/vector_store.py` - graph_search method
- `env.example` - Cache + NLI configuration

## Key Technical Decisions (v3.0)
- **Semantic Cache**: Redis Stack with VSS for similarity matching
- **Cache Threshold**: 0.95 similarity for L1 hits
- **NLI Model**: cross-encoder/nli-deberta-v3-small
- **Suspicion Threshold**: 0.5 for uncertainty patterns
- **Graph Depth**: 2 hops for multi-hop queries
- **Planner**: Pattern-based query classification

## API Endpoints (v3.0)
- `GET /metrics` - JSON metrics
- `GET /metrics/prometheus` - Prometheus format
- `GET /metrics/logs` - Recent query logs
- `POST /cache/invalidate` - Clear all caches

## Commands
```bash
# Run all tests including v3.0
python -m pytest tests/ -v

# Run only Hyperion tests
python -m pytest tests/test_hyperion.py -v

# Start full stack with Redis
docker-compose up -d

# Invalidate cache after re-indexing
curl -X POST http://localhost:8080/cache/invalidate
```

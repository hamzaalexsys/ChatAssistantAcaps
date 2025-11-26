# Atlas-RAG Project Scratchpad

## Task Overview
Building a RAG chatbot system with Zero Hallucination policy for ACAPS internal use.

## Architecture Components
1. **Inference Engine (vLLM)** - Qwen 2.5-7B-Instruct
2. **Knowledge Store (Qdrant)** - Vector DB with 1024 dimensions
3. **Backend API (FastAPI + LlamaIndex)** - Orchestration layer
4. **Security (NeMo Guardrails)** - Input/Output filtering
5. **Data Pipeline** - Markdown parsing & embedding
6. **Frontend UI** - Chat interface

## Progress Tracking
[X] Phase 1: Project Structure Setup
[X] Phase 2: Data Pipeline (ETL) - Parser, Embedder, VectorStore, Pipeline
[X] Phase 3: Vector DB Setup (Qdrant) - Config and client
[X] Phase 4: Inference Engine (vLLM) - Docker config
[X] Phase 5: Guardrails - Input/Output validation
[X] Phase 6: Backend API - FastAPI + RAG Engine
[X] Phase 7: Frontend UI - React + TypeScript + Tailwind
[X] Phase 8: Docker Compose & Integration

## Test Results
- **Data Ingestion Tests**: 26/26 PASSED ✅
- **Guardrails Tests**: 17/17 PASSED ✅
- **Backend API Tests**: 18/18 PASSED ✅
- **TOTAL: 61/61 PASSED** ✅

## Key Technical Decisions
- Embedding Model: BAAI/bge-m3 (1024 dimensions)
- Re-ranker: BAAI/bge-reranker-v2-m3
- LLM: Qwen/Qwen2.5-7B-Instruct
- Temperature: 0 (no creativity for accuracy)
- Similarity threshold: 0.75 (below = "cannot find")

## Files Created
- `data_ingestion/` - Parser, embedder, vector_store, pipeline, config
- `backend_api/app/` - Main, engine, models, config, guardrails
- `frontend_ui/` - React TypeScript application
- `tests/` - Comprehensive test suite
- `docker-compose.yml` - Full orchestration
- `requirements.txt` - Python dependencies

## Lessons Learned
- Use BAAI/bge-m3 for multilingual embeddings
- Temperature 0 reduces hallucinations
- Mock components are essential for testing without external dependencies
- Cosine similarity can be negative for random vectors
- PowerShell doesn't support && operator

## Next Steps (Production)
1. Copy `env.example` to `.env` and configure
2. Run `docker-compose up -d` to start all services
3. Ingest documents using the pipeline
4. Access UI at http://localhost:3000

## Commands
```bash
# Run tests
python -m pytest tests/ -v

# Run backend (development)
python -m uvicorn backend_api.app.main:app --reload --port 8080

# Run frontend (development)
cd frontend_ui && npm run dev

# Start full stack with Docker
docker-compose up -d
```

"""
Main FastAPI Application for Atlas-Hyperion v3.0
API endpoints for chat, health, metrics, and administration.
"""
import logging
import uuid
from typing import Optional
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from .config import get_settings, Settings
from .models import (
    QueryRequest, QueryResponse, CitationSchema,
    ErrorResponse, HealthResponse,
    IngestRequest, IngestResponse,
    FeedbackRequest
)
from .engine import RAGEngine, get_engine
from .guardrails import get_guardrails, Guardrails, BlockReason
from .telemetry import Telemetry, get_telemetry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Get settings early for lifespan
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for startup and shutdown events."""
    # Startup
    logger.info("Atlas-Hyperion v3.0 API starting up...")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Guardrails enabled: {settings.enable_guardrails}")
    logger.info(f"Cache enabled: {settings.cache_enabled}")
    logger.info(f"NLI enabled: {settings.nli_enabled}")
    
    # Initialize telemetry
    telemetry = get_telemetry(enabled=True)
    logger.info("Telemetry initialized")
    
    yield
    
    # Shutdown
    logger.info("Atlas-Hyperion v3.0 API shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Atlas-Hyperion v3.0 API",
    description="Cache-Reason-Verify RAG system for ACAPS internal documentation",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency injection
def get_rag_engine() -> RAGEngine:
    """Get RAG engine instance."""
    return get_engine(use_mock=settings.debug)


def get_guardrails_instance() -> Guardrails:
    """Get guardrails instance."""
    return get_guardrails(
        enabled=settings.enable_guardrails,
        confidence_threshold=settings.similarity_threshold,
        enable_nli=settings.nli_enabled,
        use_mock=settings.debug
    )


def get_telemetry_instance() -> Telemetry:
    """Get telemetry instance."""
    return get_telemetry(enabled=True)


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="http_error",
            message=str(exc.detail),
            detail=None
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_error",
            message="An internal error occurred",
            detail=str(exc) if settings.debug else None
        ).model_dump()
    )


# Endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Atlas-RAG API",
        "version": "1.0.0",
        "description": "RAG chatbot for ACAPS documentation",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(engine: RAGEngine = Depends(get_rag_engine)):
    """Check health of all system components."""
    health = engine.health_check()
    
    return HealthResponse(
        status="healthy" if health.get("overall", False) else "unhealthy",
        components=health,
        version="1.0.0",
        timestamp=datetime.now(timezone.utc)
    )


@app.post("/chat", response_model=QueryResponse, tags=["Chat"])
async def chat(
    request: QueryRequest,
    engine: RAGEngine = Depends(get_rag_engine),
    guardrails: Guardrails = Depends(get_guardrails_instance),
    telemetry: Telemetry = Depends(get_telemetry_instance)
):
    """
    Process a chat query through Atlas-Hyperion v3.0 pipeline.
    
    Flow:
    1. Input guardrails validation
    2. Semantic cache check (L1/L2)
    3. Agentic planner decides retrieval strategy
    4. Graph-enhanced retrieval if needed
    5. LLM generation
    6. 3-tier output verification (Pattern + Suspicion + NLI)
    7. Response caching
    8. Telemetry logging
    """
    conversation_id = request.conversation_id or str(uuid.uuid4())
    
    logger.info(f"[{conversation_id}] Processing query: {request.question[:100]}...")
    
    # Step 1: Input guardrails
    input_check = guardrails.check_input(request.question)
    if input_check.should_block:
        logger.warning(f"[{conversation_id}] Input blocked: {input_check.blocked_reason}")
        return QueryResponse(
            answer=guardrails.get_blocked_response(input_check.blocked_reason),
            citations=[],
            confidence=0.0,
            conversation_id=conversation_id,
            metadata={"blocked": True, "reason": input_check.blocked_reason.value}
        )
    
    # Step 2: Process through RAG engine (includes cache, planner, graph search)
    try:
        rag_response = engine.query(request.question)
    except Exception as e:
        logger.error(f"[{conversation_id}] RAG engine error: {e}")
        telemetry.log_error(str(e), {"conversation_id": conversation_id})
        raise HTTPException(status_code=500, detail="Error processing query")
    
    # Step 3: Output guardrails with 3-tier verification
    output_check = guardrails.check_output(
        response=rag_response.answer,
        context=rag_response.context_used,
        retrieval_score=rag_response.confidence,
        run_nli=not rag_response.cache_hit  # Skip NLI for cached responses
    )
    
    verification_passed = not output_check.should_block
    
    if output_check.should_block:
        logger.warning(f"[{conversation_id}] Output blocked: {output_check.blocked_reason}")
        
        # Log blocked query
        telemetry.log_query(
            query=request.question,
            retrieval_ids=[c.title for c in rag_response.citations],
            retrieval_scores=[c.score for c in rag_response.citations],
            answer=guardrails.get_blocked_response(output_check.blocked_reason),
            confidence=output_check.confidence,
            latency_ms=rag_response.latency_ms,
            cache_hit=rag_response.cache_hit,
            cache_level=rag_response.cache_level,
            planner_action=rag_response.planner_action,
            verification_passed=False,
            verification_details=output_check.verification_details,
            session_id=conversation_id
        )
        
        return QueryResponse(
            answer=guardrails.get_blocked_response(output_check.blocked_reason),
            citations=[],
            confidence=output_check.confidence,
            conversation_id=conversation_id,
            metadata={
                "blocked": True, 
                "reason": output_check.blocked_reason.value,
                "tier_results": output_check.tier_results
            }
        )
    
    # Step 4: Format response
    citations = [
        CitationSchema(
            title=c.title,
            url=c.url,
            score=c.score,
            snippet=c.text_snippet
        )
        for c in rag_response.citations
    ]
    
    # Step 5: Log to telemetry
    telemetry.log_query(
        query=request.question,
        retrieval_ids=[c.title for c in rag_response.citations],
        retrieval_scores=[c.score for c in rag_response.citations],
        answer=rag_response.answer,
        confidence=rag_response.confidence,
        latency_ms=rag_response.latency_ms,
        cache_hit=rag_response.cache_hit,
        cache_level=rag_response.cache_level,
        planner_action=rag_response.planner_action,
        verification_passed=verification_passed,
        verification_details=output_check.verification_details,
        session_id=conversation_id
    )
    
    logger.info(f"[{conversation_id}] Response generated with {len(citations)} citations (cache={rag_response.cache_hit}, latency={rag_response.latency_ms:.0f}ms)")
    
    # Build metadata with Atlas-Hyperion v3.0 info
    metadata = {
        **rag_response.metadata,
        "cache_hit": rag_response.cache_hit,
        "cache_level": rag_response.cache_level,
        "planner_action": rag_response.planner_action,
        "latency_ms": rag_response.latency_ms,
        "verification_passed": verification_passed
    }
    
    return QueryResponse(
        answer=rag_response.answer,
        citations=citations,
        confidence=rag_response.confidence,
        conversation_id=conversation_id,
        metadata=metadata
    )


@app.post("/ingest", response_model=IngestResponse, tags=["Admin"])
async def ingest_documents(request: IngestRequest):
    """
    Ingest documents into the vector store.
    
    Admin endpoint for processing Markdown files.
    """
    from data_ingestion.pipeline import IngestionPipeline
    
    logger.info(f"Starting ingestion: file={request.file_path}, dir={request.directory}")
    
    try:
        pipeline = IngestionPipeline(use_mock=settings.debug)
        result = pipeline.run(
            source_dir=request.directory,
            file_path=request.file_path,
            recreate_collection=request.recreate_collection
        )
        
        return IngestResponse(
            success=result.success,
            files_processed=result.files_processed,
            chunks_created=result.chunks_created,
            chunks_upserted=result.chunks_upserted,
            errors=result.errors
        )
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback", tags=["Feedback"])
async def submit_feedback(request: FeedbackRequest):
    """
    Submit user feedback for an answer.
    
    Used to improve the system over time.
    """
    logger.info(f"Feedback received for {request.conversation_id}: rating={request.rating}, helpful={request.helpful}")
    
    # In production, store feedback in a database
    # For now, just log it
    
    return {
        "status": "received",
        "conversation_id": request.conversation_id,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/stats", tags=["Admin"])
async def get_stats(engine: RAGEngine = Depends(get_rag_engine)):
    """Get system statistics."""
    try:
        # Get vector store info - ensure it's JSON-safe
        vector_info = {}
        try:
            raw_info = engine.vector_store.get_collection_info()
            if isinstance(raw_info, dict):
                vector_info = {
                    "name": raw_info.get("name", "unknown"),
                    "points_count": raw_info.get("points_count", 0),
                    "status": str(raw_info.get("status", "unknown"))
                }
        except Exception as ve:
            logger.warning(f"Failed to get vector store info: {ve}")
            vector_info = {"status": "unavailable"}
        
        # Get cache stats safely
        cache_stats = {}
        try:
            raw_cache_stats = engine.get_cache_stats()
            if isinstance(raw_cache_stats, dict):
                # Only include primitive values
                for k, v in raw_cache_stats.items():
                    if isinstance(v, (str, int, float, bool, type(None))):
                        cache_stats[str(k)] = v
        except Exception as ce:
            logger.warning(f"Failed to get cache stats: {ce}")
            cache_stats = {"status": "unavailable"}
        
        return {
            "vector_store": vector_info,
            "cache": cache_stats,
            "config": {
                "model": engine.settings.vllm_model,
                "embedding_model": engine.settings.embedding_model,
                "top_k": engine.settings.top_k_results,
                "threshold": engine.settings.similarity_threshold,
                "cache_enabled": engine.enable_cache,
                "nli_enabled": settings.nli_enabled
            }
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# Atlas-Hyperion v3.0 Metrics Endpoints
# =====================================================

@app.get("/metrics", tags=["Metrics"])
async def get_metrics(telemetry: Telemetry = Depends(get_telemetry_instance)):
    """
    Get system metrics in JSON format.
    
    Includes:
    - Query counts and cache hit rates
    - Latency percentiles (p50, p95, p99)
    - Verification pass rates
    - Planner action distribution
    """
    metrics = telemetry.get_metrics()
    return {
        "timestamp": metrics.timestamp,
        "queries": {
            "total": metrics.total_queries,
            "cache_hits": metrics.cache_hits,
            "cache_misses": metrics.cache_misses,
            "cache_hit_rate": metrics.cache_hit_rate
        },
        "latency_ms": {
            "avg": metrics.avg_latency_ms,
            "p50": metrics.p50_latency_ms,
            "p95": metrics.p95_latency_ms,
            "p99": metrics.p99_latency_ms
        },
        "verification": {
            "pass_rate": metrics.verification_pass_rate
        },
        "planner_actions": metrics.planner_actions,
        "errors": metrics.error_count,
        "uptime_seconds": telemetry.get_uptime()
    }


@app.get("/metrics/prometheus", tags=["Metrics"])
async def get_prometheus_metrics(telemetry: Telemetry = Depends(get_telemetry_instance)):
    """
    Get metrics in Prometheus format.
    
    Can be scraped by Prometheus server for monitoring.
    """
    return PlainTextResponse(
        content=telemetry.get_prometheus_metrics(),
        media_type="text/plain"
    )


@app.get("/metrics/logs", tags=["Metrics"])
async def get_recent_logs(
    limit: int = 100,
    telemetry: Telemetry = Depends(get_telemetry_instance)
):
    """
    Get recent query logs for debugging.
    
    Args:
        limit: Maximum number of logs to return (default: 100)
    """
    logs = telemetry.get_recent_logs(limit=min(limit, 1000))
    return {
        "count": len(logs),
        "logs": logs
    }


@app.post("/cache/invalidate", tags=["Admin"])
async def invalidate_cache(engine: RAGEngine = Depends(get_rag_engine)):
    """
    Invalidate all cache entries.
    
    Use after re-indexing documents.
    """
    count = engine.invalidate_cache()
    return {
        "status": "success",
        "entries_invalidated": count
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)


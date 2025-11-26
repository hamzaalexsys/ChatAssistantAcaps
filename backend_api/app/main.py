"""
Main FastAPI Application for Atlas-RAG
API endpoints for chat, health, and administration.
"""
import logging
import uuid
from typing import Optional
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import get_settings, Settings
from .models import (
    QueryRequest, QueryResponse, CitationSchema,
    ErrorResponse, HealthResponse,
    IngestRequest, IngestResponse,
    FeedbackRequest
)
from .engine import RAGEngine, get_engine
from .guardrails import get_guardrails, Guardrails, BlockReason

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
    logger.info("Atlas-RAG API starting up...")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Guardrails enabled: {settings.enable_guardrails}")
    yield
    # Shutdown
    logger.info("Atlas-RAG API shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Atlas-RAG API",
    description="RAG-based chatbot API for ACAPS internal documentation",
    version="1.0.0",
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
        confidence_threshold=settings.similarity_threshold
    )


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
    guardrails: Guardrails = Depends(get_guardrails_instance)
):
    """
    Process a chat query and return an answer with citations.
    
    The endpoint:
    1. Validates input through guardrails
    2. Retrieves relevant documents
    3. Generates an answer using the LLM
    4. Validates output through guardrails
    5. Returns answer with source citations
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
    
    # Step 2: Process through RAG engine
    try:
        rag_response = engine.query(request.question)
    except Exception as e:
        logger.error(f"[{conversation_id}] RAG engine error: {e}")
        raise HTTPException(status_code=500, detail="Error processing query")
    
    # Step 3: Output guardrails
    output_check = guardrails.check_output(
        response=rag_response.answer,
        context=rag_response.context_used,
        retrieval_score=rag_response.confidence
    )
    
    if output_check.should_block:
        logger.warning(f"[{conversation_id}] Output blocked: {output_check.blocked_reason}")
        return QueryResponse(
            answer=guardrails.get_blocked_response(output_check.blocked_reason),
            citations=[],
            confidence=output_check.confidence,
            conversation_id=conversation_id,
            metadata={"blocked": True, "reason": output_check.blocked_reason.value}
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
    
    logger.info(f"[{conversation_id}] Response generated with {len(citations)} citations")
    
    return QueryResponse(
        answer=rag_response.answer,
        citations=citations,
        confidence=rag_response.confidence,
        conversation_id=conversation_id,
        metadata=rag_response.metadata
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
        vector_info = engine.vector_store.get_collection_info()
        return {
            "vector_store": vector_info,
            "config": {
                "model": engine.settings.vllm_model,
                "embedding_model": engine.settings.embedding_model,
                "top_k": engine.settings.top_k_results,
                "threshold": engine.settings.similarity_threshold
            }
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)


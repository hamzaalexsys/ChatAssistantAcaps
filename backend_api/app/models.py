"""
Pydantic Models for Atlas-RAG API
Request and response schemas.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict


def utc_now():
    """Get current UTC time."""
    return datetime.now(timezone.utc)


class ChatMessage(BaseModel):
    """A single chat message."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: Optional[datetime] = Field(default_factory=utc_now)


class QueryRequest(BaseModel):
    """Request schema for chat queries."""
    question: str = Field(..., description="User's question", min_length=1, max_length=2000)
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID for context")
    language: Optional[str] = Field("auto", description="Response language (auto, fr, en, ar)")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "What does Article 5 say about sick leave?",
                "conversation_id": "conv-123",
                "language": "auto"
            }
        }
    )


class CitationSchema(BaseModel):
    """Citation information for a source."""
    title: str = Field(..., description="Document section title")
    url: str = Field(..., description="Link to the source document")
    score: float = Field(..., description="Relevance score (0-1)")
    snippet: str = Field(..., description="Text snippet from the source")


class QueryResponse(BaseModel):
    """Response schema for chat queries."""
    answer: str = Field(..., description="Generated answer")
    citations: List[CitationSchema] = Field(default_factory=list, description="Source citations")
    confidence: float = Field(..., description="Overall confidence score")
    conversation_id: Optional[str] = Field(None, description="Conversation ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "answer": "According to Article 5, employees must notify their supervisor within 24 hours...",
                "citations": [
                    {
                        "title": "Titre II > Article 5 - Congés de Maladie",
                        "url": "https://portal.acaps.ma/reglements#article-5",
                        "score": 0.92,
                        "snippet": "En cas de maladie, l'agent doit informer son responsable..."
                    }
                ],
                "confidence": 0.92,
                "conversation_id": "conv-123",
                "metadata": {"retrieval_count": 3, "model": "Qwen/Qwen2.5-7B-Instruct"}
            }
        }
    )


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "validation_error",
                "message": "Question cannot be empty",
                "detail": None
            }
        }
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Overall health status")
    components: Dict[str, bool] = Field(..., description="Individual component health")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(default_factory=utc_now)


class IngestRequest(BaseModel):
    """Request for document ingestion."""
    file_path: Optional[str] = Field(None, description="Single file to ingest")
    directory: Optional[str] = Field(None, description="Directory to ingest")
    recreate_collection: bool = Field(False, description="Whether to recreate the collection")


class IngestResponse(BaseModel):
    """Response from document ingestion."""
    success: bool
    files_processed: int
    chunks_created: int
    chunks_upserted: int
    errors: List[str] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    """User feedback on an answer."""
    conversation_id: str = Field(..., description="Conversation ID")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1-5")
    comment: Optional[str] = Field(None, description="Optional feedback comment")
    helpful: bool = Field(..., description="Whether the answer was helpful")


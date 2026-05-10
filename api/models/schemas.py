"""Pydantic v2 request/response schemas for the Starbucks RAG API."""

from typing import List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Incoming question payload for the /query endpoint."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Natural-language business question to answer.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of context chunks to retrieve.",
    )


class QueryResponse(BaseModel):
    """Full answer envelope returned by the /query endpoint."""

    answer: str
    sources: List[str]
    chunks_retrieved: int
    latency_ms: int
    groundedness_score: float
    intent: str


class HealthResponse(BaseModel):
    """Service health snapshot returned by GET /health."""

    status: str
    llm_provider: str
    index_vectors: int
    uptime_seconds: float


class MetricsResponse(BaseModel):
    """Aggregate usage statistics returned by GET /metrics."""

    total_queries: int
    avg_latency_ms: float
    avg_groundedness: float


class FeedbackRequest(BaseModel):
    """User feedback payload for the POST /feedback endpoint."""

    query: str
    answer: str
    rating: int = Field(..., ge=1, le=5, description="Star rating from 1 (worst) to 5 (best).")
    comment: Optional[str] = None

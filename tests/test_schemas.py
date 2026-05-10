"""Unit tests for api.models.schemas Pydantic v2 models."""

import pytest
from pydantic import ValidationError

from api.models.schemas import HealthResponse, QueryRequest, QueryResponse


def test_query_request_valid() -> None:
    """A well-formed QueryRequest must be created with the correct defaults."""
    req = QueryRequest(question="Why are sales dropping?")
    assert req.question == "Why are sales dropping?"
    assert req.top_k == 5


def test_query_request_too_short() -> None:
    """Questions shorter than 3 characters must raise ValidationError."""
    with pytest.raises(ValidationError):
        QueryRequest(question="ab")


def test_query_request_too_long() -> None:
    """Questions longer than 500 characters must raise ValidationError."""
    with pytest.raises(ValidationError):
        QueryRequest(question="x" * 501)


def test_query_response_creates() -> None:
    """QueryResponse must accept all fields and expose them with correct types."""
    response = QueryResponse(
        answer="Sales declined 12 percent in the Pacific Northwest region.",
        sources=["q3_report.md", "regional_summary.csv"],
        chunks_retrieved=5,
        latency_ms=450,
        groundedness_score=0.75,
        intent="general",
    )
    assert isinstance(response.answer, str)
    assert isinstance(response.sources, list)
    assert isinstance(response.chunks_retrieved, int)
    assert isinstance(response.latency_ms, int)
    assert isinstance(response.groundedness_score, float)
    assert isinstance(response.intent, str)


def test_health_response_creates() -> None:
    """HealthResponse must accept all fields correctly."""
    health = HealthResponse(
        status="healthy",
        llm_provider="mock",
        index_vectors=23656,
        uptime_seconds=120.5,
    )
    assert health.status == "healthy"
    assert health.index_vectors == 23656
    assert isinstance(health.uptime_seconds, float)


def test_query_request_top_k_bounds() -> None:
    """top_k must be rejected outside the [1, 20] range."""
    with pytest.raises(ValidationError):
        QueryRequest(question="Valid question here", top_k=0)
    with pytest.raises(ValidationError):
        QueryRequest(question="Valid question here", top_k=21)

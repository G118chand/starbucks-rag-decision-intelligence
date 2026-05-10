"""Starbucks RAG Decision Intelligence API — FastAPI application entry point."""

import asyncio
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

# ── project root on path ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.models.schemas import (
    FeedbackRequest,
    HealthResponse,
    MetricsResponse,
    QueryRequest,
    QueryResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("api.main")

# ── shared mutable state (populated during lifespan) ──────────────────────────
app_state: dict = {
    "start_time":        time.time(),
    "embedder":          None,
    "store":             None,
    "chunks":            None,
    "retriever":         None,
    "guardrails":        None,
    "total_queries":     0,
    "total_latency_ms":  0,
    "total_groundedness": 0.0,
}


# ── lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Load all heavy resources once at startup; release on shutdown."""
    from embeddings.embedder import EmbeddingEngine
    from langchain_core.documents import Document
    from orchestration.guardrails import GuardrailsValidator
    from retrieval.hybrid_retriever import HybridRetriever
    from vectorstore.faiss_store import FAISSVectorStore

    logger.info("Starting up — loading embedding model …")
    embedder = EmbeddingEngine()
    app_state["embedder"] = embedder

    logger.info("Loading FAISS index …")
    index_dir = str(PROJECT_ROOT / "vectorstore" / "faiss_index")
    store = FAISSVectorStore(index_dir=index_dir)
    try:
        store.load_index(embedder)
    except FileNotFoundError:
        logger.error(
            "FAISS index not found at %s — run scripts/build_index.py first", index_dir
        )
        raise
    app_state["store"] = store

    meta_path = PROJECT_ROOT / "vectorstore" / "faiss_index" / "metadata.json"
    logger.info("Loading chunk metadata from %s …", meta_path)
    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    chunks = [
        Document(page_content=m["text"], metadata=m["metadata"]) for m in meta
    ]
    app_state["chunks"] = chunks

    logger.info("Building HybridRetriever over %d chunks …", len(chunks))
    retriever = HybridRetriever(store, embedder, chunks)
    app_state["retriever"] = retriever

    app_state["guardrails"] = GuardrailsValidator()

    index_stats = store.get_stats()
    logger.info(
        "Startup complete. Index: %d vectors | %d dims | %.2f MB",
        index_stats["total_vectors"],
        index_stats["dimensions"],
        index_stats["index_size_mb"],
    )

    yield  # ── application runs here ──────────────────────────────────────────

    logger.info("Shutting down — releasing resources.")


# ── application ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Starbucks RAG Decision Intelligence API",
    description=(
        "Enterprise RAG platform that answers natural-language business questions "
        "about Starbucks operational data using hybrid retrieval and GPT-3.5."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse, summary="Answer a business question")
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """Retrieve relevant context and generate an LLM-powered answer.

    Args:
        request: :class:`~api.models.schemas.QueryRequest` with the question and top_k.

    Returns:
        :class:`~api.models.schemas.QueryResponse` with the answer, sources, and quality scores.

    Raises:
        HTTPException 503: Pipeline not ready (index not loaded).
        HTTPException 500: Internal error during retrieval or generation.
    """
    if app_state["retriever"] is None:
        raise HTTPException(
            status_code=503,
            detail="Service not ready — FAISS index is not loaded.",
        )

    try:
        from orchestration.agent import run_agent

        question: str = request.question
        result = await asyncio.to_thread(
            run_agent, question, app_state["retriever"]
        )

        guardrails_result = app_state["guardrails"].validate(
            result.answer, question, result.chunks
        )

        # update running totals
        app_state["total_queries"] += 1
        app_state["total_latency_ms"] += result.latency_ms
        app_state["total_groundedness"] += guardrails_result.groundedness_score

        # proxy latency estimate
        latency_ms: int = len(result.chunks) * 50 + 200

        # deduplicated sources preserving order
        seen: set = set()
        sources = []
        for chunk in result.chunks:
            src = chunk.get("metadata", {}).get("source", "Unknown")
            if src not in seen:
                seen.add(src)
                sources.append(src)

        logger.info(
            "Query: '%s…' | intent=%s | grounded=%.4f | passed=%s",
            question[:50], result.intent,
            guardrails_result.groundedness_score, guardrails_result.passed,
        )

        return QueryResponse(
            answer=result.answer,
            sources=sources,
            chunks_retrieved=len(result.chunks),
            latency_ms=latency_ms,
            groundedness_score=guardrails_result.groundedness_score,
            intent=result.intent,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Query endpoint error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health", response_model=HealthResponse, summary="Service health check")
async def health_endpoint() -> HealthResponse:
    """Return live service health including index stats and uptime.

    Returns:
        :class:`~api.models.schemas.HealthResponse`.
    """
    store = app_state["store"]
    status = "healthy" if store is not None else "degraded"
    index_vectors = store.get_stats().get("total_vectors", 0) if store else 0

    return HealthResponse(
        status=status,
        llm_provider=os.getenv("LLM_PROVIDER", "openai"),
        index_vectors=index_vectors,
        uptime_seconds=round(time.time() - app_state["start_time"], 2),
    )


@app.get("/metrics", response_model=MetricsResponse, summary="Aggregate usage metrics")
async def metrics_endpoint() -> MetricsResponse:
    """Return cumulative query statistics since the last server start.

    Returns:
        :class:`~api.models.schemas.MetricsResponse`.
    """
    total = app_state["total_queries"]
    return MetricsResponse(
        total_queries=total,
        avg_latency_ms=round(app_state["total_latency_ms"] / total, 1) if total else 0.0,
        avg_groundedness=round(app_state["total_groundedness"] / total, 4) if total else 0.0,
    )


@app.post("/feedback", summary="Record user feedback for a query/answer pair")
async def feedback_endpoint(request: FeedbackRequest) -> dict:
    """Append a feedback record to the JSONL feedback log.

    Args:
        request: :class:`~api.models.schemas.FeedbackRequest`.

    Returns:
        ``{"status": "recorded"}``
    """
    log_path = PROJECT_ROOT / "monitoring" / "feedback_log.jsonl"
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "query":   request.query,
        "answer":  request.answer,
        "rating":  request.rating,
        "comment": request.comment,
    }
    try:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("Feedback recorded — rating=%d", request.rating)
    except Exception as exc:
        logger.error("Failed to write feedback: %s", exc)
        raise HTTPException(status_code=500, detail="Could not record feedback.")

    return {"status": "recorded"}

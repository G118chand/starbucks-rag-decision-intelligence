"""Top-level agent entry point — thin wrapper that runs the RAGChain pipeline."""

import logging

from orchestration.rag_chain import RAGChain
from orchestration.schemas import RAGResult

logger = logging.getLogger(__name__)


def run_agent(question: str, retriever) -> RAGResult:
    """Run the full RAG pipeline for a user question.

    Instantiates a :class:`~orchestration.rag_chain.RAGChain` with the supplied
    retriever and invokes it synchronously.  The chain handles retrieval, LLM
    generation, and guardrail validation internally.

    Args:
        question:  Natural-language question from the user.
        retriever: A :class:`~retrieval.hybrid_retriever.HybridRetriever` instance
                   already loaded with the FAISS index.

    Returns:
        :class:`~orchestration.schemas.RAGResult` with the answer, sources,
        retrieved chunks, latency, and quality scores.
    """
    logger.info("run_agent — question: '%s'", question[:80])
    chain = RAGChain(retriever)
    result: RAGResult = chain.invoke(question)
    logger.info(
        "run_agent — complete | latency=%dms | grounded=%.4f",
        result.latency_ms, result.groundedness_score,
    )
    return result

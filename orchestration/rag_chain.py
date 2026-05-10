"""RAG chain — retrieval-augmented generation using LangChain LCEL."""

import logging
import time
from typing import List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from llm.llm_factory import get_llm
from orchestration.guardrails import GuardrailsValidator
from orchestration.schemas import RAGResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT: str = (
    "You are an enterprise AI Decision Intelligence assistant for Starbucks.\n\n"
    "You ONLY answer using the provided context documents below.\n"
    "Always cite the source name like: According to [source name]...\n"
    "Provide specific data points and metrics when available in the context.\n"
    "Give 2-3 actionable business recommendations at the end.\n"
    "If the context does not cover the question, say exactly: "
    "The available enterprise data does not cover this specific topic.\n"
    "Do NOT make up numbers or facts not present in the context.\n\n"
    "CONTEXT:\n{context}\n\n"
    "QUESTION: {question}\n\n"
    "ANSWER:"
)


class RAGChain:
    """Orchestrates retrieval, context formatting, LLM generation, and guardrail validation.

    The LangChain LCEL pipeline is: ``PromptTemplate | LLM | StrOutputParser``.
    Retrieved chunks come from a :class:`~retrieval.hybrid_retriever.HybridRetriever`
    and are formatted as numbered, source-labelled blocks before being injected into
    the prompt.
    """

    def __init__(self, retriever) -> None:
        """Build the LCEL chain and wire up the retriever.

        Args:
            retriever: Any object exposing ``retrieve(query, top_k) -> List[dict]``.
        """
        self.retriever = retriever
        self.llm = get_llm()
        self.prompt = PromptTemplate(
            template=SYSTEM_PROMPT,
            input_variables=["context", "question"],
        )
        self.chain = self.prompt | self.llm | StrOutputParser()
        self._guardrails = GuardrailsValidator()
        logger.info("RAGChain initialised with %s", type(self.llm).__name__)

    # ── private helpers ───────────────────────────────────────────────────────

    def _format_context(self, chunks: List[dict]) -> str:
        """Render retrieved chunks into a numbered, source-labelled context block.

        Args:
            chunks: Dicts with ``"text"`` and ``"metadata"`` keys as returned by
                    :class:`~retrieval.hybrid_retriever.HybridRetriever`.

        Returns:
            Multi-section string ready to be injected into the prompt.
        """
        parts: List[str] = []
        for i, chunk in enumerate(chunks):
            source = chunk.get("metadata", {}).get("source", "Unknown")
            parts.append(f"[Source {i + 1}: {source}]\n{chunk['text']}")
        return "\n\n---\n\n".join(parts)

    # ── public API ────────────────────────────────────────────────────────────

    def invoke(self, query: str, top_k: int = 5) -> RAGResult:
        """Run the full RAG pipeline for a single query.

        Steps:
        1. Hybrid retrieval (FAISS + BM25).
        2. Context formatting.
        3. LLM generation via the LCEL chain.
        4. Guardrail validation (groundedness, relevance, safety).

        Args:
            query:  Natural-language question from the user.
            top_k:  Number of chunks to retrieve and include in context.

        Returns:
            :class:`~orchestration.schemas.RAGResult` with the answer, sources,
            latency, and quality scores.
        """
        start = time.perf_counter()
        logger.info("RAGChain.invoke — query: '%s'", query)

        try:
            chunks: List[dict] = self.retriever.retrieve(query, top_k=top_k)
            context: str = self._format_context(chunks)

            answer: str = self.chain.invoke(
                {"context": context, "question": query}
            )
        except Exception as exc:
            logger.error("RAGChain pipeline failed: %s", exc)
            raise

        latency_ms: int = int((time.perf_counter() - start) * 1000)

        # deduplicate sources while preserving order
        seen: set = set()
        sources: List[str] = []
        for chunk in chunks:
            src = chunk.get("metadata", {}).get("source", "Unknown")
            if src not in seen:
                seen.add(src)
                sources.append(src)

        token_estimate: int = len(context.split()) + len(answer.split())

        guardrails_result = self._guardrails.validate(answer, query, chunks)

        logger.info(
            "RAGChain complete — latency=%dms | tokens≈%d | grounded=%.4f | passed=%s",
            latency_ms, token_estimate,
            guardrails_result.groundedness_score, guardrails_result.passed,
        )

        return RAGResult(
            answer=answer,
            sources=sources,
            chunks=chunks,
            latency_ms=latency_ms,
            token_estimate=token_estimate,
            groundedness_score=guardrails_result.groundedness_score,
        )

"""LangGraph StateGraph agent for the Starbucks RAG pipeline.

The graph has four nodes executed in sequence:
  retrieve → format_context → generate → validate

This makes each stage independently observable and testable, and provides
a natural extension point for conditional routing (e.g. re-ranking, fallback).
"""

import logging
import time
from typing import Any, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from llm.llm_factory import get_llm
from orchestration.guardrails import GuardrailsValidator
from orchestration.rag_chain import SYSTEM_PROMPT
from orchestration.schemas import RAGResult

logger = logging.getLogger(__name__)


# ── graph state schema ────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """Shared mutable state passed between graph nodes."""

    question: str
    top_k: int
    chunks: List[dict]
    context: str
    answer: str
    groundedness_score: float
    intent: str
    sources: List[str]


# ── graph factory ─────────────────────────────────────────────────────────────

def build_agent(retriever: Any):
    """Compile a LangGraph StateGraph wired to *retriever*.

    Args:
        retriever: A :class:`~retrieval.hybrid_retriever.HybridRetriever` instance.

    Returns:
        A compiled LangGraph runnable that accepts an :class:`AgentState` dict
        and returns a fully populated :class:`AgentState` dict.
    """
    guardrails = GuardrailsValidator()
    llm = get_llm()
    prompt = PromptTemplate(
        template=SYSTEM_PROMPT, input_variables=["context", "question"]
    )
    lcel_chain = prompt | llm | StrOutputParser()

    # ── node functions (each returns a partial state update) ─────────────────

    def retrieve(state: AgentState) -> dict:
        """Hybrid FAISS + BM25 retrieval."""
        chunks = retriever.retrieve(state["question"], top_k=state.get("top_k", 5))
        logger.info("retrieve node — %d chunks fetched", len(chunks))
        return {"chunks": chunks}

    def format_context(state: AgentState) -> dict:
        """Format retrieved chunks into a numbered, source-labelled context block."""
        parts: List[str] = []
        for i, chunk in enumerate(state["chunks"]):
            source = chunk.get("metadata", {}).get("source", "Unknown")
            parts.append(f"[Source {i + 1}: {source}]\n{chunk['text']}")
        context = "\n\n---\n\n".join(parts)
        return {"context": context}

    def generate(state: AgentState) -> dict:
        """Call the LLM via the LCEL chain."""
        answer: str = lcel_chain.invoke(
            {"context": state["context"], "question": state["question"]}
        )
        logger.info("generate node — answer length: %d chars", len(answer))
        return {"answer": answer}

    def validate(state: AgentState) -> dict:
        """Run guardrails and extract deduplicated sources."""
        gr = guardrails.validate(state["answer"], state["question"], state["chunks"])

        seen: set = set()
        sources: List[str] = []
        for chunk in state["chunks"]:
            src = chunk.get("metadata", {}).get("source", "Unknown")
            if src not in seen:
                seen.add(src)
                sources.append(src)

        logger.info(
            "validate node — grounded=%.4f passed=%s", gr.groundedness_score, gr.passed
        )
        return {
            "groundedness_score": gr.groundedness_score,
            "sources": sources,
            "intent": "general",
        }

    # ── wire the graph ────────────────────────────────────────────────────────
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("format_context", format_context)
    graph.add_node("generate", generate)
    graph.add_node("validate", validate)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "format_context")
    graph.add_edge("format_context", "generate")
    graph.add_edge("generate", "validate")
    graph.add_edge("validate", END)

    return graph.compile()


# ── public entry point ────────────────────────────────────────────────────────

def run_agent(question: str, retriever: Any) -> RAGResult:
    """Build the agent graph, run it for *question*, and return a :class:`RAGResult`.

    Args:
        question:  Natural-language business question.
        retriever: Loaded :class:`~retrieval.hybrid_retriever.HybridRetriever`.

    Returns:
        :class:`~orchestration.schemas.RAGResult` with answer, sources, chunks,
        latency, and quality scores.
    """
    logger.info("agent_graph.run_agent — '%s'", question[:80])
    start = time.perf_counter()

    agent = build_agent(retriever)

    initial_state: AgentState = {
        "question": question,
        "top_k": 5,
        "chunks": [],
        "context": "",
        "answer": "",
        "groundedness_score": 0.0,
        "intent": "general",
        "sources": [],
    }

    final_state: AgentState = agent.invoke(initial_state)
    latency_ms = int((time.perf_counter() - start) * 1000)

    token_estimate = (
        len(final_state["context"].split()) + len(final_state["answer"].split())
    )

    logger.info(
        "agent_graph complete — latency=%dms grounded=%.4f",
        latency_ms, final_state["groundedness_score"],
    )

    return RAGResult(
        answer=final_state["answer"],
        sources=final_state["sources"],
        chunks=final_state["chunks"],
        latency_ms=latency_ms,
        token_estimate=token_estimate,
        groundedness_score=final_state["groundedness_score"],
        intent=final_state["intent"],
    )

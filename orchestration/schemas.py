"""Shared dataclasses for RAG pipeline results and guardrail outcomes."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class RAGResult:
    """Complete output of a single RAG pipeline invocation.

    Attributes:
        answer:             Generated answer text from the LLM.
        sources:            Deduplicated list of source document names cited.
        chunks:             Retrieved chunks (text + metadata + scores) used to build context.
        latency_ms:         Wall-clock time for the full retrieval + generation cycle.
        token_estimate:     Rough token count (context words + answer words).
        intent:             Detected query intent label; defaults to ``"general"``.
        groundedness_score: Fraction of answer vocabulary grounded in retrieved context (0–1).
    """

    answer: str
    sources: List[str]
    chunks: List[dict]
    latency_ms: int
    token_estimate: int
    intent: str = "general"
    groundedness_score: float = 0.0


@dataclass
class GuardrailsResult:
    """Output of the guardrails validation step.

    Attributes:
        passed:             ``True`` when all three checks pass.
        groundedness_score: Word-overlap-based grounding fraction (0–1).
        is_relevant:        ``True`` when the answer addresses the query.
        is_safe:            ``True`` when no LLM refusal phrases are detected.
        reason:             Human-readable summary of the validation outcome.
    """

    passed: bool
    groundedness_score: float
    is_relevant: bool
    is_safe: bool
    reason: str

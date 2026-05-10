"""Lightweight post-generation guardrails for RAG answer validation."""

import logging
from typing import ClassVar, List, Set

from orchestration.schemas import GuardrailsResult

logger = logging.getLogger(__name__)

_STOPWORDS: Set[str] = {
    "the", "a", "an", "is", "are", "was", "were",
    "in", "on", "at", "to", "for", "of",
    "and", "or", "but", "it", "this", "that",
}


class GuardrailsValidator:
    """Validates a generated answer against three quality signals.

    Checks:
    1. **Groundedness** — fraction of answer vocabulary present in retrieved context.
    2. **Relevance**    — at least one query term appears in the answer.
    3. **Safety**       — answer does not contain LLM refusal phrases.
    """

    FAIL_PHRASES: ClassVar[List[str]] = [
        "i cannot",
        "i can't",
        "as an ai",
        "i don't have access",
    ]

    # ── individual checks ─────────────────────────────────────────────────────

    def check_groundedness(self, answer: str, chunks: List[dict]) -> float:
        """Estimate how well the answer is grounded in the retrieved chunks.

        Computes the fraction of unique, non-trivial answer words that also appear
        in the combined chunk texts.  A score ≥ 0.3 is considered adequately grounded.

        Args:
            answer: Generated answer string.
            chunks: Retrieved chunks; each dict must contain a ``"text"`` key.

        Returns:
            Grounding score in [0.0, 1.0].
        """
        if not answer or not chunks:
            return 0.0

        answer_words: Set[str] = {
            w for w in answer.lower().split() if w not in _STOPWORDS
        }
        if not answer_words:
            return 0.0

        chunk_words: Set[str] = set()
        for chunk in chunks:
            chunk_words.update(
                w for w in chunk.get("text", "").lower().split()
                if w not in _STOPWORDS
            )

        overlap = answer_words & chunk_words
        score = len(overlap) / len(answer_words)
        return round(min(score, 1.0), 4)

    def check_relevance(self, query: str, answer: str) -> bool:
        """Check whether the answer is topically relevant to the query.

        Args:
            query:  Original user query.
            answer: Generated answer string.

        Returns:
            ``True`` when the answer is ≥ 50 characters and shares at least one
            non-trivial word with the query.
        """
        if len(answer.strip()) < 50:
            return False

        query_words = {w.lower() for w in query.split() if w.lower() not in _STOPWORDS}
        answer_lower = answer.lower()
        return any(word in answer_lower for word in query_words)

    def check_safety(self, answer: str) -> bool:
        """Detect LLM refusal or capability-disclaimer phrases.

        Args:
            answer: Generated answer string.

        Returns:
            ``True`` when no refusal phrases are detected (answer is safe to serve).
        """
        answer_lower = answer.lower()
        return not any(phrase in answer_lower for phrase in self.FAIL_PHRASES)

    # ── composite validation ──────────────────────────────────────────────────

    def validate(
        self, answer: str, query: str, chunks: List[dict]
    ) -> GuardrailsResult:
        """Run all three checks and produce a consolidated validation result.

        Args:
            answer: Generated answer to validate.
            query:  Original user query (used for relevance check).
            chunks: Retrieved context chunks (used for groundedness check).

        Returns:
            :class:`GuardrailsResult` summarising the outcome of every check.
        """
        groundedness = self.check_groundedness(answer, chunks)
        is_relevant = self.check_relevance(query, answer)
        is_safe = self.check_safety(answer)

        passed = groundedness >= 0.3 and is_relevant and is_safe

        if passed:
            reason = "OK"
        else:
            failures: List[str] = []
            if groundedness < 0.3:
                failures.append(f"low groundedness ({groundedness:.4f} < 0.30)")
            if not is_relevant:
                failures.append("answer not relevant to query")
            if not is_safe:
                failures.append("refusal phrase detected")
            reason = "; ".join(failures)

        logger.info(
            "Guardrails — passed=%s | grounded=%.4f | relevant=%s | safe=%s | %s",
            passed, groundedness, is_relevant, is_safe, reason,
        )
        return GuardrailsResult(
            passed=passed,
            groundedness_score=groundedness,
            is_relevant=is_relevant,
            is_safe=is_safe,
            reason=reason,
        )

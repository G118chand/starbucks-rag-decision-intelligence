"""Unit tests for orchestration.guardrails.GuardrailsValidator."""

import pytest

from orchestration.guardrails import GuardrailsValidator
from orchestration.schemas import GuardrailsResult


@pytest.fixture
def validator() -> GuardrailsValidator:
    """Fresh GuardrailsValidator for each test."""
    return GuardrailsValidator()


def test_grounded_answer_scores_high(validator: GuardrailsValidator) -> None:
    """Answer that closely mirrors chunk vocabulary should score >= 0.25."""
    chunks = [
        {
            "text": (
                "Pacific Northwest sales declined 12 percent due to competitor opening "
                "and wait times increasing from 3 to 5 minutes"
            )
        }
    ]
    answer = (
        "Sales declined 12 percent in Pacific Northwest because competitor opened "
        "and wait times increased"
    )
    score = validator.check_groundedness(answer, chunks)
    assert score >= 0.25, f"Expected groundedness >= 0.25, got {score}"


def test_hallucinated_answer_scores_low(validator: GuardrailsValidator) -> None:
    """Answer with vocabulary absent from chunks should score < 0.25."""
    chunks = [{"text": "The sky is blue and weather is sunny today in Seattle"}]
    answer = (
        "Revenue increased 500 percent because of quantum computing blockchain "
        "synergies in metaverse"
    )
    score = validator.check_groundedness(answer, chunks)
    assert score < 0.25, f"Expected groundedness < 0.25, got {score}"


def test_validate_returns_correct_type(validator: GuardrailsValidator) -> None:
    """validate() must return a well-typed GuardrailsResult."""
    chunks = [{"text": "Sales declined 12 percent in Pacific Northwest region Q3 2024"}]
    answer = "Sales declined 12 percent in Pacific Northwest during Q3 2024 quarter review"
    query = "Why are Pacific Northwest sales declining?"

    result = validator.validate(answer, query, chunks)

    assert isinstance(result, GuardrailsResult)
    assert isinstance(result.passed, bool)
    assert isinstance(result.groundedness_score, float)
    assert 0.0 <= result.groundedness_score <= 1.0
    assert isinstance(result.reason, str)


def test_safety_catches_refusals(validator: GuardrailsValidator) -> None:
    """check_safety must block known LLM refusal phrases and pass clean answers."""
    assert validator.check_safety("I cannot provide that as an AI assistant") is False
    assert validator.check_safety(
        "Sales declined 12 percent in Q3 due to competition"
    ) is True


def test_relevance_short_answer_fails(validator: GuardrailsValidator) -> None:
    """Answers shorter than 50 characters must fail the relevance check."""
    assert validator.check_relevance("why are sales dropping", "No") is False

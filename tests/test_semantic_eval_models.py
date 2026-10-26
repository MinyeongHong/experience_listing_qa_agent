import pytest
from pydantic import ValidationError

from evals.models import SemanticEvalError, SemanticEvalSuccess, parse_eval_results


def test_success_requires_predicted_review():
    with pytest.raises(ValidationError):
        SemanticEvalSuccess(
            id="case_1",
            expected_review=True,
            verdict="possible_conflict",
            reason="Overlapping services",
        )


def test_parse_eval_results():
    raw_results = [
        {
            "id": "success_case",
            "status": "success",
            "expected_review": True,
            "predicted_review": True,
            "verdict": "possible_conflict",
            "reason": "Overlapping services",
        },
        {
            "id": "error_case",
            "status": "error",
            "expected_review": True,
            "error": "Gemini API failed",
        },
    ]

    results = parse_eval_results(raw_results)

    assert len(results) == 2
    assert isinstance(results[0], SemanticEvalSuccess)
    assert isinstance(results[1], SemanticEvalError)

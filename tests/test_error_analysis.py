from evals.error_analysis import find_mismatches
from evals.models import (
    SemanticEvalError,
    SemanticEvalSuccess,
)


def test_find_mismatches():
    results = [
        SemanticEvalSuccess(
            id="correct_case",
            expected_review=True,
            predicted_review=True,
            verdict="possible_conflict",
            reason="Overlapping services",
        ),
        SemanticEvalSuccess(
            id="false_positive",
            expected_review=False,
            predicted_review=True,
            verdict="possible_conflict",
            reason="Incorrectly flagged as conflict",
        ),
        SemanticEvalSuccess(
            id="false_negative",
            expected_review=True,
            predicted_review=False,
            verdict="no_conflict",
            reason="Missed potential conflict",
        ),
        SemanticEvalError(
            id="api_error",
            expected_review=True,
            error="Gemini API failed",
        ),
    ]

    mismatches = find_mismatches(results)

    assert len(mismatches) == 2
    assert mismatches[0].id == "false_positive"
    assert mismatches[1].id == "false_negative"

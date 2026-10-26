from evals.models import SemanticEvalError, SemanticEvalSuccess
from evals.run_semantic_eval import count_errors
from evals.semantic_metrics import calculate_metrics


def test_calculate_metrics():
    results = [
        SemanticEvalSuccess(
            id="case_1",
            expected_review=True,
            predicted_review=True,
            verdict="possible_conflict",
            reason="Overlapping services",
        ),
        SemanticEvalSuccess(
            id="case_1_uncertain",
            expected_review=True,
            predicted_review=True,
            verdict="uncertain",
            reason="Overlapping services",
        ),
        SemanticEvalSuccess(
            id="case_2",
            expected_review=True,
            predicted_review=False,
            verdict="possible_conflict",
            reason="Overlapping services",
        ),
        SemanticEvalSuccess(
            id="case_3",
            expected_review=False,
            predicted_review=True,
            verdict="possible_conflict",
            reason="Overlapping services",
        ),
        SemanticEvalSuccess(
            id="case_4",
            expected_review=False,
            predicted_review=False,
            verdict="no_conflict",
            reason="Overlapping services",
        ),
    ]

    metrics = calculate_metrics(results)

    assert metrics["tp"] == 2
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tn"] == 1
    assert metrics["precision"] == 2 / 3
    assert metrics["recall"] == 2 / 3
    assert metrics["f1"] == 2 / 3


def test_calculate_metrics_skips_errors():
    results = [
        SemanticEvalSuccess(
            id="success_case",
            expected_review=True,
            predicted_review=True,
            verdict="possible_conflict",
            reason="Overlapping services",
        ),
        SemanticEvalError(
            id="error_case",
            expected_review=True,
            error="Gemini API failed",
        ),
    ]

    metrics = calculate_metrics(results)

    assert metrics["tp"] == 1
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert metrics["tn"] == 0


def test_count_errors():
    results = [
        SemanticEvalSuccess(
            id="success_case",
            expected_review=True,
            predicted_review=True,
            verdict="possible_conflict",
            reason="Overlapping services",
        ),
        SemanticEvalError(
            id="error_case",
            expected_review=True,
            error="Gemini API failed",
        ),
        SemanticEvalError(
            id="error_case",
            expected_review=True,
            error="Gemini API failed",
        ),
    ]

    assert count_errors(results) == 2

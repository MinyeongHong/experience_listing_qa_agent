from evals.error_analysis import (
    count_false_positives,
    find_mismatches,
    print_error_analysis,
)
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


def test_count_false_positives():
    results = [
        SemanticEvalSuccess(
            id="case_a",
            expected_review=False,
            predicted_review=True,
            verdict="possible_conflict",
            reason="Incorrectly flagged",
        ),
        SemanticEvalSuccess(
            id="case_b",
            expected_review=True,
            predicted_review=False,
            verdict="no_conflict",
            reason="Missed conflict",
        ),
        SemanticEvalError(
            id="case_error",
            expected_review=True,
            error="Gemini API failed",
        ),
    ]

    result = count_false_positives(results)

    assert result == 1


def test_print_error_analysis(capsys):
    results = [
        SemanticEvalSuccess(
            id="case_fp",
            expected_review=False,
            predicted_review=True,
            verdict="possible_conflict",
            reason="Incorrectly flagged",
        ),
        SemanticEvalSuccess(
            id="case_fn",
            expected_review=True,
            predicted_review=False,
            verdict="no_conflict",
            reason="Missed conflict",
        ),
    ]

    print_error_analysis(results)

    output = capsys.readouterr().out

    assert "case_fp: false_positive" in output
    assert "case_fn: false_negative" in output

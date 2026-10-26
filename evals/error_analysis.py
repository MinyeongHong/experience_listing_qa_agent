from evals.models import (
    SemanticEvalError,
    SemanticEvalResult,
    SemanticEvalSuccess,
)


def find_mismatches(
    results: list[SemanticEvalResult],
) -> list[SemanticEvalSuccess]:
    mismatches = []

    for result in results:
        if isinstance(result, SemanticEvalError):
            continue

        if result.expected_review != result.predicted_review:
            mismatches.append(result)

    return mismatches


def classify_mismatch(result: SemanticEvalSuccess) -> str:
    expected = result.expected_review
    predicted = result.predicted_review

    if not expected and predicted:
        return "false_positive"
    elif expected and not predicted:
        return "false_negative"
    else:
        return "correct"

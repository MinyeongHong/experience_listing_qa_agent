from typing import Literal

from evals.models import (
    SemanticEvalError,
    SemanticEvalResult,
    SemanticEvalSuccess,
    SemanticMismatch,
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


def classify_mismatch(
    result: SemanticEvalSuccess,
) -> Literal["false_positive", "false_negative", "correct"]:
    expected = result.expected_review
    predicted = result.predicted_review

    if not expected and predicted:
        return "false_positive"
    elif expected and not predicted:
        return "false_negative"
    else:
        return "correct"


def count_false_positives(
    results: list[SemanticEvalResult],
) -> int:
    count = 0

    for result in results:
        if isinstance(result, SemanticEvalError):
            continue

        if not result.expected_review and result.predicted_review:
            count += 1

    return count


def print_error_analysis(
    results: list[SemanticEvalResult],
) -> None:
    print("=== Error Analysis ===")

    mismatch_results: list[SemanticEvalSuccess] = find_mismatches(results)

    if not mismatch_results:
        print("No mismatches found.")

        return

    for mismatch_result in mismatch_results:
        category = classify_mismatch(mismatch_result)

        print(f"{mismatch_result.id}: {category}")


def get_error_analysis(
    results: list[SemanticEvalResult],
) -> list[SemanticMismatch]:
    analysis: list[SemanticMismatch] = []

    for mismatch_result in find_mismatches(results):
        category = classify_mismatch(mismatch_result)
        if category == "correct":
            continue

        analysis.append(
            SemanticMismatch(
                id=mismatch_result.id,
                category=category,
            )
        )

    return analysis

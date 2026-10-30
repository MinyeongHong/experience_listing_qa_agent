from evals.error_analysis import get_error_analysis
from evals.models import (
    EvalRunMetadata,
    SemanticEvalError,
    SemanticEvalReport,
    SemanticEvalResult,
    SemanticEvalSuccess,
    SemanticMetrics,
    SemanticMismatch,
)
from evals.semantic_metrics import calculate_metrics


def build_eval_report(
    results: list[SemanticEvalResult],
    *,
    total_cases: int | None = None,
    skipped_cases: int | None = None,
    metadata: EvalRunMetadata | None = None,
) -> SemanticEvalReport:

    metrics: SemanticMetrics = calculate_metrics(results)

    mismatches: list[SemanticMismatch] = get_error_analysis(results)

    evaluated_cases = sum(
        1 for result in results if isinstance(result, SemanticEvalSuccess)
    )

    provider_errors = sum(
        1 for result in results if isinstance(result, SemanticEvalError)
    )

    attempted_cases = len(results)

    if (
        total_cases is not None
        and skipped_cases is not None
        and total_cases != attempted_cases + skipped_cases
    ):
        raise ValueError(
            "total_cases must equal attempted_cases + skipped_cases "
            f"(got total_cases={total_cases}, attempted_cases={attempted_cases}, "
            f"skipped_cases={skipped_cases})"
        )

    if total_cases is None:
        evaluation_coverage = None
    elif total_cases == 0:
        evaluation_coverage = 0.0
    else:
        evaluation_coverage = evaluated_cases / total_cases

    provider_error_rate = (
        0.0 if attempted_cases == 0 else provider_errors / attempted_cases
    )

    report = SemanticEvalReport(
        total_cases=total_cases,
        attempted_cases=attempted_cases,
        evaluated_cases=evaluated_cases,
        skipped_cases=skipped_cases,
        provider_errors=provider_errors,
        evaluation_coverage=evaluation_coverage,
        provider_error_rate=provider_error_rate,
        metadata=metadata,
        metrics=metrics,
        mismatches=mismatches,
    )

    return report

from evals.error_analysis import get_error_analysis
from evals.models import SemanticEvalResult
from evals.semantic_metrics import calculate_metrics


def build_eval_report(
    results: list[SemanticEvalResult],
) -> dict:
    metrics: dict = calculate_metrics(results)

    mismatches: list[dict] = get_error_analysis(results)

    report = {
        "metrics": metrics,
        "mismatches": mismatches,
    }

    return report

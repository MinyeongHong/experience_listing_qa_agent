"""Compare saved semantic reports without model or database access."""

import argparse
from decimal import Decimal
from math import isclose, isfinite
from pathlib import Path

from pydantic import ValidationError

from evals.models import (
    SemanticEvalReport,
    SemanticMetricDeltas,
    SemanticRegressionComparison,
)


def _comparison_issues(report: SemanticEvalReport, label: str) -> list[str]:
    issues = []
    if report.metadata is None or not report.metadata.dataset_version.strip():
        issues.append(f"{label}: dataset version is missing.")
    if report.provider_errors != 0 or report.provider_error_rate != 0:
        issues.append(f"{label}: provider errors prevent comparison.")

    total = report.total_cases
    skipped = report.skipped_cases
    evaluated = report.evaluated_cases
    if (
        total is None
        or total <= 0
        or skipped is None
        or skipped < 0
        or evaluated <= 0
        or report.attempted_cases != evaluated
        or evaluated + skipped != total
    ):
        issues.append(f"{label}: evaluation is incomplete or counts are inconsistent.")

    coverage = report.evaluation_coverage
    if (
        coverage is None
        or not isfinite(coverage)
        or not 0 <= coverage <= 1
        or (
            total is not None and total > 0 and not isclose(coverage, evaluated / total)
        )
    ):
        issues.append(f"{label}: coverage is missing or inconsistent.")

    metrics = report.metrics
    counts = (metrics.tp, metrics.fp, metrics.fn, metrics.tn)
    if any(count < 0 for count in counts) or sum(counts) != evaluated:
        issues.append(f"{label}: confusion matrix does not match evaluated cases.")
    if any(
        not isfinite(value) or not 0 <= value <= 1
        for value in (metrics.precision, metrics.recall, metrics.f1)
    ):
        issues.append(f"{label}: metrics must be finite values between 0 and 1.")
    return issues


def compare_reports(
    baseline: SemanticEvalReport,
    candidate: SemanticEvalReport,
    max_f1_drop: float = 0.02,
) -> SemanticRegressionComparison:
    comparison = SemanticRegressionComparison(
        status="not_comparable", max_f1_drop=max_f1_drop
    )
    reasons = _comparison_issues(baseline, "baseline")
    reasons.extend(_comparison_issues(candidate, "candidate"))
    if (
        baseline.metadata is not None
        and candidate.metadata is not None
        and baseline.metadata.dataset_version != candidate.metadata.dataset_version
    ):
        reasons.append("Dataset versions differ.")
    if baseline.total_cases != candidate.total_cases:
        reasons.append("Dataset sizes differ; evaluate both runs on the same dataset.")
    if baseline.evaluated_cases != candidate.evaluated_cases:
        reasons.append("Evaluated case counts differ.")
    if baseline.skipped_cases != candidate.skipped_cases:
        reasons.append("Skipped case counts differ.")
    if baseline.evaluation_coverage != candidate.evaluation_coverage:
        reasons.append("Evaluation coverage differs.")
    if reasons:
        comparison.reasons = reasons
        return comparison

    def delta(before: float, after: float) -> Decimal:
        return Decimal(str(after)) - Decimal(str(before))

    f1_delta = delta(baseline.metrics.f1, candidate.metrics.f1)
    comparison.deltas = SemanticMetricDeltas(
        precision=float(delta(baseline.metrics.precision, candidate.metrics.precision)),
        recall=float(delta(baseline.metrics.recall, candidate.metrics.recall)),
        f1=float(f1_delta),
        coverage=float(
            delta(baseline.evaluation_coverage, candidate.evaluation_coverage)
        ),
    )
    comparison.status = (
        "regression" if f1_delta < -Decimal(str(max_f1_drop)) else "passed"
    )
    return comparison


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="Baseline semantic report JSON")
    parser.add_argument("candidate", type=Path, help="Candidate semantic report JSON")
    parser.add_argument("--max-f1-drop", type=float, default=0.02)
    args = parser.parse_args()
    try:
        baseline = SemanticEvalReport.model_validate_json(
            args.baseline.read_text(encoding="utf-8")
        )
        candidate = SemanticEvalReport.model_validate_json(
            args.candidate.read_text(encoding="utf-8")
        )
        comparison = compare_reports(baseline, candidate, args.max_f1_drop)
    except (OSError, ValueError, ValidationError) as exc:
        parser.error(str(exc))

    print(comparison.model_dump_json(indent=2))
    return {"passed": 0, "regression": 1, "not_comparable": 2}[comparison.status]


if __name__ == "__main__":
    raise SystemExit(main())

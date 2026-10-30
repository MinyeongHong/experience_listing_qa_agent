from datetime import UTC, datetime

import pytest

from evals.models import EvalRunMetadata, SemanticEvalError, SemanticEvalSuccess
from evals.report import build_eval_report


def _success(case_id: str = "ok") -> SemanticEvalSuccess:
    return SemanticEvalSuccess(
        id=case_id,
        expected_review=True,
        predicted_review=True,
        verdict="possible_conflict",
        reason="match",
    )


def _error(case_id: str = "err") -> SemanticEvalError:
    return SemanticEvalError(
        id=case_id,
        expected_review=True,
        error="provider failed",
    )


def test_build_eval_report_success_cases():
    results = [_success("a"), _success("b")]

    report = build_eval_report(results)

    assert report.evaluated_cases == 2
    assert report.provider_errors == 0
    assert report.attempted_cases == 2
    assert report.provider_error_rate == 0.0


def test_build_eval_report_provider_error():
    results = [_success("a"), _error("b")]

    report = build_eval_report(results)

    assert report.attempted_cases == 2
    assert report.evaluated_cases == 1
    assert report.provider_errors == 1
    assert report.provider_error_rate == 0.5


def test_build_eval_report_skipped_coverage():
    results = [_success("a"), _success("b"), _success("c")]

    report = build_eval_report(
        results,
        total_cases=5,
        skipped_cases=2,
    )

    assert report.evaluation_coverage == 0.6
    assert report.skipped_cases == 2
    assert report.total_cases == 5


def test_build_eval_report_all_provider_errors():
    results = [_error("a"), _error("b")]

    report = build_eval_report(results)

    assert report.evaluated_cases == 0
    assert report.provider_errors == 2
    assert report.provider_error_rate == 1.0


def test_build_eval_report_empty_data():
    report = build_eval_report(
        [],
        total_cases=0,
        skipped_cases=0,
    )

    assert report.evaluation_coverage == 0.0
    assert report.provider_error_rate == 0.0
    assert report.attempted_cases == 0
    assert report.evaluated_cases == 0


def test_build_eval_report_aggregate_mismatch():
    results = [_success("a"), _success("b"), _success("c")]

    with pytest.raises(ValueError):
        build_eval_report(
            results,
            total_cases=10,
            skipped_cases=2,
        )


def test_build_eval_report_missing_totals():
    results = [_success("a")]

    report = build_eval_report(results)

    assert report.evaluation_coverage is None
    assert report.total_cases is None
    assert report.skipped_cases is None


def test_build_eval_report_metadata():
    metadata = EvalRunMetadata(
        model="gemini-2.5-flash",
        prompt_version="v1",
        dataset_version="v1",
        mode="live",
        timestamp=datetime.now(UTC),
    )

    report = build_eval_report([_success("a")], metadata=metadata)

    assert report.metadata == metadata

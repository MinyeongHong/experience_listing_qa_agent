import json
import subprocess
import sys
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from evals.compare_semantic_reports import compare_reports
from evals.models import EvalRunMetadata, SemanticEvalReport, SemanticMetrics


def report(tp=40, skipped=20, version="dataset-v2"):
    evaluated = 100 - skipped
    return SemanticEvalReport(
        total_cases=100,
        attempted_cases=evaluated,
        evaluated_cases=evaluated,
        skipped_cases=skipped,
        provider_errors=0,
        evaluation_coverage=evaluated / 100,
        provider_error_rate=0,
        metadata=EvalRunMetadata(
            model="fake-model",
            prompt_version="v1",
            dataset_version=version,
            mode="live",
            timestamp=datetime.now(UTC),
        ),
        metrics=SemanticMetrics(
            tp=tp,
            fp=50 - tp,
            fn=50 - tp,
            tn=evaluated - 100 + tp,
            precision=tp / 50,
            recall=tp / 50,
            f1=tp / 50,
        ),
        mismatches=[],
    )


@pytest.mark.parametrize(
    ("tp", "threshold", "status"),
    [
        (41, 0.02, "passed"),
        (40, 0.02, "passed"),
        (39, 0.02, "passed"),
        (38, 0.02, "regression"),
        (38, 0.04, "passed"),
        (39, 0, "regression"),
    ],
)
def test_f1_regression_boundary(tp, threshold, status):
    comparison = compare_reports(report(), report(tp=tp), threshold)
    assert comparison.status == status
    assert comparison.deltas.f1 == pytest.approx((tp - 40) / 50)
    assert comparison.reasons == []


def test_metric_deltas_are_candidate_minus_baseline():
    comparison = compare_reports(report(), report(tp=41))
    assert comparison.status == "passed"
    assert comparison.deltas.model_dump() == pytest.approx(
        {"precision": 0.02, "recall": 0.02, "f1": 0.02, "coverage": 0}
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_different_complete_evaluation_counts_are_not_comparable(reverse):
    baseline, candidate = report(), report(tp=41, skipped=10)
    if reverse:
        baseline, candidate = candidate, baseline
    comparison = compare_reports(baseline, candidate)
    assert comparison.status == "not_comparable"
    assert comparison.deltas is None
    assert "Evaluated case counts differ." in comparison.reasons
    assert "Skipped case counts differ." in comparison.reasons
    assert "Evaluation coverage differs." in comparison.reasons


def test_different_coverage_is_rejected_even_with_matching_counts():
    candidate = report().model_copy(update={"evaluation_coverage": 0.8000000001})
    comparison = compare_reports(report(), candidate)
    assert comparison.status == "not_comparable"
    assert comparison.deltas is None
    assert comparison.reasons == ["Evaluation coverage differs."]


def test_precision_and_recall_can_move_in_opposite_directions():
    candidate = report().model_copy(
        update={
            "metrics": SemanticMetrics(
                tp=36, fp=4, fn=14, tn=26, precision=0.9, recall=0.72, f1=0.8
            )
        }
    )
    comparison = compare_reports(report(), candidate)
    assert comparison.status == "passed"
    assert comparison.deltas.model_dump() == pytest.approx(
        {"precision": 0.1, "recall": -0.08, "f1": 0, "coverage": 0}
    )


@pytest.mark.parametrize("side", ["baseline", "candidate"])
@pytest.mark.parametrize(
    "updates",
    [
        {"metadata": None},
        {"metadata": report(version="  ").metadata},
        {"total_cases": None, "skipped_cases": None, "evaluation_coverage": None},
        {"evaluated_cases": 79},
        {"attempted_cases": 79},
        {"skipped_cases": 19},
        {"provider_errors": 1},
        {"provider_error_rate": 0.01},
        {"evaluation_coverage": 0.9},
        {"evaluation_coverage": float("nan")},
        {"metrics": report().metrics.model_copy(update={"f1": float("nan")})},
        {"metrics": report().metrics.model_copy(update={"tp": -1})},
        {"metrics": report().metrics.model_copy(update={"tn": 21})},
        {
            "attempted_cases": 0,
            "evaluated_cases": 0,
            "skipped_cases": 100,
            "evaluation_coverage": 0,
        },
        {
            "total_cases": 0,
            "evaluated_cases": 0,
            "attempted_cases": 0,
            "skipped_cases": 0,
        },
    ],
)
def test_unusable_reports_are_not_comparable(side, updates):
    reports = {"baseline": report(), "candidate": report()}
    reports[side] = reports[side].model_copy(update=updates)
    comparison = compare_reports(**reports)
    assert comparison.status == "not_comparable"
    assert comparison.deltas is None
    assert any(side in reason for reason in comparison.reasons)


def test_dataset_version_mismatch():
    comparison = compare_reports(
        report(version="dataset-v1"), report(version="dataset-v2")
    )
    assert comparison.status == "not_comparable"
    assert "Dataset versions differ." in comparison.reasons


def test_dataset_size_mismatch_even_when_version_matches():
    # Synthetic reports only: do not compare the saved 8-case and 14-case results.
    candidate = report().model_copy(
        update={
            "total_cases": 101,
            "skipped_cases": 21,
            "evaluation_coverage": 80 / 101,
        }
    )
    comparison = compare_reports(report(), candidate)
    assert comparison.status == "not_comparable"
    assert any("sizes differ" in reason for reason in comparison.reasons)


@pytest.mark.parametrize("threshold", [-0.01, 1.01, float("nan"), float("inf")])
def test_invalid_threshold(threshold):
    with pytest.raises(ValidationError):
        compare_reports(report(), report(), threshold)


@pytest.mark.parametrize(
    ("candidate", "threshold", "exit_code", "status"),
    [
        (report(tp=41), "0.02", 0, "passed"),
        (report(tp=38), "0.02", 1, "regression"),
        (report(tp=38), "0.04", 0, "passed"),
        (report(version="another-version"), "0.02", 2, "not_comparable"),
        (report(skipped=10), "0.02", 2, "not_comparable"),
    ],
)
def test_cli(tmp_path, candidate, threshold, exit_code, status):
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(report().model_dump_json(), encoding="utf-8")
    candidate_path.write_text(candidate.model_dump_json(), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "evals.compare_semantic_reports",
            str(baseline_path),
            str(candidate_path),
            "--max-f1-drop",
            threshold,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == exit_code, result.stderr
    assert json.loads(result.stdout)["status"] == status


@pytest.mark.parametrize("content", ["not json", "{}", None])
def test_cli_rejects_invalid_or_missing_report(tmp_path, content):
    path = tmp_path / "report.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "evals.compare_semantic_reports", str(path), str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "error:" in result.stderr

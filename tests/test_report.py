from evals.models import SemanticEvalSuccess
from evals.report import build_eval_report


def test_build_eval_report():
    results = [
        SemanticEvalSuccess(
            id="case_fp",
            expected_review=False,
            predicted_review=True,
            verdict="possible_conflict",
            reason="Incorrectly flagged",
        )
    ]

    report = build_eval_report(results)

    assert report["metrics"]["fp"] == 1
    assert len(report["mismatches"]) == 1
    assert report["mismatches"][0]["category"] == "false_positive"

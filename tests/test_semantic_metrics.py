from evals.run_semantic_eval import count_errors
from evals.semantic_metrics import calculate_metrics


def test_calculate_metrics():
    results = [
        {"expected_review": True, "predicted_review": True},
        {"expected_review": True, "predicted_review": True},
        {"expected_review": False, "predicted_review": True},
        {"expected_review": True, "predicted_review": False},
        {"expected_review": False, "predicted_review": False},
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
        {
            "id": "success_case",
            "status": "success",
            "expected_review": True,
            "predicted_review": True,
        },
        {
            "id": "error_case",
            "status": "error",
            "expected_review": True,
            "error": "Gemini API failed",
        },
    ]

    metrics = calculate_metrics(results)

    assert metrics["tp"] == 1
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert metrics["tn"] == 0


def test_count_errors():
    results = [
        {"status": "success"},
        {"status": "error"},
        {"status": "error"},
        {"id": "legacy_case"},
    ]

    assert count_errors(results) == 2

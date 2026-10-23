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

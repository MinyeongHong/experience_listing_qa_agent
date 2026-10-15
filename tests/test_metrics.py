from evals.metrics import aggregate_metrics, evaluate_case


def test_perfect_prediction():
    metrics = evaluate_case(
        expected_fields={"city": "Barcelona", "duration_minutes": 120},
        actual_fields={"city": "Barcelona", "duration_minutes": 120},
        expected_issue_codes=["MISSING_MEETING_POINT"],
        actual_issue_codes=["MISSING_MEETING_POINT"],
    )

    assert metrics.field_accuracy == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0


def test_partial_prediction():
    metrics = evaluate_case(
        expected_fields={"city": "Barcelona", "duration_minutes": 120},
        actual_fields={"city": "Madrid", "duration_minutes": 120},
        expected_issue_codes=[
            "MISSING_MEETING_POINT",
            "CONTRADICTORY_INCLUSION",
        ],
        actual_issue_codes=[
            "MISSING_MEETING_POINT",
            "ZERO_PRICE",
        ],
    )

    assert metrics.field_accuracy == 0.5
    assert metrics.true_positives == 1
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.f1 == 0.5


def test_no_issues_on_either_side():
    metrics = evaluate_case(
        expected_fields={},
        actual_fields={},
        expected_issue_codes=[],
        actual_issue_codes=[],
    )

    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0


def test_missed_issue():
    metrics = evaluate_case(
        expected_fields={},
        actual_fields={},
        expected_issue_codes=["MISSING_MEETING_POINT"],
        actual_issue_codes=[],
    )

    assert metrics.true_positives == 0
    assert metrics.false_negatives == 1
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0


def test_false_positive():
    metrics = evaluate_case(
        expected_fields={},
        actual_fields={},
        expected_issue_codes=[],
        actual_issue_codes=["ZERO_PRICE"],
    )

    assert metrics.false_positives == 1
    assert metrics.precision == 0.0
    assert metrics.f1 == 0.0


def test_aggregate_metrics():
    first = evaluate_case(
        expected_fields={"city": "Barcelona"},
        actual_fields={"city": "Barcelona"},
        expected_issue_codes=["MISSING_MEETING_POINT"],
        actual_issue_codes=["MISSING_MEETING_POINT"],
    )

    second = evaluate_case(
        expected_fields={"city": "Barcelona"},
        actual_fields={"city": "Madrid"},
        expected_issue_codes=["CONTRADICTORY_INCLUSION"],
        actual_issue_codes=["ZERO_PRICE"],
    )

    summary = aggregate_metrics([first, second])

    assert summary["field_accuracy"] == 0.5
    assert summary["tp"] == 1
    assert summary["fp"] == 1
    assert summary["fn"] == 1
    assert summary["precision"] == 0.5
    assert summary["recall"] == 0.5
    assert summary["f1"] == 0.5

import pytest
from pydantic import ValidationError

from evals.models import SemanticEvalCase, parse_eval_cases


def test_parse_eval_cases():
    raw_cases = [
        {
            "id": "case_01",
            "included": "Lunch",
            "excluded": "Meals",
            "expected_review": True,
        }
    ]

    cases = parse_eval_cases(raw_cases)

    assert len(cases) == 1
    assert isinstance(cases[0], SemanticEvalCase)
    assert cases[0].included == "Lunch"


def test_parse_eval_cases_missing_required_field():
    raw_cases = [
        {
            "id": "case_02",
            "included": "Lunch",
            "expected_review": True,
        }
    ]

    with pytest.raises(ValidationError):
        parse_eval_cases(raw_cases)


def test_parse_eval_cases_rejects_string_boolean():
    raw_cases = [
        {
            "id": "case_03",
            "included": "Lunch",
            "excluded": "Meals",
            "expected_review": "false",
        }
    ]

    with pytest.raises(ValidationError):
        parse_eval_cases(raw_cases)

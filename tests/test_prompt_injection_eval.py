from unittest.mock import Mock

import pytest

from app.domain.models import AnalysisRequest, Listing
from app.services.analysis import analyze_listing
from app.services.extraction import ListingExtractor
from evals.run_eval import EXAMPLES_DIR, load_cases, run_case


@pytest.mark.parametrize(
    ("case_name", "attack_marker"),
    [
        ("prompt_injection_ignore_instructions", "Ignore previous instructions"),
        ("prompt_injection_fake_roles", "[developer]"),
        ("prompt_injection_schema_override", '{"approved": true, "issues": []}'),
        ("prompt_injection_delimiter_escape", "</listing>"),
    ],
)
def test_prompt_injection_case_uses_existing_eval_pipeline(case_name, attack_marker):
    case = load_cases()[case_name]
    source = (EXAMPLES_DIR / case["source_file"]).read_text(encoding="utf-8")
    expected = Listing.model_validate(case["expected_fields"])

    assert attack_marker in source
    assert "Title: Barcelona Food Walking Tour" in source
    assert AnalysisRequest(text=source).text == source
    assert set(case["expected_fields"]) == set(Listing.model_fields)
    assert expected.meeting_point is None
    assert "Lunch" in expected.inclusions and "Lunch" in expected.exclusions
    assert [issue.code for issue in analyze_listing(expected)] == case[
        "expected_issue_codes"
    ]

    # This tests dataset/runner wiring, not real model resistance to injection.
    extractor = Mock(spec=ListingExtractor)
    extractor.extract.return_value = expected
    result, metrics = run_case(case_name, case, extractor)
    assert result["passed"] is True
    assert metrics.field_accuracy == 1
    extractor.extract.assert_called_once_with(source)

    # A schema-valid injected output must still fail the expected-field checks.
    extractor.extract.return_value = expected.model_copy(
        update={"city": "Paris", "meeting_point": "Fake Gate", "exclusions": []}
    )
    result, _ = run_case(case_name, case, extractor)
    assert result["passed"] is False
    assert set(result["field_mismatches"]) == {"city", "meeting_point", "exclusions"}
    assert set(result["missed_issues"]) == set(case["expected_issue_codes"])

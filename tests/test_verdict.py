import pytest

from app.domain.models import Issue, ReviewCandidate
from app.services import analysis


def make_issue(code):
    return Issue(code=code, field="test", message=f"Details for {code}.")


def make_candidate(code="POSSIBLE_INCLUSION_CONFLICT", status="pending"):
    return ReviewCandidate(
        code=code,
        field="inclusions",
        included_item="Hotel pickup",
        excluded_item="Transportation",
        reason="Manual review needed for this pair.",
        status=status,
    )


def test_no_issues_or_candidates_passes():
    verdict, review_required, reason = analysis.determine_verdict([], [])
    assert verdict == "pass"
    assert review_required is False
    assert reason.strip()


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("CONTRADICTORY_INCLUSION", "fail"),
        ("DURATION_CONTRADICTION", "fail"),
        ("MISSING_CITY", "review"),
        ("MISSING_DURATION", "review"),
        ("MISSING_MEETING_POINT", "review"),
        ("ZERO_PRICE", "review"),
    ],
)
def test_deterministic_issue_policy(code, expected):
    issue = make_issue(code)
    verdict, review_required, reason = analysis.determine_verdict([issue], [])
    assert verdict == expected
    assert review_required is True
    assert issue.code in reason
    assert issue.message in reason


@pytest.mark.parametrize(
    "code",
    [
        "POSSIBLE_INCLUSION_CONFLICT",
        "SEMANTIC_CLASSIFICATION_FAILED",
        "SEMANTIC_BUDGET_EXHAUSTED",
    ],
)
def test_pending_semantic_candidate_requires_review(code):
    candidate = make_candidate(code=code)
    verdict, review_required, reason = analysis.determine_verdict([], [candidate])
    assert verdict == "review"
    assert review_required is True
    assert candidate.code in reason
    assert candidate.reason in reason
    assert candidate.included_item in reason
    assert candidate.excluded_item in reason


@pytest.mark.parametrize("status", ["confirmed", "dismissed"])
def test_nonpending_candidate_alone_does_not_require_review(status):
    verdict, review_required, _ = analysis.determine_verdict(
        [], [make_candidate(status=status)]
    )
    assert verdict == "pass"
    assert review_required is False


@pytest.mark.parametrize("reverse", [False, True])
def test_fail_takes_precedence_and_reason_preserves_all_active_evidence(reverse):
    missing = make_issue("MISSING_CITY")
    conflict = make_issue("DURATION_CONTRADICTION")
    issues = [missing, conflict] if not reverse else [conflict, missing]
    pending = make_candidate(code="SEMANTIC_BUDGET_EXHAUSTED")
    dismissed = make_candidate(status="dismissed")
    verdict, review_required, reason = analysis.determine_verdict(
        issues, [dismissed, pending]
    )
    assert verdict == "fail"
    assert review_required is True
    assert conflict.message in reason and missing.message in reason
    assert reason.index(conflict.code) < reason.index(missing.code)
    assert pending.code in reason
    assert dismissed.code not in reason


def test_verdict_is_repeatable_and_does_not_mutate_inputs():
    issues = [make_issue("ZERO_PRICE"), make_issue("CONTRADICTORY_INCLUSION")]
    candidates = [make_candidate()]
    original_issues = [issue.model_dump() for issue in issues]
    original_candidates = [candidate.model_dump() for candidate in candidates]

    first = analysis.determine_verdict(issues, candidates)
    assert analysis.determine_verdict(issues, candidates) == first
    assert [issue.model_dump() for issue in issues] == original_issues
    assert [candidate.model_dump() for candidate in candidates] == original_candidates

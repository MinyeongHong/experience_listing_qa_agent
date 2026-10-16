import pytest

from app.domain.models import Listing, SemanticDecision
from app.services.review import (
    SemanticClassificationError,
    build_comparison_pairs,
    find_review_candidates,
)


class FakeSemanticClassifier:
    def __init__(
        self,
        decisions: dict[tuple[str, str], SemanticDecision],
    ):
        self.decision = decisions

    def classify(
        self,
        included_item: str,
        excluded_item: str,
    ) -> SemanticDecision:
        return self.decision.get(
            (included_item, excluded_item),
            SemanticDecision(
                verdict="no_conflict",
                reason="No conflict identified in this test.",
            ),
        )


class FailingSemanticClassifier:
    def classify(
        self,
        _included_item: str,
        _excluded_item: str,
    ) -> SemanticDecision:
        raise SemanticClassificationError("Model request timed out")


class PartiallyFailingClassifier:
    def classify(
        self,
        included_item: str,
        _excluded_item: str,
    ) -> SemanticDecision:
        if included_item == "Food tasting":
            raise SemanticClassificationError("Timeout")

        return SemanticDecision(
            verdict="possible_conflict",
            reason="Potential overlap.",
        )


def make_listing(
    inclusions: list[str],
    exclusions: list[str],
) -> Listing:
    return Listing(
        title="Barcelona Walking Tour",
        city="Barcelona",
        duration_minutes=120,
        meeting_point="Plaça de Catalunya",
        inclusions=inclusions,
        exclusions=exclusions,
    )


@pytest.mark.parametrize(
    ("included", "excluded", "expected_count"),
    [
        ("Lunch", "Lunch", 0),
        ("Food tasting", "Food tastings", 1),
        ("Food tasting", "Full meal", 0),
        ("Hotel pickup", "Transportation", 1),
    ],
)
def test_review_candidate_classification(
    included: str,
    excluded: str,
    expected_count: int,
):
    listing = make_listing(
        inclusions=[included],
        exclusions=[excluded],
    )

    classifier = FakeSemanticClassifier(
        decisions={
            ("Food tasting", "Food tastings"): SemanticDecision(
                verdict="possible_conflict",
                reason="Both expressions refer to food tasting.",
            ),
            ("Hotel pickup", "Transportation"): SemanticDecision(
                verdict="uncertain",
                reason="The transportation scope is unclear.",
            ),
        }
    )

    candidates = find_review_candidates(
        listing=listing,
        classifier=classifier,
    )

    assert len(candidates) == expected_count

    if candidates:
        candidate = candidates[0]

        assert candidate.code == "POSSIBLE_INCLUSION_CONFLICT"
        assert candidate.included_item == included
        assert candidate.excluded_item == excluded
        assert candidate.status == "pending"


def test_build_comparison_pairs():
    listing = make_listing(
        inclusions=["Lunch", "Food tasting", "Hotel pickup"],
        exclusions=["Lunch", "Food tastings", "Transportation"],
    )

    pairs = build_comparison_pairs(listing)

    assert pairs == [
        ("Lunch", "Food tastings"),
        ("Lunch", "Transportation"),
        ("Food tasting", "Lunch"),
        ("Food tasting", "Food tastings"),
        ("Food tasting", "Transportation"),
        ("Hotel pickup", "Lunch"),
        ("Hotel pickup", "Food tastings"),
        ("Hotel pickup", "Transportation"),
    ]


def test_uncertain_decision_creates_review_candidate():
    listing = make_listing(
        inclusions=["Hotel pickup"],
        exclusions=["Transportation"],
    )

    classifier = FakeSemanticClassifier(
        decisions={
            ("Hotel pickup", "Transportation"): SemanticDecision(
                verdict="uncertain",
                reason="Transportation scope is not specified.",
            )
        }
    )

    candidates = find_review_candidates(
        listing=listing,
        classifier=classifier,
    )

    assert len(candidates) == 1
    assert candidates[0].status == "pending"
    assert candidates[0].reason == "Transportation scope is not specified."


def test_classifier_failure_creates_review_candidate():
    listing = make_listing(
        inclusions=["Food tasting"],
        exclusions=["Food tastings"],
    )

    candidates = find_review_candidates(
        listing=listing,
        classifier=FailingSemanticClassifier(),
    )

    assert len(candidates) == 1

    candidate = candidates[0]

    assert candidate.code == "SEMANTIC_CLASSIFICATION_FAILED"
    assert candidate.status == "pending"
    assert candidate.included_item == "Food tasting"
    assert candidate.excluded_item == "Food tastings"
    assert "manual review" in candidate.reason


def test_classifier_failure_does_not_stop_other_pairs():
    listing = make_listing(
        inclusions=["Food tasting", "Hotel pickup"],
        exclusions=["Transportation"],
    )

    candidates = find_review_candidates(
        listing=listing,
        classifier=PartiallyFailingClassifier(),
    )

    assert len(candidates) == 2

    assert {candidate.code for candidate in candidates} == {
        "SEMANTIC_CLASSIFICATION_FAILED",
        "POSSIBLE_INCLUSION_CONFLICT",
    }

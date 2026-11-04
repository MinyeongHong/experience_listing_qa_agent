from unittest.mock import Mock, call

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


def test_build_comparison_pairs_deduplicates_normalized_pairs_in_first_seen_order():
    listing = make_listing(
        inclusions=[" Straße ", "STRASSE", "Lunch", " lunch "],
        exclusions=[" Drinks ", "DRINKS", "strasse", "Lunch"],
    )

    assert build_comparison_pairs(listing) == [
        (" Straße ", " Drinks "),
        (" Straße ", "Lunch"),
        ("Lunch", " Drinks "),
        ("Lunch", "strasse"),
    ]


def test_build_comparison_pairs_preserves_direction_when_deduplicating():
    listing = make_listing(
        inclusions=["Lunch", "Drinks", " lunch "],
        exclusions=["Drinks", "Lunch", " drinks "],
    )

    assert build_comparison_pairs(listing) == [
        ("Lunch", "Drinks"),
        ("Drinks", "Lunch"),
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


@pytest.mark.parametrize("budget", [None, 3, 4])
def test_budget_allows_all_pairs_when_unlimited_or_sufficient(budget):
    listing = make_listing(["Lunch", "Guide", "Pickup"], ["Meals"])
    classifier = Mock()
    classifier.classify.side_effect = [
        SemanticDecision(verdict="possible_conflict", reason="Overlapping services"),
        SemanticDecision(verdict="no_conflict", reason="Different services"),
        SemanticDecision(verdict="uncertain", reason="Unclear scope"),
    ]

    candidates = find_review_candidates(listing, classifier, max_llm_calls=budget)

    assert classifier.classify.call_args_list == [
        call("Lunch", "Meals"),
        call("Guide", "Meals"),
        call("Pickup", "Meals"),
    ]
    assert [candidate.included_item for candidate in candidates] == ["Lunch", "Pickup"]
    assert [candidate.reason for candidate in candidates] == [
        "Overlapping services",
        "Unclear scope",
    ]
    assert all(
        candidate.code == "POSSIBLE_INCLUSION_CONFLICT" for candidate in candidates
    )


@pytest.mark.parametrize("budget", [0, 1])
def test_budget_exhaustion_marks_each_unchecked_pair_in_order(budget):
    listing = make_listing(["Lunch", "Guide", "Pickup"], ["Meals"])
    classifier = Mock()
    classifier.classify.return_value = SemanticDecision(
        verdict="no_conflict", reason="Different services"
    )

    candidates = find_review_candidates(listing, classifier, max_llm_calls=budget)

    assert classifier.classify.call_count == budget
    assert [candidate.included_item for candidate in candidates] == listing.inclusions[
        budget:
    ]
    for candidate in candidates:
        assert candidate.code == "SEMANTIC_BUDGET_EXHAUSTED"
        assert candidate.field == "inclusions"
        assert candidate.excluded_item == "Meals"
        assert candidate.status == "pending"
        assert candidate.reason == (
            "Semantic comparison not performed: call budget exhausted; manual review required."
        )


def test_failed_attempt_consumes_budget_and_preserves_failure_candidate():
    listing = make_listing(["Lunch", "Guide", "Pickup"], ["Meals"])
    classifier = Mock()
    classifier.classify.side_effect = [
        SemanticClassificationError("Timeout"),
        SemanticDecision(verdict="uncertain", reason="Unclear scope"),
    ]

    candidates = find_review_candidates(listing, classifier, max_llm_calls=2)

    assert classifier.classify.call_args_list == [
        call("Lunch", "Meals"),
        call("Guide", "Meals"),
    ]
    assert [candidate.code for candidate in candidates] == [
        "SEMANTIC_CLASSIFICATION_FAILED",
        "POSSIBLE_INCLUSION_CONFLICT",
        "SEMANTIC_BUDGET_EXHAUSTED",
    ]
    assert [candidate.included_item for candidate in candidates] == listing.inclusions
    assert (
        candidates[0].reason
        == "Semantic classification failed; manual review required."
    )
    assert candidates[1].reason == "Unclear scope"


def test_exact_matches_and_duplicates_do_not_consume_budget():
    listing = make_listing(
        [" Lunch ", "LUNCH", "Drinks"], ["lunch", " Drinks ", "DRINKS"]
    )
    classifier = Mock()
    classifier.classify.return_value = SemanticDecision(
        verdict="uncertain", reason="Unclear"
    )

    candidates = find_review_candidates(listing, classifier, max_llm_calls=1)

    classifier.classify.assert_called_once_with(" Lunch ", " Drinks ")
    assert [
        (candidate.included_item, candidate.excluded_item) for candidate in candidates
    ] == [(" Lunch ", " Drinks "), ("Drinks", "lunch")]
    assert [candidate.code for candidate in candidates] == [
        "POSSIBLE_INCLUSION_CONFLICT",
        "SEMANTIC_BUDGET_EXHAUSTED",
    ]


def test_negative_budget_is_rejected_even_without_pairs():
    classifier = Mock()
    with pytest.raises(ValueError, match="max_llm_calls"):
        find_review_candidates(make_listing([], []), classifier, max_llm_calls=-1)
    classifier.classify.assert_not_called()

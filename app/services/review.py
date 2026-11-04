from typing import Protocol

from app.domain.models import Listing, ReviewCandidate, SemanticDecision


class SemanticClassifier(Protocol):
    def classify(
        self,
        included_item: str,
        excluded_item: str,
    ) -> SemanticDecision: ...


class SemanticClassificationError(Exception):
    """Semantic classifier could not produce a decision."""


def build_comparison_pairs(listing: Listing) -> list[tuple[str, str]]:
    pairs = []
    seen: set[tuple[str, str]] = set()

    for included in listing.inclusions:
        for excluded in listing.exclusions:
            pair = (included.strip().casefold(), excluded.strip().casefold())
            if pair[0] == pair[1] or pair in seen:
                continue

            seen.add(pair)
            pairs.append((included, excluded))

    return pairs


def find_review_candidates(
    listing: Listing,
    classifier: SemanticClassifier,
) -> list[ReviewCandidate]:
    candidates = []

    for included, excluded in build_comparison_pairs(listing):
        try:
            decision = classifier.classify(included, excluded)
        except SemanticClassificationError:
            candidates.append(
                ReviewCandidate(
                    code="SEMANTIC_CLASSIFICATION_FAILED",
                    field="inclusions",
                    included_item=included,
                    excluded_item=excluded,
                    reason="Semantic classification failed; manual review required.",
                )
            )
            continue

        if decision.verdict == "no_conflict":
            continue

        candidates.append(
            ReviewCandidate(
                code="POSSIBLE_INCLUSION_CONFLICT",
                field="inclusions",
                included_item=included,
                excluded_item=excluded,
                reason=decision.reason,
            )
        )

    return candidates

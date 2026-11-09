from typing import Literal

from app.domain.models import Issue, Listing, ReviewCandidate


def normalize_text(value: str) -> str:
    return value.strip().casefold()


def analyze_listing(listing: Listing) -> list[Issue]:
    issues = []

    if listing.city is None or not listing.city.strip():
        issues.append(
            Issue(
                code="MISSING_CITY",
                field="city",
                message="City is missing.",
            )
        )

    if listing.duration_minutes is None:
        issues.append(
            Issue(
                code="MISSING_DURATION",
                field="duration_minutes",
                message="Duration_minutes is missing.",
            )
        )

    if listing.meeting_point is None or not listing.meeting_point.strip():
        issues.append(
            Issue(
                code="MISSING_MEETING_POINT",
                field="meeting_point",
                message="Meeting point is missing.",
            )
        )

    normalized_exclusions = []

    for exclusion in listing.exclusions:
        normalized_exclusions.append(normalize_text(exclusion))

    # list comprehension
    # normalized_exclusions = [
    #     normalize_text(exclusion) for exclusion in listing.exclusions
    # ]

    for inclusion in listing.inclusions:
        normalized_inclusion = normalize_text(inclusion)

        if normalized_inclusion in normalized_exclusions:
            issues.append(
                Issue(
                    code="CONTRADICTORY_INCLUSION",
                    field="inclusions",
                    message=f"{inclusion} appears in both inclusions and exclusions.",
                )
            )

    if listing.price == 0:
        issues.append(
            Issue(
                code="ZERO_PRICE",
                field="price",
                message="Price is zero.",
            )
        )

    if len(set(listing.duration_mentions)) > 1:
        issues.append(
            Issue(
                code="DURATION_CONTRADICTION",
                field="duration_minutes",
                message="Multiple conflicting duration values were found.",
            )
        )

    return issues


def determine_verdict(
    issues: list[Issue],
    review_candidates: list[ReviewCandidate],
) -> tuple[Literal["pass", "review", "fail"], bool, str]:
    """Combine QA evidence; unrecognized issues conservatively require review."""
    fail_codes = {"CONTRADICTORY_INCLUSION", "DURATION_CONTRADICTION"}
    failures = [issue for issue in issues if issue.code in fail_codes]
    other_issues = [issue for issue in issues if issue.code not in fail_codes]
    pending = [
        candidate for candidate in review_candidates if candidate.status == "pending"
    ]

    reasons = [f"[{issue.code}] {issue.message}" for issue in failures + other_issues]
    reasons.extend(
        f"[{candidate.code}] {candidate.included_item} / {candidate.excluded_item}: {candidate.reason}"
        for candidate in pending
    )
    verdict: Literal["pass", "review", "fail"] = (
        "fail" if failures else "review" if reasons else "pass"
    )
    reason = (
        " ".join(reasons)
        if reasons
        else "No deterministic issues or pending semantic review candidates."
    )
    return verdict, verdict != "pass", reason

from app.domain.models import Issue, Listing


def normalize_text(value: str) -> str:
    return value.strip().lower()


def analyze_listing(listing: Listing) -> list[Issue]:
    issues = []

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

    return issues

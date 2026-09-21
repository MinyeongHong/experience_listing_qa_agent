from app.domain.models import Issue, Listing


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

    return issues

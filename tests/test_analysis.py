from app.domain.models import Listing
from app.services.analysis import analyze_listing


def test_missing_meeting_point():
    listing = Listing(
        title="Barcelona Tour",
        city="Barcelona",
        duration_minutes=180,
        price=45.0,
        meeting_point=None,
    )

    issues = analyze_listing(listing)

    assert len(issues) == 1
    assert issues[0].code == "MISSING_MEETING_POINT"


def test_valid_meeting_point():
    listing = Listing(
        title="Barcelona Tour",
        city="Barcelona",
        duration_minutes=180,
        price=45.0,
        meeting_point="La Rambla",
    )

    issues = analyze_listing(listing)

    assert len(issues) == 0

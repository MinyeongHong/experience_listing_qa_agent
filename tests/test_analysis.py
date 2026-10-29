from app.domain.models import Listing
from app.services.analysis import analyze_listing


def test_analysis_detects_blank_city():
    listing = Listing(
        title="Historic Walking Tour",
        city="  ",
        duration_minutes=90,
        price=45.0,
        meeting_point="La Rambla",
        inclusions=["Lunch"],
        exclusions=["Drinks"],
    )

    issues = analyze_listing(listing)

    assert len(issues) == 1
    assert issues[0].code == "MISSING_CITY"


def test_analysis_detects_missing_city_and_duration():
    listing = Listing(
        title="Barcelona Tour",
        city=None,
        duration_minutes=None,
        price=45.0,
        meeting_point="La Rambla",
        inclusions=["Lunch"],
        exclusions=["Drinks"],
    )

    issues = analyze_listing(listing)

    codes = {issue.code for issue in issues}

    assert "MISSING_CITY" in codes
    assert "MISSING_DURATION" in codes
    assert codes == {"MISSING_CITY", "MISSING_DURATION"}


def test_different_inclusion_and_exclusion():
    listing = Listing(
        title="Barcelona Tour",
        city="Barcelona",
        duration_minutes=180,
        price=45.0,
        meeting_point="La Rambla",
        inclusions=["Lunch"],
        exclusions=["Drinks"],
    )

    issues = analyze_listing(listing)

    assert len(issues) == 0


def test_empty_inclusions_and_exclusions():
    listing = Listing(
        title="Barcelona Tour",
        city="Barcelona",
        duration_minutes=180,
        price=45.0,
        meeting_point="La Rambla",
        inclusions=[],
        exclusions=[],
    )

    issues = analyze_listing(listing)

    assert len(issues) == 0


def test_contradictory_inclusion_is_normalized():
    listing = Listing(
        title="Barcelona Tour",
        city="Barcelona",
        duration_minutes=180,
        price=45.0,
        meeting_point="La Rambla",
        inclusions=["hotel pickup"],
        exclusions=["  HOTEL PICKUP  "],
    )

    issues = analyze_listing(listing)

    assert len(issues) == 1
    assert issues[0].code == "CONTRADICTORY_INCLUSION"


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


def test_contradictory_inclusion():
    listing = Listing(
        title="Barcelona Tour",
        city="Barcelona",
        duration_minutes=180,
        price=45.0,
        meeting_point="La Rambla",
        inclusions=["Hotel pickup"],
        exclusions=[" hotel PICKUP "],
        # inclusions=["Lunch", "Hotel pickup"],
        # exclusions=["Hotel pickup", "Drinks"],
    )

    issues = analyze_listing(listing)

    assert len(issues) == 1
    assert issues[0].code == "CONTRADICTORY_INCLUSION"


def test_duration_contradiction():
    listing = Listing(
        title="Barcelona Tour",
        city="Barcelona",
        duration_minutes=120,
        price=45,
        meeting_point="La Rambla",
        inclusions=[],
        exclusions=[],
        duration_mentions=[120, 180],
    )

    issues = analyze_listing(listing)

    assert len(issues) == 1
    assert issues[0].code == "DURATION_CONTRADICTION"


def test_same_duration_mentions():
    listing = Listing(
        title="Barcelona Tour",
        city="Barcelona",
        duration_minutes=120,
        price=45,
        meeting_point="La Rambla",
        inclusions=[],
        exclusions=[],
        duration_mentions=[120, 120],
    )

    issues = analyze_listing(listing)

    assert len(issues) == 0


def test_empty_duration_mentions():
    listing = Listing(
        title="Barcelona Tour",
        city="Barcelona",
        duration_minutes=120,
        price=45,
        meeting_point="La Rambla",
        inclusions=[],
        exclusions=[],
        duration_mentions=[],
    )

    issues = analyze_listing(listing)

    assert len(issues) == 0


def test_zero_price():
    listing = Listing(
        title="Barcelona Tour",
        city="Barcelona",
        duration_minutes=180,
        price=0,
        meeting_point="La Rambla",
        inclusions=[],
        exclusions=[],
    )

    issues = analyze_listing(listing)

    assert len(issues) == 1
    assert issues[0].code == "ZERO_PRICE"


def test_positive_price():
    listing = Listing(
        title="Barcelona Tour",
        city="Barcelona",
        duration_minutes=180,
        price=50,
        meeting_point="La Rambla",
        inclusions=[],
        exclusions=[],
    )

    issues = analyze_listing(listing)

    assert len(issues) == 0


def test_missing_price_does_not_trigger_zero_price():
    listing = Listing(
        title="Barcelona Tour",
        city="Barcelona",
        duration_minutes=180,
        price=None,
        meeting_point="La Rambla",
        inclusions=[],
        exclusions=[],
        duration_mentions=[180],
    )

    issues = analyze_listing(listing)

    assert all(issue.code != "ZERO_PRICE" for issue in issues)

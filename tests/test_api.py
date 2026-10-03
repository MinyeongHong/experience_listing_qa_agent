from fastapi.testclient import TestClient

from app.domain.models import Listing
from app.main import app
from app.services.analysis import analyze_listing

client = TestClient(app)


def test_create_analysis(monkeypatch):
    fake_listing = Listing(
        title="Barcelona Food Walking Tour",
        city="Barcelona",
        duration_minutes=120,
        price=35,
        meeting_point=None,
        inclusions=["Local guide", "Lunch"],
        exclusions=["Lunch", "Drinks"],
    )

    def fake_extract_listing(raw_text: str) -> Listing:
        return fake_listing

    monkeypatch.setattr(
        "app.main.extract_listing",
        fake_extract_listing,
    )

    response = client.post(
        "/analyses",
        json={"text": "some raw listing text"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["extracted_fields"]["title"] == "Barcelona Food Walking Tour"
    assert len(data["issues"]) == 2
    assert data["issues"][0]["code"] == "MISSING_MEETING_POINT"
    assert data["issues"][1]["code"] == "CONTRADICTORY_INCLUSION"


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

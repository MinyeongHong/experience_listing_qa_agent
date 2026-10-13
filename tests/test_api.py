from fastapi.testclient import TestClient

from app.domain.models import Listing
from app.main import app
from app.services.extraction import (
    ExtractionError,
    ExtractionTimeoutError,
    RateLimitError,
)

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


def test_create_analysis_returns_502_when_extraction_fails(monkeypatch):
    def fake_extract_listing(raw_text: str):
        raise ExtractionError("Gemini failed")

    monkeypatch.setattr(
        "app.main.extract_listing",
        fake_extract_listing,
    )

    response = client.post(
        "/analyses",
        json={"text": "some raw listing text"},
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "Listing extraction service failed."}


def test_create_analysis_returns_503_when_rate_limited(monkeypatch):
    def fake_extract_listing(raw_text: str):
        raise RateLimitError("Gemini rate limit exceeded")

    monkeypatch.setattr(
        "app.main.extract_listing",
        fake_extract_listing,
    )

    response = client.post(
        "/analyses",
        json={"text": "some raw listing text"},
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Listing extraction service is temporarily unavailable."
    }


def test_create_analysis_returns_504_when_extraction_times_out(monkeypatch):
    def fake_extract_listing(raw_text: str):
        raise ExtractionTimeoutError("Gemini request timed out")

    monkeypatch.setattr(
        "app.main.extract_listing",
        fake_extract_listing,
    )

    response = client.post(
        "/analyses",
        json={"text": "some raw listing text"},
    )

    assert response.status_code == 504
    assert response.json() == {"detail": "Listing extraction service timed out."}

from types import SimpleNamespace
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from app.domain.models import Listing
from app.main import app, get_listing_extractor
from app.services.extraction import (
    ExtractionError,
    ExtractionTimeoutError,
    RateLimitError,
)


@pytest.fixture(autouse=True)
def clear_dependency_overrides():
    yield
    app.dependency_overrides.clear()


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

    class FakeExtractor:
        def extract(self, raw_text: str) -> Listing:
            return fake_listing

    app.dependency_overrides[get_listing_extractor] = lambda: FakeExtractor()

    saved = []

    test_id = UUID("12345678-1234-5678-1234-567812345678")

    def fake_save_analysis(session, raw_text, result):
        saved.append((raw_text, result))

        return SimpleNamespace(id=test_id)

    monkeypatch.setattr(
        "app.main.save_analysis",
        fake_save_analysis,
    )

    response = client.post(
        "/analyses",
        json={"text": "some raw listing text"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["id"] == str(test_id)
    assert data["extracted_fields"]["title"] == "Barcelona Food Walking Tour"
    assert len(data["issues"]) == 2
    assert data["issues"][0]["code"] == "MISSING_MEETING_POINT"
    assert data["issues"][1]["code"] == "CONTRADICTORY_INCLUSION"

    assert len(saved) == 1
    assert saved[0][0] == "some raw listing text"

    saved_result = saved[0][1]

    assert saved_result.extracted_fields.title == "Barcelona Food Walking Tour"
    assert len(saved_result.issues) == 2


def test_create_analysis_with_missing_fields(monkeypatch):
    fake_listing = Listing(
        title="Historic Walking Tour",
        city=None,
        duration_minutes=None,
        price=35,
        meeting_point=None,
        inclusions=[],
        exclusions=[],
    )

    class FakeExtractor:
        def extract(self, raw_text: str) -> Listing:
            return fake_listing

    app.dependency_overrides[get_listing_extractor] = lambda: FakeExtractor()

    saved = []

    test_id = UUID("12345678-1234-5678-1234-567812345678")

    def fake_save_analysis(session, raw_text, result):
        saved.append((raw_text, result))

        return SimpleNamespace(id=test_id)

    monkeypatch.setattr(
        "app.main.save_analysis",
        fake_save_analysis,
    )

    response = client.post(
        "/analyses",
        json={"text": "some raw listing text"},
    )

    data = response.json()

    assert data["extracted_fields"]["city"] is None
    assert data["extracted_fields"]["duration_minutes"] is None

    assert len(saved) == 1
    assert saved[0][1].extracted_fields.city is None

    codes = {issue["code"] for issue in data["issues"]}

    assert "MISSING_CITY" in codes
    assert "MISSING_DURATION" in codes

    assert response.status_code == 200


def test_create_analysis_returns_502_when_extraction_fails(monkeypatch):
    class FakeExtractor:
        def extract(self, raw_text: str) -> Listing:
            raise ExtractionError("Gemini failed")

    app.dependency_overrides[get_listing_extractor] = lambda: FakeExtractor()

    response = client.post(
        "/analyses",
        json={"text": "some raw listing text"},
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "Listing extraction service failed."}


def test_create_analysis_returns_503_when_rate_limited(monkeypatch):
    class FakeExtractor:
        def extract(self, raw_text: str) -> Listing:
            raise RateLimitError("Gemini failed")

    app.dependency_overrides[get_listing_extractor] = lambda: FakeExtractor()

    response = client.post(
        "/analyses",
        json={"text": "some raw listing text"},
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Listing extraction service is temporarily unavailable."
    }


def test_create_analysis_returns_504_when_extraction_times_out(monkeypatch):
    class FakeExtractor:
        def extract(self, raw_text: str) -> Listing:
            raise ExtractionTimeoutError("Gemini failed")

    app.dependency_overrides[get_listing_extractor] = lambda: FakeExtractor()

    response = client.post(
        "/analyses",
        json={"text": "some raw listing text"},
    )

    assert response.status_code == 504
    assert response.json() == {"detail": "Listing extraction service timed out."}


def test_get_analysis(monkeypatch):
    test_id = UUID("12345678-1234-5678-1234-567812345678")

    fake_record = SimpleNamespace(
        id=test_id,
        extracted_fields={
            "title": "Barcelona Food Tour",
            "city": "Barcelona",
            "duration_minutes": 120,
            "price": 35,
            "meeting_point": None,
            "tags": [],
            "inclusions": [],
            "exclusions": [],
            "duration_mentions": [],
        },
        issues=[
            SimpleNamespace(
                code="MISSING_MEETING_POINT",
                field="meeting_point",
                message="Meeting point is missing.",
            )
        ],
        confidence=1.0,
    )

    def fake_get_analysis(session, analysis_id):
        assert analysis_id == test_id
        return fake_record

    monkeypatch.setattr(
        "app.main.get_analysis",
        fake_get_analysis,
    )

    response = client.get(f"/analyses/{test_id}")

    assert response.status_code == 200

    data = response.json()

    assert data["id"] == str(test_id)
    assert data["extracted_fields"]["title"] == "Barcelona Food Tour"
    assert len(data["issues"]) == 1
    assert data["issues"][0]["code"] == "MISSING_MEETING_POINT"


def test_get_analysis_not_found(monkeypatch):
    test_id = UUID("12345678-1234-5678-1234-567812345678")

    def fake_get_analysis(session, analysis_id):
        return None

    monkeypatch.setattr(
        "app.main.get_analysis",
        fake_get_analysis,
    )

    response = client.get(f"/analyses/{test_id}")

    assert response.status_code == 404
    assert response.json()["detail"] == "Analysis not found"


def test_get_analysis_with_missing_fields(monkeypatch):
    test_id = UUID("12345678-1234-5678-1234-567812345678")

    fake_record = SimpleNamespace(
        id=test_id,
        extracted_fields={
            "title": "Historic Walking Tour",
            "city": None,
            "duration_minutes": None,
            "price": 35,
            "meeting_point": None,
            "inclusions": [],
            "exclusions": [],
        },
        issues=[
            SimpleNamespace(
                code="MISSING_CITY",
                field="city",
                message="City is missing.",
            ),
            SimpleNamespace(
                code="MISSING_DURATION",
                field="duration_minutes",
                message="Duration is missing.",
            ),
        ],
        confidence=1.0,
    )

    def fake_get_analysis(session, analysis_id):
        return fake_record

    monkeypatch.setattr(
        "app.main.get_analysis",
        fake_get_analysis,
    )

    response = client.get(f"/analyses/{test_id}")

    assert response.status_code == 200

    data = response.json()

    assert data["extracted_fields"]["city"] is None
    assert data["extracted_fields"]["duration_minutes"] is None

    codes = {issue["code"] for issue in data["issues"]}

    assert "MISSING_CITY" in codes
    assert "MISSING_DURATION" in codes

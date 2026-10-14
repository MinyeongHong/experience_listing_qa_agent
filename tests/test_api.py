from types import SimpleNamespace
from uuid import UUID

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

    # 1. Gemini 추출 함수 대체
    def fake_extract_listing(raw_text: str) -> Listing:
        return fake_listing

    monkeypatch.setattr(
        "app.main.extract_listing",
        fake_extract_listing,
    )

    # 2. DB 저장 함수 대체
    saved = []

    test_id = UUID("12345678-1234-5678-1234-567812345678")

    def fake_save_analysis(session, raw_text, result):
        saved.append((raw_text, result))

        return SimpleNamespace(id=test_id)

    monkeypatch.setattr(
        "app.main.save_analysis",
        fake_save_analysis,
    )

    # 3. Mock 설정이 모두 끝난 후 API 호출
    response = client.post(
        "/analyses",
        json={"text": "some raw listing text"},
    )

    # 4. 응답 검증
    assert response.status_code == 200

    data = response.json()

    assert data["id"] == str(test_id)
    assert data["extracted_fields"]["title"] == "Barcelona Food Walking Tour"
    assert len(data["issues"]) == 2
    assert data["issues"][0]["code"] == "MISSING_MEETING_POINT"
    assert data["issues"][1]["code"] == "CONTRADICTORY_INCLUSION"

    # 5. DB 저장 함수 호출 검증
    assert len(saved) == 1
    assert saved[0][0] == "some raw listing text"

    saved_result = saved[0][1]

    assert saved_result.extracted_fields.title == "Barcelona Food Walking Tour"
    assert len(saved_result.issues) == 2


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

from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.db.session import engine
from app.domain.models import Listing
from app.main import app


@pytest.mark.integration
def test_create_and_read_analysis(monkeypatch):
    # 1. 테스트용 PostgreSQL 연결 확인
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))

    # 2. Gemini만 Mock
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

    client = TestClient(app)

    # 3. 실제 POST 요청 → PostgreSQL 저장
    post_response = client.post(
        "/analyses",
        json={"text": "Integration test: Barcelona food tour"},
    )

    assert post_response.status_code == 200

    created = post_response.json()
    analysis_id = UUID(created["id"])

    assert created["extracted_fields"]["title"] == fake_listing.title
    assert len(created["issues"]) == 2

    # 4. 실제 GET 요청 → PostgreSQL 조회
    get_response = client.get(f"/analyses/{analysis_id}")

    assert get_response.status_code == 200

    loaded = get_response.json()

    assert loaded["id"] == str(analysis_id)
    assert loaded["extracted_fields"] == created["extracted_fields"]
    assert loaded["issues"] == created["issues"]
    assert loaded["confidence"] == created["confidence"]

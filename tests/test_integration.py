import uuid
from uuid import UUID

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError

from app.db.models import AnalysisRecord, IssueRecord
from app.domain.models import Listing


@pytest.mark.integration
def test_create_and_read_analysis(
    monkeypatch,
    integration_client,
    db_session,
):
    # 1. 실제 연결된 DB 확인
    database_name = db_session.execute(text("SELECT current_database()")).scalar_one()

    assert database_name == "listingqa_test"

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

    # 3. 실제 POST 요청 → 테스트 DB 저장
    post_response = integration_client.post(
        "/analyses",
        json={"text": "Integration test: Barcelona food tour"},
    )

    assert post_response.status_code == 200

    created = post_response.json()
    analysis_id = UUID(created["id"])

    assert created["extracted_fields"]["title"] == fake_listing.title
    assert len(created["issues"]) == 2

    # 4. POST 결과가 실제 DB에 저장됐는지 확인
    saved_count = db_session.execute(text("SELECT COUNT(*) FROM analyses")).scalar_one()

    assert saved_count == 1

    # 5. 실제 GET 요청 → 테스트 DB 조회
    get_response = integration_client.get(f"/analyses/{analysis_id}")

    assert get_response.status_code == 200

    loaded = get_response.json()

    assert loaded["id"] == str(analysis_id)
    assert loaded["extracted_fields"] == created["extracted_fields"]
    assert loaded["issues"] == created["issues"]
    assert loaded["confidence"] == created["confidence"]


@pytest.mark.integration
def test_analysis_and_issues_rollback_on_failure(
    db_session,
    verification_session,
):
    test_text = f"Rollback test {uuid.uuid4()}"

    analysis = AnalysisRecord(
        raw_text=test_text,
        extracted_fields={
            "title": "Rollback Test",
            "city": "Barcelona",
            "duration_minutes": 60,
        },
        confidence=1.0,
        status="completed",
        issues=[
            IssueRecord(
                code=None,  # 의도적으로 NOT NULL 위반
                field="meeting_point",
                message="Test issue",
            )
        ],
    )

    db_session.add(analysis)

    with pytest.raises(IntegrityError):
        db_session.commit()

    # 실패한 Session을 정상 상태로 복구
    db_session.rollback()

    saved_analysis = verification_session.scalar(
        select(AnalysisRecord).where(AnalysisRecord.raw_text == test_text)
    )

    assert saved_analysis is None

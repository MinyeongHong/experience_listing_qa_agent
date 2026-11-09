import uuid
from unittest.mock import Mock
from uuid import UUID

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.db.models import AnalysisRecord, IssueRecord, ReviewCandidateRecord
from app.domain.models import (
    AnalysisResult,
    Issue,
    Listing,
    ReviewCandidate,
    SemanticDecision,
)
from app.integrations.gemini_classifier import GeminiSemanticClassifier
from app.main import app, get_listing_extractor, get_semantic_classifier
from app.services.persistence import save_analysis
from app.services.review import SemanticClassificationError


@pytest.mark.integration
@pytest.mark.parametrize(
    "scenario",
    [
        "pass",
        "review",
        "fail",
        "semantic_error",
        "budget",
        "nul_reason",
        "unicode_exact",
    ],
)
def test_create_and_read_analysis(
    integration_client,
    db_session,
    monkeypatch,
    scenario,
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
        meeting_point="Main gate",
        inclusions=["Local guide", "Food tasting"],
        exclusions=["Meals"],
    )
    monkeypatch.setenv("SEMANTIC_MAX_LLM_CALLS", "5")
    classifier = Mock()
    classifier.classify.return_value = SemanticDecision(
        verdict="no_conflict", reason="Different services"
    )
    expected_candidates = 0
    expected_issues = 0
    if scenario == "review":
        classifier.classify.return_value = SemanticDecision(
            verdict="uncertain", reason="Scope unclear"
        )
        expected_candidates = 2
    elif scenario == "fail":
        fake_listing.duration_mentions = [120, 180]
        expected_issues = 1
    elif scenario == "semantic_error":
        classifier.classify.side_effect = SemanticClassificationError("Timeout")
        fake_listing.meeting_point = None
        expected_candidates = 2
        expected_issues = 1
    elif scenario == "budget":
        monkeypatch.setenv("SEMANTIC_MAX_LLM_CALLS", "1")
        expected_candidates = 1
    elif scenario == "nul_reason":
        provider = Mock()
        provider.models.generate_content.return_value.parsed = {
            "verdict": "uncertain",
            "reason": "Scope unclear\x00",
        }
        classifier = Mock(wraps=GeminiSemanticClassifier(provider))
        expected_candidates = 2
    elif scenario == "unicode_exact":
        fake_listing.inclusions = [" Straße tour "]
        fake_listing.exclusions = ["STRASSE TOUR"]
        expected_issues = 1
    expected_verdict = scenario if scenario in {"pass", "review", "fail"} else "review"
    if scenario == "unicode_exact":
        expected_verdict = "fail"

    class FakeExtractor:
        def extract(self, raw_text: str) -> Listing:
            return fake_listing

    app.dependency_overrides[get_listing_extractor] = lambda: FakeExtractor()
    app.dependency_overrides[get_semantic_classifier] = lambda: classifier

    try:
        post_response = integration_client.post(
            "/analyses",
            json={"text": "Integration test: Barcelona food tour"},
        )

        assert post_response.status_code == 200

        created = post_response.json()
        analysis_id = UUID(created["id"])

        assert created["extracted_fields"]["title"] == fake_listing.title
        assert len(created["issues"]) == expected_issues
        assert len(created["review_candidates"]) == expected_candidates
        assert created["verdict"] == expected_verdict
        assert created["review_required"] is (expected_verdict != "pass")
        assert created["confidence"] is None
        if scenario in {"semantic_error", "nul_reason"}:
            assert all(
                item["code"] == "SEMANTIC_CLASSIFICATION_FAILED"
                for item in created["review_candidates"]
            )
        elif scenario == "budget":
            assert (
                created["review_candidates"][0]["code"] == "SEMANTIC_BUDGET_EXHAUSTED"
            )
        assert all(item["status"] == "pending" for item in created["review_candidates"])
        if scenario == "nul_reason":
            assert "\x00" not in created["reason"]
            assert all(
                "\x00" not in item["reason"] for item in created["review_candidates"]
            )
        elif scenario == "unicode_exact":
            classifier.classify.assert_not_called()
        stored = db_session.get(AnalysisRecord, analysis_id)
        assert stored.extracted_fields == fake_listing.model_dump()
        assert stored.verdict == expected_verdict
        rows = db_session.scalars(
            select(ReviewCandidateRecord).order_by(ReviewCandidateRecord.position)
        ).all()
        assert [row.position for row in rows] == list(range(expected_candidates))
        assert all(row.analysis_id == analysis_id for row in rows)
        assert [
            {field: getattr(row, field) for field in ReviewCandidate.model_fields}
            for row in rows
        ] == created["review_candidates"]
        calls_after_post = classifier.classify.call_count

        saved_count = db_session.execute(
            text("SELECT COUNT(*) FROM analyses")
        ).scalar_one()

        assert saved_count == 1

        get_response = integration_client.get(f"/analyses/{analysis_id}")

        assert get_response.status_code == 200

        loaded = get_response.json()

        assert loaded["id"] == str(analysis_id)
        assert loaded["extracted_fields"] == created["extracted_fields"]
        assert loaded["issues"] == created["issues"]
        assert loaded["confidence"] == created["confidence"]
        assert loaded == created
        assert classifier.classify.call_count == calls_after_post

    finally:
        app.dependency_overrides.pop(get_listing_extractor, None)
        app.dependency_overrides.pop(get_semantic_classifier, None)


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


@pytest.mark.integration
def test_candidate_insert_failure_rolls_back_entire_analysis(
    db_session, verification_session
):
    result = AnalysisResult(
        extracted_fields=Listing(title="Rollback tour"),
        issues=[Issue(code="MISSING_CITY", field="city", message="City missing")],
        # Pydantic permits this code, but PostgreSQL's VARCHAR(100) rejects it.
        review_candidates=[
            ReviewCandidate(
                code="X" * 101,
                field="inclusions",
                included_item="Lunch",
                excluded_item="Meals",
                reason="Scope unclear",
            )
        ],
        verdict="review",
        review_required=True,
        reason="Review needed",
        confidence=None,
    )
    with pytest.raises(SQLAlchemyError):
        save_analysis(db_session, "Rollback source", result)
    for table in ("analyses", "issues", "review_candidates"):
        assert verification_session.scalar(text(f"SELECT COUNT(*) FROM {table}")) == 0

from unittest.mock import MagicMock

import pytest
from sqlalchemy.exc import SQLAlchemyError

from app.domain.models import AnalysisResult, Issue, Listing, ReviewCandidate
from app.services.persistence import save_analysis


def test_save_analysis_rolls_back_when_commit_fails():
    session = MagicMock()
    session.commit.side_effect = SQLAlchemyError("Commit failed")

    listing = Listing(
        title="Barcelona Walking Tour",
        city="Barcelona",
        duration_minutes=120,
        price=35,
        meeting_point="Plaça de Catalunya",
        inclusions=["Local guide"],
        exclusions=[],
    )

    result = AnalysisResult(
        extracted_fields=listing,
        issues=[],
        confidence=1.0,
        verdict="pass",
        review_required=False,
        reason="No issues",
    )

    with pytest.raises(SQLAlchemyError, match="Commit failed"):
        save_analysis(
            session=session,
            raw_text="Barcelona walking tour",
            result=result,
        )

    session.add.assert_called_once()
    session.commit.assert_called_once()
    session.rollback.assert_called_once()
    session.refresh.assert_not_called()


def test_save_analysis_stores_all_qa_results_with_one_commit():
    session = MagicMock()
    candidates = [
        ReviewCandidate(
            code="SEMANTIC_CLASSIFICATION_FAILED",
            field="inclusions",
            included_item="Lunch",
            excluded_item="Meals",
            reason="Failed",
        ),
        ReviewCandidate(
            code="SEMANTIC_BUDGET_EXHAUSTED",
            field="inclusions",
            included_item="Guide",
            excluded_item="Meals",
            reason="Budget exhausted",
        ),
    ]
    result = AnalysisResult(
        extracted_fields=Listing(title="Tour"),
        issues=[Issue(code="MISSING_CITY", field="city", message="City is missing")],
        review_candidates=candidates,
        verdict="review",
        review_required=True,
        reason="City missing; semantic checks incomplete",
        confidence=None,
    )

    record = save_analysis(session, "Source listing", result)

    session.add.assert_called_once_with(record)
    session.commit.assert_called_once()
    session.refresh.assert_called_once_with(record)
    assert record.extracted_fields == result.extracted_fields.model_dump()
    assert record.confidence is None
    assert (record.verdict, record.review_required, record.reason) == (
        result.verdict,
        True,
        result.reason,
    )
    assert record.issues[0].code == "MISSING_CITY"
    assert [row.position for row in record.review_candidates] == [0, 1]
    assert [row.code for row in record.review_candidates] == [
        item.code for item in candidates
    ]
    assert all(row.status == "pending" for row in record.review_candidates)

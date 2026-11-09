from uuid import UUID

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, selectinload

from app.db.models import AnalysisRecord, IssueRecord, ReviewCandidateRecord
from app.domain.models import AnalysisResult


def save_analysis(
    session: Session,
    raw_text: str,
    result: AnalysisResult,
) -> AnalysisRecord:
    record = AnalysisRecord(
        raw_text=raw_text,
        extracted_fields=result.extracted_fields.model_dump(),
        confidence=result.confidence,
        verdict=result.verdict,
        review_required=result.review_required,
        reason=result.reason,
        status="completed",
        issues=[
            IssueRecord(
                code=issue.code,
                field=issue.field,
                message=issue.message,
            )
            for issue in result.issues
        ],
        review_candidates=[
            ReviewCandidateRecord(position=position, **candidate.model_dump())
            for position, candidate in enumerate(result.review_candidates)
        ],
    )

    try:
        session.add(record)
        session.commit()
        session.refresh(record)
    except SQLAlchemyError:
        session.rollback()
        raise

    return record


def get_analysis(
    session: Session,
    analysis_id: UUID,
) -> AnalysisRecord | None:
    return session.get(
        AnalysisRecord,
        analysis_id,
        options=[
            selectinload(AnalysisRecord.issues),
            selectinload(AnalysisRecord.review_candidates),
        ],
    )

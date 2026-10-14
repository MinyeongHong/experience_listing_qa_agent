from uuid import UUID

from sqlalchemy.orm import Session, selectinload

from app.db.models import AnalysisRecord, IssueRecord
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
        status="completed",
        issues=[
            IssueRecord(
                code=issue.code,
                field=issue.field,
                message=issue.message,
            )
            for issue in result.issues
        ],
    )

    session.add(record)
    session.commit()
    session.refresh(record)

    return record


def get_analysis(
    session: Session,
    analysis_id: UUID,
) -> AnalysisRecord | None:
    return session.get(
        AnalysisRecord,
        analysis_id,
        options=[selectinload(AnalysisRecord.issues)],
    )

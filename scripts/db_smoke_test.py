import uuid

from app.db.models import AnalysisRecord, IssueRecord
from app.db.session import SessionLocal


def main():
    analysis_id = uuid.uuid4()

    with SessionLocal() as session:
        analysis = AnalysisRecord(
            id=analysis_id,
            raw_text="Barcelona Food Tour. Duration: 2 hours.",
            extracted_fields={
                "title": "Barcelona Food Tour",
                "city": "Barcelona",
                "duration_minutes": 120,
            },
            confidence=1.0,
            status="completed",
            issues=[
                IssueRecord(
                    code="MISSING_MEETING_POINT",
                    field="meeting_point",
                    message="Meeting point is missing.",
                )
            ],
        )

        session.add(analysis)
        session.commit()

        print("Saved analysis:", analysis_id)

    with SessionLocal() as session:
        saved_analysis = session.get(AnalysisRecord, analysis_id)

        if saved_analysis is None:
            raise RuntimeError("Analysis not found")

        print("Loaded title:", saved_analysis.extracted_fields["title"])
        print("Loaded status:", saved_analysis.status)


if __name__ == "__main__":
    main()

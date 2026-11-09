import uuid

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles, deregister
from sqlalchemy.orm import Session

from app.db import models
from app.domain.models import ReviewCandidate


@pytest.fixture
def orm_engine():
    # Exercise ORM persistence offline; PostgreSQL migration is tested separately.
    @compiles(JSONB, "sqlite")
    def compile_jsonb(type_, compiler, **kwargs):
        return "JSON"

    engine = create_engine("sqlite://")
    try:
        models.Base.metadata.create_all(engine)
        yield engine
    finally:
        engine.dispose()
        deregister(JSONB)


def test_analysis_nullable_confidence_and_conservative_defaults(orm_engine):
    with Session(orm_engine) as session:
        record = models.AnalysisRecord(
            raw_text="Tour", extracted_fields={"title": "Tour"}
        )
        session.add(record)
        session.commit()
        session.refresh(record)

        assert record.confidence is None
        assert record.verdict == "review"
        assert record.review_required is True
        assert "Semantic QA" in record.reason and "not" in record.reason
        assert record.status == "completed"
        assert record.review_candidates == []


def test_review_candidates_round_trip_order_and_delete_orphan(orm_engine):
    expected = ReviewCandidate(
        code="SEMANTIC_BUDGET_EXHAUSTED",
        field="inclusions",
        included_item="Lunch",
        excluded_item="Meals",
        reason="Budget exhausted",
    )
    with Session(orm_engine) as session:
        record = models.AnalysisRecord(
            raw_text="Tour",
            extracted_fields={"title": "Tour"},
            confidence=None,
            verdict="fail",
            review_required=True,
            reason="Deterministic contradiction",
            review_candidates=[
                models.ReviewCandidateRecord(position=1, **expected.model_dump()),
                models.ReviewCandidateRecord(position=0, **expected.model_dump()),
            ],
        )
        session.add(record)
        session.commit()
        session.expire_all()
        loaded = session.get(models.AnalysisRecord, record.id)
        assert loaded.verdict == "fail"
        assert [item.position for item in loaded.review_candidates] == [0, 1]
        for candidate in loaded.review_candidates:
            assert isinstance(candidate.id, uuid.UUID)
            assert candidate.analysis_id == record.id
            assert candidate.analysis is loaded
            assert {
                field: getattr(candidate, field)
                for field in type(expected).model_fields
            } == expected.model_dump()

        removed_id = loaded.review_candidates.pop().id
        session.commit()
        assert session.get(models.ReviewCandidateRecord, removed_id) is None
        session.delete(loaded)
        session.commit()
        assert session.scalars(select(models.ReviewCandidateRecord)).all() == []

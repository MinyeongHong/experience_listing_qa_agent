import uuid
from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, true
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class AnalysisRecord(Base):
    __tablename__ = "analyses"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    raw_text: Mapped[str] = mapped_column(Text, nullable=False)

    extracted_fields: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
    )

    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    verdict: Mapped[str] = mapped_column(
        String(10), nullable=False, server_default="review"
    )
    review_required: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=true()
    )
    reason: Mapped[str] = mapped_column(
        Text, nullable=False, server_default="Semantic QA has not been performed."
    )

    status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="completed",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )

    issues: Mapped[list["IssueRecord"]] = relationship(
        back_populates="analysis",
        cascade="all, delete-orphan",
    )

    review_candidates: Mapped[list["ReviewCandidateRecord"]] = relationship(
        back_populates="analysis",
        cascade="all, delete-orphan",
        order_by="ReviewCandidateRecord.position",
    )


class IssueRecord(Base):
    __tablename__ = "issues"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    analysis_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("analyses.id"),
        nullable=False,
        index=True,
    )

    code: Mapped[str] = mapped_column(String(100), nullable=False)
    field: Mapped[str] = mapped_column(String(100), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)

    analysis: Mapped["AnalysisRecord"] = relationship(
        back_populates="issues",
    )


class ReviewCandidateRecord(Base):
    __tablename__ = "review_candidates"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    analysis_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("analyses.id"), nullable=False, index=True
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    code: Mapped[str] = mapped_column(String(100), nullable=False)
    field: Mapped[str] = mapped_column(String(100), nullable=False)
    included_item: Mapped[str] = mapped_column(Text, nullable=False)
    excluded_item: Mapped[str] = mapped_column(Text, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        String(30), nullable=False, server_default="pending"
    )

    analysis: Mapped["AnalysisRecord"] = relationship(
        back_populates="review_candidates"
    )

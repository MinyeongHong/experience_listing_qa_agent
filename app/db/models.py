import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, ForeignKey, String, Text
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

    confidence: Mapped[float] = mapped_column(Float, nullable=False)

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

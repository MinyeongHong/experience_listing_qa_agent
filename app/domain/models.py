from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, PositiveInt, field_validator


class Listing(BaseModel):
    title: str
    city: str | None = None
    duration_minutes: int | None = Field(default=None, ge=1)
    price: float | None = Field(default=None, ge=0)
    meeting_point: str | None = None
    tags: list[str] = Field(default_factory=list)
    inclusions: list[str] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)
    duration_mentions: list[PositiveInt] = Field(default_factory=list)

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("title cannot be empty")

        return value


class Issue(BaseModel):
    code: str
    field: str
    message: str


class ReviewCandidate(BaseModel):
    code: str
    field: str
    included_item: str
    excluded_item: str
    reason: str
    status: Literal["pending", "confirmed", "dismissed"] = "pending"


class AnalysisResult(BaseModel):
    extracted_fields: Listing
    issues: list[Issue]
    review_candidates: list[ReviewCandidate] = Field(default_factory=list)
    verdict: Literal["pass", "review", "fail"]
    review_required: bool
    reason: str
    confidence: float | None = Field(default=None, ge=0, le=1)


class AnalysisRequest(BaseModel):
    text: str = Field(max_length=10_000)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text cannot be blank")
        return value


class AnalysisResponse(AnalysisResult):
    id: UUID


class SemanticDecision(BaseModel):
    verdict: Literal[
        "possible_conflict",
        "no_conflict",
        "uncertain",
    ]
    reason: str = Field(min_length=1)

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        if "\x00" in value:
            raise ValueError("reason cannot contain NUL characters")
        if not value.strip():
            raise ValueError("reason cannot be blank")
        return value

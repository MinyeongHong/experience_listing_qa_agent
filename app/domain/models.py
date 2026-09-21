from pydantic import BaseModel, Field, field_validator


class Listing(BaseModel):
    title: str
    city: str
    duration_minutes: int = Field(gt=0)
    price: float = Field(ge=0)
    meeting_point: str | None = None
    tags: list[str] = Field(default_factory=list)

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


class AnalysisResult(BaseModel):
    extracted_fields: Listing
    issues: list[Issue]
    confidence: float = Field(ge=0, le=1)

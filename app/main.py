from typing import Annotated
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.domain.models import (
    AnalysisRequest,
    AnalysisResponse,
    AnalysisResult,
    Issue,
    Listing,
)
from app.services.analysis import analyze_listing
from app.services.extraction import (
    ExtractionError,
    ExtractionTimeoutError,
    RateLimitError,
    extract_listing,
)
from app.services.persistence import get_analysis, save_analysis

app = FastAPI()


@app.post("/analyses", response_model=AnalysisResponse)
def create_analysis(
    request: AnalysisRequest,
    session: Annotated[Session, Depends(get_db)],
) -> AnalysisResponse:
    try:
        listing = extract_listing(request.text)

    except RateLimitError as exc:
        raise HTTPException(
            status_code=503,
            detail="Listing extraction service is temporarily unavailable.",
        ) from exc

    except ExtractionTimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail="Listing extraction service timed out.",
        ) from exc

    except ExtractionError as exc:
        raise HTTPException(
            status_code=502,
            detail="Listing extraction service failed.",
        ) from exc

    issues = analyze_listing(listing)

    result = AnalysisResult(
        extracted_fields=listing,
        issues=issues,
        confidence=1.0,
    )

    record = save_analysis(session, request.text, result)

    return AnalysisResponse(
        id=record.id,
        **result.model_dump(),
    )


@app.get("/analyses/{analysis_id}", response_model=AnalysisResponse)
def read_analysis(
    analysis_id: UUID,
    session: Annotated[Session, Depends(get_db)],
) -> AnalysisResponse:
    record = get_analysis(session, analysis_id)

    if record is None:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found",
        )

    return AnalysisResponse(
        id=record.id,
        extracted_fields=Listing.model_validate(record.extracted_fields),
        issues=[
            Issue(
                code=issue.code,
                field=issue.field,
                message=issue.message,
            )
            for issue in record.issues
        ],
        confidence=record.confidence,
    )

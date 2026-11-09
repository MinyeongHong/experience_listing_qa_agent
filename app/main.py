import os
from contextlib import asynccontextmanager
from typing import Annotated
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Request
from sqlalchemy.orm import Session

from app.db.models import AnalysisRecord
from app.db.session import get_db
from app.domain.models import (
    AnalysisRequest,
    AnalysisResponse,
    AnalysisResult,
    Issue,
    Listing,
    ReviewCandidate,
)
from app.integrations.gemini_classifier import GeminiSemanticClassifier
from app.integrations.gemini_client import create_gemini_client
from app.services.analysis import analyze_listing, determine_verdict
from app.services.extraction import (
    ExtractionError,
    ExtractionTimeoutError,
    ListingExtractor,
    RateLimitError,
)
from app.services.persistence import get_analysis, save_analysis
from app.services.review import SemanticClassifier, find_review_candidates


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = create_gemini_client()

    app.state.gemini_client = client

    try:
        get_semantic_call_budget()
        yield
    finally:
        client.close()


app = FastAPI(lifespan=lifespan)


def get_listing_extractor(request: Request) -> ListingExtractor:
    client = request.app.state.gemini_client
    model = os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"

    return ListingExtractor(
        client=client,
        model=model,
    )


def get_semantic_classifier(request: Request) -> SemanticClassifier:
    return GeminiSemanticClassifier(
        client=request.app.state.gemini_client,
        model=os.getenv("GEMINI_MODEL") or "gemini-2.5-flash",
    )


def get_semantic_call_budget() -> int:
    try:
        budget = int(os.getenv("SEMANTIC_MAX_LLM_CALLS", "5"))
    except ValueError as exc:
        raise ValueError(
            "SEMANTIC_MAX_LLM_CALLS must be a non-negative integer"
        ) from exc
    if budget < 0:
        raise ValueError("SEMANTIC_MAX_LLM_CALLS must be a non-negative integer")
    return budget


def _analysis_response(record: AnalysisRecord) -> AnalysisResponse:
    return AnalysisResponse(
        id=record.id,
        extracted_fields=Listing.model_validate(record.extracted_fields),
        issues=[
            Issue.model_validate(issue, from_attributes=True) for issue in record.issues
        ],
        review_candidates=[
            ReviewCandidate.model_validate(candidate, from_attributes=True)
            for candidate in record.review_candidates
        ],
        verdict=record.verdict,
        review_required=record.review_required,
        reason=record.reason,
        confidence=record.confidence,
    )


@app.post("/analyses", response_model=AnalysisResponse)
def create_analysis(
    request: AnalysisRequest,
    session: Annotated[Session, Depends(get_db)],
    extractor: Annotated[ListingExtractor, Depends(get_listing_extractor)],
    classifier: Annotated[SemanticClassifier, Depends(get_semantic_classifier)],
    max_llm_calls: Annotated[int, Depends(get_semantic_call_budget)],
) -> AnalysisResponse:
    try:
        listing = extractor.extract(request.text)

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
    review_candidates = find_review_candidates(
        listing, classifier, max_llm_calls=max_llm_calls
    )
    verdict, review_required, reason = determine_verdict(issues, review_candidates)

    result = AnalysisResult(
        extracted_fields=listing,
        issues=issues,
        review_candidates=review_candidates,
        verdict=verdict,
        review_required=review_required,
        reason=reason,
        confidence=None,
    )

    record = save_analysis(session, request.text, result)

    return _analysis_response(record)


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

    return _analysis_response(record)

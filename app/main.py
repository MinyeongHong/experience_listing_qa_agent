from fastapi import FastAPI, HTTPException

from app.domain.models import AnalysisRequest, AnalysisResult
from app.services.analysis import analyze_listing
from app.services.extraction import (
    ExtractionError,
    ExtractionTimeoutError,
    RateLimitError,
    extract_listing,
)

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyses", response_model=AnalysisResult)
def create_analysis(request: AnalysisRequest) -> AnalysisResult:
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

    return AnalysisResult(
        extracted_fields=listing,
        issues=issues,
        confidence=1.0,
    )

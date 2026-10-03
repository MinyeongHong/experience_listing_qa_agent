from fastapi import FastAPI

from app.domain.models import AnalysisRequest, AnalysisResult
from app.services.analysis import analyze_listing
from app.services.extraction import extract_listing

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyses", response_model=AnalysisResult)
def create_analysis(request: AnalysisRequest) -> AnalysisResult:
    listing = extract_listing(request.text)
    issues = analyze_listing(listing)

    return AnalysisResult(
        extracted_fields=listing,
        issues=issues,
        confidence=1.0,
    )

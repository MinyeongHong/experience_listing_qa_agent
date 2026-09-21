from fastapi import FastAPI

from app.domain.models import AnalysisResult, Listing
from app.services.analysis import analyze_listing

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyses", response_model=AnalysisResult)
def create_analysis(listing: Listing) -> AnalysisResult:
    issues = analyze_listing(listing)

    return AnalysisResult(
        extracted_fields=listing,
        issues=issues,
        confidence=1.0,
    )

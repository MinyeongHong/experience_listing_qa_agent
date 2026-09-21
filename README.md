# Experience Listing QA Agent

An AI-assisted backend system for analyzing tour and activity listings.

The goal of this project is to build a production-like AI workflow that extracts structured listing data, detects missing or inconsistent information, and provides evidence for human review.

## Current Features

- FastAPI backend
- Pydantic request and response schemas
- Listing field validation
- Deterministic QA rules
- Structured issue reporting
- Pytest-based rule tests

## API

### GET /health

Checks whether the API is running.

### POST /analyses

Accepts a structured listing and returns an analysis result.

Example response:

```json
{
  "extracted_fields": {
    "title": "Barcelona Tour",
    "city": "Barcelona",
    "duration_minutes": 180,
    "price": 45.0,
    "meeting_point": null,
    "tags": []
  },
  "issues": [
    {
      "code": "MISSING_MEETING_POINT",
      "field": "meeting_point",
      "message": "Meeting point is missing."
    }
  ],
  "confidence": 1.0
}
```

## Run Locally

```bash
uv sync
uv run uvicorn app.main:app --reload
```

API documentation:

`http://127.0.0.1:8000/docs`

## Tests

```bash
uv run pytest
```

## Tech Stack

- Python 3.12
- FastAPI
- Pydantic
- Pytest
- Ruff

## Status

Work in progress.

Planned features include LLM-based structured extraction, additional validation rules, persistence, retrieval with citations, evaluation, and human review workflows.
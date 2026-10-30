# Experience Listing QA Agent

An AI-assisted backend for identifying missing information and inconsistencies in tour and activity listings.

The application extracts structured fields from unstructured listing text with Gemini, applies deterministic quality-assurance rules, and stores analysis results for later retrieval. A separate semantic-conflict classifier and evaluation workflow are implemented for experimentation; semantic classification is **not yet connected to the analysis API**.

## Current capabilities

- **Structured extraction:** Gemini returns a Pydantic `Listing` from raw listing text.
- **Missing-field checks:** Deterministic rules flag missing information, including city, duration, and meeting point.
- **Consistency checks:** Deterministic rules identify issues such as exact matches between inclusions and exclusions.
- **Persistence:** SQLAlchemy stores analyses and their issues in PostgreSQL.
- **Retrieval:** Analyses can be fetched by UUID.
- **Error handling:** Extraction errors are mapped to HTTP 502, rate limits to 503, and timeouts to 504.
- **Semantic evaluation:** A separate Gemini-based inclusion/exclusion classifier can be evaluated in live and replay modes.

## Architecture

```text
Raw listing text
       |
       v
POST /analyses (FastAPI)
       |
       v
ListingExtractor (Gemini structured output)
       |
       v
Listing (Pydantic)
       |
       v
Deterministic QA rules
       |
       v
AnalysisResult + issues
       |
       v
PostgreSQL (SQLAlchemy)
       |
       v
GET /analyses/{analysis_id}
```

The Gemini client is injected into `ListingExtractor`. The FastAPI application manages the client lifecycle, and tests can replace the extractor without making external API calls.

The semantic classifier and its evaluation pipeline currently run separately from this API flow.

## Tech stack

- Python 3.12+
- FastAPI and Pydantic
- Google Gen AI SDK (Gemini)
- PostgreSQL, SQLAlchemy, Alembic, and psycopg
- uv for dependency management
- pytest for testing and Ruff for linting

## Getting started

### Prerequisites

Install Python 3.12+, [uv](https://docs.astral.sh/uv/), and PostgreSQL. Create a PostgreSQL database and obtain a Gemini API key.

### Install dependencies

```bash
uv sync
```

### Configure environment variables

Create a `.env` file in the project root. Do not commit it.

```dotenv
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@localhost:5432/DATABASE_NAME
```

Replace the placeholder values with your own credentials and database details. `GEMINI_MODEL` is used by the evaluation runner and can also configure the API extractor; the API extractor has a default model if it is omitted.

### Prepare the database

The project uses Alembic for database migrations. After configuring `DATABASE_URL`, apply the migrations:

```bash
uv run alembic upgrade head
```

### Run the API

```bash
uv run uvicorn app.main:app --reload
```

Interactive API documentation is available at <http://127.0.0.1:8000/docs>.

## API

### `POST /analyses`

Accepts **raw listing text** and returns an analysis with a generated ID, extracted fields, detected issues, and a confidence value.

Example request:

```bash
curl -X POST http://127.0.0.1:8000/analyses \
  -H 'Content-Type: application/json' \
  -d '{"text":"Barcelona food walking tour. Duration: 2 hours. Price: €35. Includes a local guide and lunch. Excludes lunch and drinks."}'
```

Example response structure (illustrative, not a guaranteed model output):

```json
{
  "id": "12345678-1234-5678-1234-567812345678",
  "extracted_fields": {
    "title": "Barcelona Food Walking Tour",
    "city": "Barcelona",
    "duration_minutes": 120,
    "price": 35,
    "meeting_point": null,
    "tags": [],
    "inclusions": ["Local guide", "Lunch"],
    "exclusions": ["Lunch", "Drinks"],
    "duration_mentions": [120]
  },
  "issues": [
    {
      "code": "MISSING_MEETING_POINT",
      "field": "meeting_point",
      "message": "Meeting point is missing."
    },
    {
      "code": "CONTRADICTORY_INCLUSION",
      "field": "inclusions",
      "message": "An item appears in both inclusions and exclusions."
    }
  ],
  "confidence": 1.0
}
```

The example shows the response shape; exact issue messages and extracted values depend on the implementation and input. The current confidence value is fixed in the API flow and should **not** be interpreted as a calibrated probability.

Extraction-related HTTP errors:

| Status | Meaning |
| --- | --- |
| `502` | Extraction service failure or invalid structured response |
| `503` | Gemini rate limit |
| `504` | Gemini request timeout |

### `GET /analyses/{analysis_id}`

Returns a previously stored analysis by UUID. Returns `404` when the analysis does not exist.

```bash
curl http://127.0.0.1:8000/analyses/12345678-1234-5678-1234-567812345678
```

## Extraction and QA rules

The extraction prompt distinguishes between explicit information and missing fields:

- **Title:** Extract an existing title or generate a concise title grounded in the source.
- **City:** Extract an explicit city or infer it only from an unambiguous landmark or geographic reference; otherwise return `null`.
- **Duration:** Extract only explicitly stated durations and convert them to minutes; otherwise return `null`. Preserve duration mentions for conflict analysis.
- **Price:** Do not invent missing prices.
- **Meeting point:** Require a specific meeting location; vague descriptions do not qualify.
- **Tags:** Extract from explicit tags or category sections only.
- **Inclusions/exclusions:** Preserve explicitly stated items and contradictions rather than silently correcting them.

Deterministic rules report missing fields and known contradictions. Semantic similarity between differently worded inclusion and exclusion items is handled by a separate experimental classifier, not by the current API endpoint.

## Semantic evaluation

The evaluation runner is `evals/run_semantic_eval.py`. It uses cases from `evals/semantic_cases.json` and classifies inclusion/exclusion pairs with Gemini.

### Live evaluation

Requires `GEMINI_API_KEY` and `GEMINI_MODEL` in the environment.

```bash
uv run python -m evals.run_semantic_eval
```

The runner writes:

- `evals/semantic_results.json`: per-case successful decisions and provider errors
- `evals/semantic_report.json`: aggregate metrics, coverage, error counts, mismatches, and run metadata

Exact string matches are skipped because they are handled by deterministic rules.

### Replay evaluation

Recalculate metrics from saved results without calling Gemini:

```bash
uv run python -m evals.run_semantic_eval --replay
```

To replay a different results file:

```bash
uv run python -m evals.run_semantic_eval --replay --results-path path/to/results.json
```

Metrics include confusion-matrix counts (TP, FP, FN, TN), precision, recall, and F1. Live reports also track evaluation coverage, provider error rate, skipped cases, and metadata. Replay mode cannot reconstruct skipped-case counts or full dataset coverage from a results file alone.

## Tests and linting

Run all tests:

```bash
uv run pytest -q
```

Run unit tests for extraction, API, and persistence behavior:

```bash
uv run pytest tests/test_extraction.py tests/test_api.py tests/test_persistence.py -q
```

Run the database integration test separately when a configured test database is available:

```bash
uv run pytest -m integration -q
```

Run Ruff:

```bash
uv run ruff check .
```

Tests use fake Gemini clients and FastAPI dependency overrides to avoid external model calls. Database integration tests require a real, appropriately configured test database.

## Current limitations and next steps

- Semantic conflict detection is evaluated separately and is not yet part of the POST analysis pipeline.
- The API's `confidence` field is currently a fixed value rather than a measured or calibrated confidence score.
- Extraction depends on an external model, so responses can vary; factual accuracy and prompt-injection robustness require further evaluation.
- Dataset expansion, regression evaluation, CI, and input-size safeguards remain future work.
- Retry/idempotency behavior and post-commit failure handling may require additional hardening for production use.

## Project status

Active development. The application has working extraction, deterministic analysis, persistence, retrieval, and tests, but is not presented as a production-ready service.

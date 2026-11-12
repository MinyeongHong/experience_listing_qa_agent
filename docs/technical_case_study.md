# Experience Listing QA Agent

A scoped AI engineering portfolio project combining LLM extraction, explicit QA
policy, bounded semantic reasoning, and a persistent REST backend.

## 1. Problem & Objective

Travel experience listings can omit practical details or contradict themselves
about included services. Exact matching cannot resolve every overlap, such as
included lunch versus excluded meals. The objective is to surface quality issues
and retain evidence for human review, without treating uncertain model judgments
as established defects.

## 2. Architecture

**Raw Listing → Gemini Structured Extraction → Deterministic QA → Budgeted Semantic QA → Verdict → PostgreSQL**

[FastAPI](../app/main.py) accepts text through authenticated `POST /analyses`.
Gemini receives extraction instructions separately from source text and returns
JSON validated as a [Pydantic Listing](../app/domain/models.py). A shared Gemini
client is managed through the application lifespan.

[Deterministic rules](../app/services/analysis.py) check missing city, duration,
and meeting point; zero price; conflicting durations; and normalized exact
inclusion/exclusion matches. Remaining unique pairs undergo
[semantic classification](../app/integrations/gemini_classifier.py).
A pure function combines evidence using **fail > review > pass**, without another
LLM call. SQLAlchemy stores the source, Listing JSONB, issues, ordered review
candidates, and verdict together. UUID-based `GET /analyses/{id}` retrieves stored
results without rerunning Gemini.

## 3. Key Engineering Decisions

**Rules versus semantic reasoning.** Explicit inclusion/exclusion and duration
contradictions produce `fail`; missing information and zero price produce `review`.
Semantic `possible_conflict` and `uncertain` decisions remain pending candidates
with reasons. This preserves human judgment; no reviewer UI or adjudication API
is implemented.

**Bounded usage with conservative fallback.** The API defaults to a maximum of
five semantic classifier attempts per request, configurable through
`SEMANTIC_MAX_LLM_CALLS`. Failed attempts consume budget. Normalized duplicate
pairs and exact matches do not. Classification errors and budget exhaustion create
pending candidates rather than silently passing unchecked pairs. Successful
extraction and deterministic results are preserved; extraction failures instead
return HTTP errors. SDK retries and extraction are outside this classifier budget.

**Validation and traceability.** Pydantic constrains structure and values, while
source grounding remains an evaluation concern. Both analysis endpoints require
`X-API-Key`, checked with `secrets.compare_digest()`.
[Persistence](../app/services/persistence.py) commits related records in one
transaction; [Alembic migrations](../migrations/versions/a91c3e7d2b40_add_verdict_and_review_candidates.py)
maintain schema evolution. Retaining source and extracted fields supports inspection,
but does not make future model outputs reproducible.

## 4. Local Development & Deployment

[Docker Compose](../compose.yaml) provisions only local PostgreSQL using
`postgres:16`, a named data volume, and a localhost port binding. Local FastAPI
runs directly in a Python virtual environment via Uvicorn. Docker provides the
database environment; it does not containerize the application.

[Render](../render.yaml) uses its native Python runtime for FastAPI and managed
PostgreSQL for production data. Alembic must succeed before Uvicorn starts.
Gemini remains an external API. The documented deployment selects
`gemini-3.1-flash-lite` through `GEMINI_MODEL`.

The [deployment record](../README.md#verified-deployment-and-production-api-examples)
reports authenticated POST/GET persistence and HTTP 200 health checks. One normal
listing produced `pass`; lunch versus all meals produced pending `review`; identical
lunch inclusion/exclusion produced `fail`. These are recorded smoke checks, not
production traffic or reliability measurements; they were not externally rechecked
for this document.

## 5. Evaluation Results

Evaluations were conducted on November 10 and 12, 2026, using **gemini-3.1-flash-lite**.
The Semantic report records the model explicitly; the Extraction JSON identifies
it through its filename, without model metadata. Both fixtures contain 14 cases.

### Extraction + Deterministic QA

Source: [Extraction results](../evals/results/extraction_gemini-3.1-flash-lite.json).
The runner compares expected field values and deterministic issue-code sets.

| Metric | Result |
| --- | --- |
| Total / passed / successfully evaluated | 14 / 10 / 13 |
| Field accuracy | 95.69% (111/116) |
| Issue precision / recall / F1 | 100% / 84.62% / 91.67% |
| Provider errors | 1 |

### Semantic QA

Sources: [results](../evals/semantic_results.json) and
[report](../evals/semantic_report.json). The runner evaluates pair-level review
classification independently of the API budget; both uncertain and possible-conflict
outputs count as review predictions.

| Metric | Result |
| --- | --- |
| Total / successfully evaluated | 14 / 11 |
| TP / FP / FN / TN | 6 / 0 / 0 / 5 |
| Precision / recall / F1 | 100% / 100% / 100% |
| Provider errors / intentional exact-match skips | 2 / 1 |
| Coverage | 78.57% (11/14) |
| Provider error rate | 15.38% of 13 attempts; 14.29% of all 14 cases |

Quality metrics exclude provider failures. **Semantic F1 of 100% applies only to
11 successfully evaluated cases.** Extraction Issue F1 and Semantic F1 measure
different tasks and must not be compared as equivalent scores.

## 6. Failure Analysis

**Observed extraction mismatches.** The duration-conflict case selected 180 minutes
where the fixture expected 120; both duration mentions and the contradiction issue
were retained. Another case returned duplicate `[120, 120]` mentions instead of
`[120]`. Both failed field matching without changing expected issue codes.

**Observed injection failure.** Delimiter escape changed Barcelona to Rome,
invented a meeting point, and removed exclusions, causing two missed QA issues.
Structured JSON and instruction/data separation did not ensure source grounding.

**Observed provider failures.** One Extraction error and two Semantic errors reduced
coverage across two separate evaluation runs. These are three failed evaluation
cases, not three unique production incidents. The README reports HTTP 504 during
the Extraction run; its saved JSON retains only a generic error, not the status.

**Architectural budget risk.** Pairs are processed sequentially in source order,
without ranking. Earlier pairs can exhaust the budget before a potentially important
later conflict is classified. This risk follows from code inspection, not an
observed failure in the standalone Semantic evaluation.

## 7. Limitations & Next Steps

Small datasets, limited adversarial coverage, and single runs do not establish
generalization or repeatability. New analyses return `confidence: null`; no
calibrated confidence score exists. Next steps are to evaluate candidate ranking
for budget efficiency, broaden injection and grounding tests, and improve provider
reliability measurement and run metadata.

The system is deployed and functionally validated, but production-scale reliability, security hardening, and operational readiness have not been established.
There is no evidence of production-scale reliability, and bounded classifier attempts do not impose an overall request deadline.

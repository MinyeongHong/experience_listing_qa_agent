from datetime import datetime
from typing import Literal

from pydantic import BaseModel, StrictBool, TypeAdapter


class SemanticMetrics(BaseModel):
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float


class SemanticMismatch(BaseModel):
    id: str
    category: Literal["false_positive", "false_negative"]


class EvalRunMetadata(BaseModel):
    model: str
    prompt_version: str
    dataset_version: str
    mode: Literal["live", "replay"]
    timestamp: datetime


class SemanticEvalReport(BaseModel):
    total_cases: int | None
    attempted_cases: int
    evaluated_cases: int
    skipped_cases: int | None
    provider_errors: int

    evaluation_coverage: float | None
    provider_error_rate: float

    metadata: EvalRunMetadata | None

    metrics: SemanticMetrics
    mismatches: list[SemanticMismatch]


class SemanticEvalSuccess(BaseModel):
    id: str
    status: Literal["success"] = "success"
    expected_review: bool
    predicted_review: bool
    verdict: Literal["possible_conflict", "no_conflict", "uncertain"]
    reason: str


class SemanticEvalError(BaseModel):
    id: str
    status: Literal["error"] = "error"
    expected_review: bool
    error: str


SemanticEvalResult = SemanticEvalSuccess | SemanticEvalError


def parse_eval_results(
    raw_results: list[dict],
) -> list[SemanticEvalResult]:
    adapter = TypeAdapter(list[SemanticEvalResult])

    results = adapter.validate_python(raw_results)

    return results


class SemanticEvalCase(BaseModel):
    id: str
    included: str
    excluded: str
    expected_review: StrictBool


def parse_eval_cases(raw_cases: list[dict]) -> list[SemanticEvalCase]:
    adapter = TypeAdapter(list[SemanticEvalCase])

    cases: list[SemanticEvalCase] = adapter.validate_python(raw_cases)

    return cases

from typing import Literal

from pydantic import BaseModel, TypeAdapter


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

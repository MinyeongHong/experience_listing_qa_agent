from types import SimpleNamespace
from unittest.mock import Mock
from uuid import UUID

import httpx
import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from app import main
from app.db.session import get_db
from app.domain.models import Listing, SemanticDecision
from app.services.extraction import (
    ExtractionError,
    ExtractionTimeoutError,
    RateLimitError,
)
from app.services.review import SemanticClassificationError


@pytest.fixture
def semantic_api(monkeypatch, app_api_key):
    monkeypatch.delenv("SEMANTIC_MAX_LLM_CALLS", raising=False)
    listing = Listing(
        title="Tour",
        city="Rome",
        duration_minutes=60,
        price=25,
        meeting_point="Main gate",
        inclusions=["Local guide"],
        exclusions=["Meals"],
    )
    extractor = Mock()
    extractor.extract.return_value = listing
    classifier = Mock()
    classifier.classify.return_value = SemanticDecision(
        verdict="no_conflict", reason="Different services"
    )
    records = {}

    def save(session, raw_text, result):
        record = SimpleNamespace(
            id=UUID(int=1),
            extracted_fields=result.extracted_fields.model_dump(),
            issues=result.issues,
            review_candidates=result.review_candidates,
            verdict=result.verdict,
            review_required=result.review_required,
            reason=result.reason,
            confidence=result.confidence,
        )
        records[record.id] = record
        return record

    persistence = Mock(side_effect=save)
    monkeypatch.setattr(main, "save_analysis", persistence)
    monkeypatch.setattr(main, "get_analysis", lambda session, id_: records.get(id_))
    main.app.dependency_overrides[get_db] = lambda: Mock()
    main.app.dependency_overrides[main.get_listing_extractor] = lambda: extractor
    main.app.dependency_overrides[main.get_semantic_classifier] = lambda: classifier
    try:
        yield (
            TestClient(main.app, headers={"X-API-Key": app_api_key}),
            listing,
            extractor,
            classifier,
            persistence,
        )
    finally:
        main.app.dependency_overrides.clear()


def test_semantic_no_conflict_passes_and_post_matches_get(semantic_api, app_api_key):
    client, listing, _, classifier, persistence = semantic_api
    response = client.post("/analyses", json={"text": "Source listing"})
    assert response.status_code == 200
    data = response.json()
    assert app_api_key not in response.text
    assert data["verdict"] == "pass"
    assert data["review_required"] is False
    assert data["review_candidates"] == []
    assert data["confidence"] is None
    classifier.classify.assert_called_once_with("Local guide", "Meals")
    result = persistence.call_args.args[2]
    assert result.extracted_fields is listing
    assert client.get(f"/analyses/{data['id']}").json() == data
    classifier.classify.assert_called_once()


@pytest.mark.parametrize("verdict", ["possible_conflict", "uncertain"])
def test_semantic_pending_decision_is_saved_as_review(semantic_api, verdict):
    client, _, _, classifier, _ = semantic_api
    classifier.classify.return_value = SemanticDecision(
        verdict=verdict, reason="Scope overlaps"
    )
    response = client.post("/analyses", json={"text": "Source listing"})
    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] == "review" and data["review_required"] is True
    assert data["review_candidates"][0]["status"] == "pending"
    assert data["review_candidates"][0]["reason"] == "Scope overlaps"
    assert client.get(f"/analyses/{data['id']}").json() == data


def test_semantic_failure_preserves_listing_issues_and_earlier_decisions(semantic_api):
    client, listing, _, classifier, persistence = semantic_api
    listing.meeting_point = None
    listing.inclusions = ["Local guide", "Pickup"]
    classifier.classify.side_effect = [
        SemanticDecision(verdict="uncertain", reason="Scope unclear"),
        SemanticClassificationError("Provider unavailable"),
    ]
    response = client.post("/analyses", json={"text": "Source listing"})
    assert response.status_code == 200
    data = response.json()
    assert data["extracted_fields"] == listing.model_dump()
    assert [issue["code"] for issue in data["issues"]] == ["MISSING_MEETING_POINT"]
    assert [item["code"] for item in data["review_candidates"]] == [
        "POSSIBLE_INCLUSION_CONFLICT",
        "SEMANTIC_CLASSIFICATION_FAILED",
    ]
    assert all(item["status"] == "pending" for item in data["review_candidates"])
    assert data["verdict"] == "review" and data["review_required"] is True
    assert "SEMANTIC_CLASSIFICATION_FAILED" in data["reason"]
    persistence.assert_called_once()
    assert client.get(f"/analyses/{data['id']}").json() == data


def test_deterministic_failure_takes_precedence_over_semantic_failure(semantic_api):
    client, listing, _, classifier, _ = semantic_api
    listing.duration_mentions = [60, 90]
    classifier.classify.side_effect = SemanticClassificationError("Timeout")
    data = client.post("/analyses", json={"text": "Source listing"}).json()
    assert data["verdict"] == "fail"
    assert data["review_required"] is True
    assert data["review_candidates"][0]["code"] == "SEMANTIC_CLASSIFICATION_FAILED"


@pytest.mark.parametrize("configured,expected_calls", [(None, 5), ("2", 2), ("0", 0)])
def test_api_enforces_default_and_configured_budget(
    semantic_api, monkeypatch, configured, expected_calls
):
    client, listing, _, classifier, _ = semantic_api
    listing.inclusions = ["A", "B", "C"]
    listing.exclusions = ["D", "E"]
    if configured is not None:
        monkeypatch.setenv("SEMANTIC_MAX_LLM_CALLS", configured)
    response = client.post("/analyses", json={"text": "Source listing"})
    assert response.status_code == 200
    data = response.json()
    assert classifier.classify.call_count == expected_calls
    assert len(data["review_candidates"]) == 6 - expected_calls
    assert all(
        item["code"] == "SEMANTIC_BUDGET_EXHAUSTED"
        for item in data["review_candidates"]
    )
    assert data["verdict"] == "review" and data["review_required"] is True


@pytest.mark.parametrize(
    "error,status",
    [
        (ExtractionError("bad"), 502),
        (ExtractionTimeoutError("late"), 504),
        (RateLimitError("limit"), 503),
    ],
)
def test_extraction_failure_never_calls_semantic_or_persistence(
    semantic_api, error, status
):
    client, _, extractor, classifier, persistence = semantic_api
    extractor.extract.side_effect = error
    assert (
        client.post("/analyses", json={"text": "Source listing"}).status_code == status
    )
    classifier.classify.assert_not_called()
    persistence.assert_not_called()


def test_invalid_input_never_calls_either_model_or_persistence(semantic_api):
    client, _, extractor, classifier, persistence = semantic_api
    assert client.post("/analyses", json={"text": "  "}).status_code == 422
    extractor.extract.assert_not_called()
    classifier.classify.assert_not_called()
    persistence.assert_not_called()


@pytest.mark.parametrize("configured", ["-1", "invalid", "", "1.5"])
def test_invalid_server_budget_is_rejected(monkeypatch, configured):
    monkeypatch.setenv("SEMANTIC_MAX_LLM_CALLS", configured)
    with pytest.raises(ValueError, match="SEMANTIC_MAX_LLM_CALLS"):
        main.get_semantic_call_budget()


def test_semantic_dependency_reuses_existing_client(monkeypatch):
    client = Mock()
    request = Request(
        {
            "type": "http",
            "app": SimpleNamespace(state=SimpleNamespace(gemini_client=client)),
        }
    )
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    classifier = main.get_semantic_classifier(request)
    assert classifier.client is client
    assert classifier.model == "test-model"


@pytest.mark.parametrize(
    "error",
    [
        httpx.ConnectError("offline"),
        httpx.ReadTimeout("late"),
        RuntimeError("provider failure"),
    ],
)
def test_real_classifier_errors_become_pending_in_api(semantic_api, error):
    from app.integrations.gemini_classifier import GeminiSemanticClassifier

    client, listing, _, _, _ = semantic_api
    provider = Mock()
    provider.models.generate_content.side_effect = error
    main.app.dependency_overrides[main.get_semantic_classifier] = lambda: (
        GeminiSemanticClassifier(provider)
    )
    response = client.post("/analyses", json={"text": "Source listing"})
    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] == "review"
    assert data["extracted_fields"] == listing.model_dump()
    assert data["review_candidates"][0]["code"] == "SEMANTIC_CLASSIFICATION_FAILED"
    assert data["review_candidates"][0]["status"] == "pending"


@pytest.mark.parametrize(
    "response",
    [
        None,
        SimpleNamespace(),
        SimpleNamespace(parsed={"verdict": "no_conflict", "reason": "  "}),
    ],
)
def test_malformed_semantic_response_is_pending_in_api(semantic_api, response):
    from app.integrations.gemini_classifier import GeminiSemanticClassifier

    client, _, _, _, _ = semantic_api
    provider = Mock()
    provider.models.generate_content.return_value = response
    main.app.dependency_overrides[main.get_semantic_classifier] = lambda: (
        GeminiSemanticClassifier(provider)
    )
    result = client.post("/analyses", json={"text": "Source listing"})
    assert result.status_code == 200
    data = result.json()
    assert data["verdict"] == "review"
    assert data["review_candidates"][0]["code"] == "SEMANTIC_CLASSIFICATION_FAILED"


def test_nul_semantic_reason_is_replaced_with_safe_pending_failure(semantic_api):
    from app.integrations.gemini_classifier import GeminiSemanticClassifier

    client, listing, _, _, _ = semantic_api
    provider = Mock()
    provider.models.generate_content.return_value.parsed = {
        "verdict": "uncertain",
        "reason": "Scope unclear\x00",
    }
    main.app.dependency_overrides[main.get_semantic_classifier] = lambda: (
        GeminiSemanticClassifier(provider)
    )
    response = client.post("/analyses", json={"text": "Source listing"})
    assert response.status_code == 200
    data = response.json()
    assert data["extracted_fields"] == listing.model_dump()
    assert data["verdict"] == "review" and data["review_required"] is True
    assert data["review_candidates"][0]["code"] == "SEMANTIC_CLASSIFICATION_FAILED"
    assert data["review_candidates"][0]["status"] == "pending"
    assert "\x00" not in data["review_candidates"][0]["reason"]
    assert "\x00" not in data["reason"]


def test_unicode_exact_match_fails_without_semantic_call(semantic_api):
    client, listing, _, classifier, _ = semantic_api
    listing.inclusions = [" Straße tour "]
    listing.exclusions = ["STRASSE TOUR"]
    response = client.post("/analyses", json={"text": "Source listing"})
    assert response.status_code == 200
    data = response.json()
    assert data["verdict"] == "fail" and data["review_required"] is True
    assert [issue["code"] for issue in data["issues"]] == ["CONTRADICTORY_INCLUSION"]
    assert data["review_candidates"] == []
    classifier.classify.assert_not_called()

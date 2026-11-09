from unittest.mock import Mock

import httpx
import pytest
from google.genai import errors

from app.domain.models import Listing, SemanticDecision
from app.integrations.gemini_classifier import GeminiSemanticClassifier
from app.services.review import SemanticClassificationError, find_review_candidates


@pytest.mark.parametrize(
    "error", [TimeoutError("timeout"), httpx.ReadTimeout("timeout")]
)
def test_gemini_classifier_converts_timeout(error):
    client = Mock()
    client.models.generate_content.side_effect = error

    with pytest.raises(SemanticClassificationError, match="timed out") as exc_info:
        GeminiSemanticClassifier(client=client).classify("Lunch", "Food")

    assert exc_info.value.__cause__ is error


def test_timeout_creates_review_candidate_and_continues():
    client = Mock()
    client.models.generate_content.side_effect = [
        httpx.ReadTimeout("timeout"),
        Mock(
            parsed=SemanticDecision(verdict="no_conflict", reason="Different services")
        ),
    ]
    listing = Listing(title="Tour", inclusions=["Lunch", "Guide"], exclusions=["Food"])

    candidates = find_review_candidates(
        listing, GeminiSemanticClassifier(client=client)
    )

    assert len(candidates) == 1
    assert candidates[0].code == "SEMANTIC_CLASSIFICATION_FAILED"
    assert candidates[0].included_item == "Lunch"
    assert client.models.generate_content.call_count == 2


def test_gemini_classifier_returns_structured_decision():
    # Arrange
    mock_client = Mock()

    mock_client.models.generate_content.return_value.parsed = SemanticDecision(
        verdict="possible_conflict",
        reason="Both items refer to food tasting.",
    )

    classifier = GeminiSemanticClassifier(
        client=mock_client,
    )

    # Act
    decision = classifier.classify(
        included_item="Food tasting",
        excluded_item="Food tastings",
    )

    # Assert
    assert isinstance(decision, SemanticDecision)
    assert decision.verdict == "possible_conflict"
    assert decision.reason == "Both items refer to food tasting."

    mock_client.models.generate_content.assert_called_once()


def test_gemini_classifier_rejects_empty_response():
    mock_client = Mock()
    mock_client.models.generate_content.return_value.parsed = None

    classifier = GeminiSemanticClassifier(client=mock_client)

    with pytest.raises(
        SemanticClassificationError,
        match="no structured decision",
    ):
        classifier.classify(
            included_item="Food tasting",
            excluded_item="Food tastings",
        )


def test_gemini_classifier_rejects_invalid_decision():
    mock_client = Mock()

    mock_client.models.generate_content.return_value.parsed = {
        "verdict": "definitely_safe",
        "reason": "No problem.",
    }

    classifier = GeminiSemanticClassifier(client=mock_client)

    with pytest.raises(
        SemanticClassificationError,
        match="invalid decision",
    ):
        classifier.classify(
            included_item="Food tasting",
            excluded_item="Food tastings",
        )


def test_gemini_classifier_converts_api_error():
    mock_client = Mock()

    mock_client.models.generate_content.side_effect = errors.ClientError(
        429,
        {"error": {"message": "Rate limit exceeded"}},
    )

    classifier = GeminiSemanticClassifier(client=mock_client)

    with pytest.raises(
        SemanticClassificationError,
        match="API request failed",
    ):
        classifier.classify(
            included_item="Food tasting",
            excluded_item="Food tastings",
        )


@pytest.mark.parametrize(
    "error",
    [
        httpx.ConnectError("offline"),
        ConnectionError("offline"),
        RuntimeError("provider failure"),
    ],
)
def test_gemini_classifier_wraps_connection_and_provider_failures(error):
    client = Mock()
    client.models.generate_content.side_effect = error
    with pytest.raises(SemanticClassificationError) as exc_info:
        GeminiSemanticClassifier(client).classify("Lunch", "Meals")
    assert exc_info.value.__cause__ is error


@pytest.mark.parametrize("response", [None, object(), {"parsed": "bad"}])
def test_gemini_classifier_rejects_malformed_response(response):
    client = Mock()
    client.models.generate_content.return_value = response
    with pytest.raises(SemanticClassificationError):
        GeminiSemanticClassifier(client).classify("Lunch", "Meals")


def test_gemini_classifier_converts_nul_reason_to_classification_error():
    client = Mock()
    client.models.generate_content.return_value.parsed = {
        "verdict": "uncertain",
        "reason": "Scope unclear\x00",
    }
    with pytest.raises(SemanticClassificationError, match="invalid decision"):
        GeminiSemanticClassifier(client).classify("Lunch", "Meals")

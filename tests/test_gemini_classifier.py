from unittest.mock import Mock

import pytest
from google.genai import errors

from app.domain.models import SemanticDecision
from app.integrations.gemini_classifier import GeminiSemanticClassifier
from app.services.review import SemanticClassificationError


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

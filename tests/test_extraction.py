from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors

from app.domain.models import Listing
from app.services.extraction import (
    ExtractionError,
    ExtractionTimeoutError,
    ListingExtractor,
    RateLimitError,
)


def test_extract_listing():
    fake_listing = Listing(
        title="Barcelona Food Walking Tour",
        city="Barcelona",
        duration_minutes=120,
        price=35,
        meeting_point=None,
        inclusions=["Local guide", "Lunch"],
        exclusions=["Lunch", "Drinks"],
    )

    def fake_generate_content(*args, **kwargs):
        assert kwargs["model"] == "test-model"

        return SimpleNamespace(parsed=fake_listing)

    fake_client = SimpleNamespace(
        models=SimpleNamespace(
            generate_content=fake_generate_content,
        )
    )

    extractor = ListingExtractor(
        client=fake_client,
        model="test-model",
    )

    result = extractor.extract("some listing text")

    assert result == fake_listing


def test_extract_listing_raises_error_when_provider_fails():
    def fake_generate_content(*args, **kwargs):
        raise RuntimeError("Gemini API failed")

    fake_client = SimpleNamespace(
        models=SimpleNamespace(
            generate_content=fake_generate_content,
        )
    )

    extractor = ListingExtractor(
        client=fake_client,
        model="test-model",
    )

    with pytest.raises(ExtractionError):
        extractor.extract("some listing text")


def test_extract_listing_raises_error_when_parsed_response_is_none():
    def fake_generate_content(*args, **kwargs):
        return SimpleNamespace(parsed=None)

    fake_client = SimpleNamespace(
        models=SimpleNamespace(
            generate_content=fake_generate_content,
        )
    )

    extractor = ListingExtractor(
        client=fake_client,
        model="test-model",
    )

    with pytest.raises(ExtractionError):
        extractor.extract("some listing text")


def test_extract_listing_raises_rate_limit_error_on_429():
    def fake_generate_content(*args, **kwargs):
        raise errors.ClientError(
            429,
            {"message": "Rate limit exceeded"},
            None,
        )

    fake_client = SimpleNamespace(
        models=SimpleNamespace(
            generate_content=fake_generate_content,
        )
    )

    extractor = ListingExtractor(
        client=fake_client,
        model="test-model",
    )
    with pytest.raises(RateLimitError):
        extractor.extract("some listing text")


def test_extract_listing_raises_timeout_error():
    def fake_generate_content(*args, **kwargs):
        raise TimeoutError("Request timed out")

    fake_client = SimpleNamespace(
        models=SimpleNamespace(
            generate_content=fake_generate_content,
        )
    )

    extractor = ListingExtractor(
        client=fake_client,
        model="test-model",
    )

    with pytest.raises(ExtractionTimeoutError):
        extractor.extract("some listing text")


def test_extract_listing_raises_timeout_error_on_httpx_timeout():
    def fake_generate_content(*args, **kwargs):
        raise httpx.ReadTimeout("Gemini request timed out")

    fake_client = SimpleNamespace(
        models=SimpleNamespace(
            generate_content=fake_generate_content,
        )
    )

    extractor = ListingExtractor(
        client=fake_client,
        model="test-model",
    )

    with pytest.raises(ExtractionTimeoutError):
        extractor.extract("some listing text")

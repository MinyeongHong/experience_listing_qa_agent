from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors

from app.services import extraction
from app.services.extraction import (
    ExtractionError,
    ExtractionTimeoutError,
    RateLimitError,
)


def test_extract_listing_raises_error_when_provider_fails(monkeypatch):
    def fake_generate_content(*args, **kwargs):
        raise RuntimeError("Gemini API failed")

    monkeypatch.setattr(
        extraction.client.models,
        "generate_content",
        fake_generate_content,
    )

    with pytest.raises(ExtractionError):
        extraction.extract_listing("some listing text")


def test_extract_listing_raises_error_when_parsed_response_is_none(monkeypatch):
    def fake_generate_content(*args, **kwargs):
        return SimpleNamespace(parsed=None)

    monkeypatch.setattr(
        extraction.client.models,
        "generate_content",
        fake_generate_content,
    )

    with pytest.raises(ExtractionError):
        extraction.extract_listing("some listing text")


def test_extract_listing_raises_rate_limit_error_on_429(monkeypatch):
    def fake_generate_content(*args, **kwargs):
        raise errors.ClientError(
            429,
            {"message": "Rate limit exceeded"},
            None,
        )

    monkeypatch.setattr(
        extraction.client.models,
        "generate_content",
        fake_generate_content,
    )

    with pytest.raises(RateLimitError):
        extraction.extract_listing("some listing text")


def test_extract_listing_raises_timeout_error(monkeypatch):
    def fake_generate_content(*args, **kwargs):
        raise TimeoutError("Request timed out")

    monkeypatch.setattr(
        extraction.client.models,
        "generate_content",
        fake_generate_content,
    )

    with pytest.raises(ExtractionTimeoutError):
        extraction.extract_listing("some listing text")


def test_extract_listing_raises_timeout_error_on_httpx_timeout(monkeypatch):
    def fake_generate_content(*args, **kwargs):
        raise httpx.ReadTimeout("Gemini request timed out")

    monkeypatch.setattr(
        extraction.client.models,
        "generate_content",
        fake_generate_content,
    )

    with pytest.raises(ExtractionTimeoutError):
        extraction.extract_listing("some listing text")

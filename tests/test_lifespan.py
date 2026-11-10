from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.main import app


def test_gemini_client_lifecycle(monkeypatch, app_api_key):
    calls = []

    fake_client = SimpleNamespace(
        close=lambda: calls.append("closed"),
    )

    def fake_create_gemini_client():
        calls.append("created")
        return fake_client

    monkeypatch.setattr(
        "app.main.create_gemini_client",
        fake_create_gemini_client,
    )

    with TestClient(app):
        assert calls == ["created"]
        assert app.state.gemini_client is fake_client

    assert calls == ["created", "closed"]


def test_invalid_semantic_budget_closes_client_and_rejects_startup(
    monkeypatch, app_api_key
):
    calls = []
    fake_client = SimpleNamespace(close=lambda: calls.append("closed"))
    monkeypatch.setattr("app.main.create_gemini_client", lambda: fake_client)
    monkeypatch.setenv("SEMANTIC_MAX_LLM_CALLS", "-1")
    with pytest.raises(ValueError, match="SEMANTIC_MAX_LLM_CALLS"), TestClient(app):
        pytest.fail("Invalid server budget must prevent startup")
    assert calls == ["closed"]

from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.main import app


def test_gemini_client_lifecycle(monkeypatch):
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

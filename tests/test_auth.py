from unittest.mock import Mock
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from app import main


@pytest.mark.parametrize("method", ["POST", "GET"])
@pytest.mark.parametrize("supplied", [None, "", "wrong-key", "wrong-\xff-key"])
def test_unauthorized_requests_stop_before_db_and_model_dependencies(
    monkeypatch, caplog, method, supplied
):
    monkeypatch.setenv("APP_API_KEY", "private-test-key")
    monkeypatch.setattr(main, "create_gemini_client", lambda: Mock())
    operations = [
        Mock(side_effect=AssertionError("Unauthorized operation executed"))
        for _ in range(5)
    ]
    db, extractor, classifier, save, read = operations
    main.app.dependency_overrides[main.get_db] = db
    main.app.dependency_overrides[main.get_listing_extractor] = extractor
    main.app.dependency_overrides[main.get_semantic_classifier] = classifier
    monkeypatch.setattr(main, "save_analysis", save)
    monkeypatch.setattr(main, "get_analysis", read)
    headers = {} if supplied is None else {"X-API-Key": supplied}
    # HTTP header bytes may be latin-1; malformed/non-ASCII keys must still be 401.
    if supplied and "\xff" in supplied:
        headers = {b"X-API-Key": supplied.encode("latin-1")}
    try:
        with TestClient(main.app, raise_server_exceptions=False) as client:
            response = client.request(
                method,
                "/analyses" if method == "POST" else f"/analyses/{UUID(int=1)}",
                headers=headers,
                **({"json": {"text": "Tour"}} if method == "POST" else {}),
            )
        assert response.status_code == 401
        assert "private-test-key" not in response.text
        assert "private-test-key" not in caplog.text
        if supplied:
            assert supplied not in response.text
            assert supplied not in caplog.text
        for operation in operations:
            operation.assert_not_called()
    finally:
        main.app.dependency_overrides.clear()


@pytest.mark.parametrize("configured", [None, "", " \t "])
def test_missing_or_empty_api_key_prevents_startup(monkeypatch, configured):
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    if configured is None:
        monkeypatch.delenv("APP_API_KEY", raising=False)
    else:
        monkeypatch.setenv("APP_API_KEY", configured)
    create_client = Mock()
    monkeypatch.setattr(main, "create_gemini_client", create_client)
    with pytest.raises(RuntimeError, match="APP_API_KEY"), TestClient(main.app):
        pytest.fail("Missing API key must prevent application startup")
    create_client.assert_not_called()


def test_authentication_uses_compare_digest(monkeypatch):
    from app import security

    monkeypatch.setenv("APP_API_KEY", "private-test-key")
    compare = Mock(wraps=security.secrets.compare_digest)
    monkeypatch.setattr(security.secrets, "compare_digest", compare)
    security.require_api_key("private-test-key")
    compare.assert_called_once()

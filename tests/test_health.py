from unittest.mock import MagicMock, Mock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import NullPool

from app import main


@pytest.mark.parametrize("available,status", [(True, 200), (False, 503)])
def test_health_is_public_and_does_not_use_analysis_dependencies(
    monkeypatch, available, status
):
    check = Mock(return_value=available)
    monkeypatch.setattr(main, "check_database_health", check, raising=False)
    operations = [
        Mock(side_effect=AssertionError("Analysis operation executed"))
        for _ in range(5)
    ]
    db, extractor, classifier, save, read = operations
    main.app.dependency_overrides[main.get_db] = db
    main.app.dependency_overrides[main.get_listing_extractor] = extractor
    main.app.dependency_overrides[main.get_semantic_classifier] = classifier
    monkeypatch.setattr(main, "save_analysis", save)
    monkeypatch.setattr(main, "get_analysis", read)
    try:
        # No lifespan or credentials required to exercise this public route.
        response = TestClient(main.app).get("/health")
        assert response.status_code == status
        if available:
            assert response.json() == {"status": "ok"}
        check.assert_called_once_with()
        for operation in operations:
            operation.assert_not_called()
    finally:
        main.app.dependency_overrides.clear()


def test_health_uses_select_only_and_bounded_connection_settings(monkeypatch):
    from app.db import health

    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://user:secret@db/test")
    engine = MagicMock()
    connection = engine.connect.return_value.__enter__.return_value
    connection.execute.return_value.scalar_one.return_value = 1
    create_engine = Mock(return_value=engine)
    monkeypatch.setattr(health, "create_engine", create_engine)

    assert health.check_database_health() is True

    options = create_engine.call_args.kwargs
    assert options["poolclass"] is NullPool
    assert options["connect_args"]["connect_timeout"] == 2
    assert "statement_timeout=2000" in options["connect_args"]["options"]
    connection.execute.assert_called_once()
    assert str(connection.execute.call_args.args[0]) == "SELECT 1"
    connection.commit.assert_not_called()
    engine.dispose.assert_called_once()


def test_health_connection_error_is_safely_reported(monkeypatch):
    from app.db import health

    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://user:secret@db/test")
    engine = Mock()
    engine.connect.side_effect = OperationalError(
        "SELECT 1", {}, Exception("secret database error")
    )
    monkeypatch.setattr(health, "create_engine", Mock(return_value=engine))
    response = TestClient(main.app).get("/health")
    assert response.status_code == 503
    assert "secret" not in response.text
    engine.dispose.assert_called_once()


def test_health_query_error_is_safely_reported(monkeypatch):
    from app.db import health

    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://user:secret@db/test")
    engine = MagicMock()
    connection = engine.connect.return_value.__enter__.return_value
    connection.execute.side_effect = OperationalError(
        "SELECT 1", {}, Exception("secret query error")
    )
    monkeypatch.setattr(health, "create_engine", Mock(return_value=engine))
    response = TestClient(main.app).get("/health")
    assert response.status_code == 503
    assert "secret" not in response.text
    connection.commit.assert_not_called()
    engine.dispose.assert_called_once()


def test_health_missing_database_configuration_is_503(monkeypatch):
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert TestClient(main.app).get("/health").status_code == 503


@pytest.mark.integration
@pytest.mark.parametrize("slow_query", [False, True])
def test_postgresql_health_and_statement_timeout(test_engine, monkeypatch, slow_query):
    import time

    from sqlalchemy import text

    from app.db import health

    monkeypatch.setenv(
        "DATABASE_URL", test_engine.url.render_as_string(hide_password=False)
    )
    if slow_query:
        monkeypatch.setattr(health, "text", lambda _: text("SELECT pg_sleep(10)"))
    started = time.monotonic()
    response = TestClient(main.app).get("/health")
    assert response.status_code == (503 if slow_query else 200)
    assert time.monotonic() - started < 5

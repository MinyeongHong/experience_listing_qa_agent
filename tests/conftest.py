import os
from collections.abc import Generator
from unittest.mock import Mock

import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from app.db.session import get_db
from app.main import app


@pytest.fixture(scope="session")
def test_engine():
    load_dotenv()
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL is required for integration tests.")

    # 실수로 개발 DB에 테스트 데이터를 저장하지 않도록 방어
    if make_url(database_url).database != "listingqa_test":
        raise RuntimeError("Integration tests must use listingqa_test database.")

    engine = create_engine(database_url, pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            if connection.scalar(text("SELECT current_database()")) != "listingqa_test":
                raise RuntimeError("Integration tests must use listingqa_test database.")
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def db_session(clean_test_db, test_engine) -> Generator[Session, None, None]:
    with sessionmaker(bind=test_engine, autoflush=False)() as session:
        yield session


@pytest.fixture
def integration_client(
    db_session: Session,
    monkeypatch,
) -> Generator[TestClient, None, None]:
    monkeypatch.setattr("app.main.create_gemini_client", lambda: Mock())
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.pop(get_db, None)


@pytest.fixture
def clean_test_db(test_engine) -> Generator[None, None, None]:
    # 테스트 시작 전 기존 데이터 정리
    with test_engine.begin() as connection:
        connection.execute(text("TRUNCATE TABLE issues, analyses CASCADE"))

    try:
        yield
    finally:
        # 테스트 종료 후 데이터 정리
        with test_engine.begin() as connection:
            connection.execute(text("TRUNCATE TABLE issues, analyses CASCADE"))


@pytest.fixture
def verification_session(clean_test_db, test_engine):
    with sessionmaker(bind=test_engine, autoflush=False)() as session:
        yield session

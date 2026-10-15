import os
from collections.abc import Generator

import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.db.session import get_db
from app.main import app

load_dotenv()

TEST_DATABASE_URL = os.environ["TEST_DATABASE_URL"]

# 실수로 개발 DB에 테스트 데이터를 저장하지 않도록 방어
if TEST_DATABASE_URL.rsplit("/", 1)[-1] != "listingqa_test":
    raise RuntimeError("Integration tests must use listingqa_test database.")

test_engine = create_engine(TEST_DATABASE_URL, pool_pre_ping=True)

TestSessionLocal = sessionmaker(
    bind=test_engine,
    autoflush=False,
)


@pytest.fixture
def db_session(clean_test_db) -> Generator[Session, None, None]:
    with TestSessionLocal() as session:
        yield session


@pytest.fixture
def integration_client(
    db_session: Session,
) -> Generator[TestClient, None, None]:
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.pop(get_db, None)


@pytest.fixture
def clean_test_db() -> Generator[None, None, None]:
    # 테스트 시작 전 기존 데이터 정리
    with test_engine.begin() as connection:
        connection.execute(text("TRUNCATE TABLE issues, analyses CASCADE"))

    yield

    # 테스트 종료 후 데이터 정리
    with test_engine.begin() as connection:
        connection.execute(text("TRUNCATE TABLE issues, analyses CASCADE"))


@pytest.fixture
def verification_session(clean_test_db):
    with TestSessionLocal() as session:
        yield session

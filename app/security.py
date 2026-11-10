import os
import secrets
from typing import Annotated

from dotenv import load_dotenv
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_app_api_key() -> str:
    load_dotenv()
    key = os.getenv("APP_API_KEY")
    if not key or not key.strip():
        raise RuntimeError("APP_API_KEY must be set and non-empty")
    return key


def require_api_key(api_key: Annotated[str | None, Security(api_key_header)]) -> None:
    expected = get_app_api_key()
    if api_key is None or not secrets.compare_digest(
        api_key.encode("utf-8"), expected.encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

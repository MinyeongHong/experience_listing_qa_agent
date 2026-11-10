import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import NullPool


def check_database_health() -> bool:
    engine = None
    try:
        load_dotenv()
        # Do not wait for the application pool or inherit its query timeout.
        engine = create_engine(
            os.environ["DATABASE_URL"],
            poolclass=NullPool,
            connect_args={
                "connect_timeout": 2,
                "options": "-c statement_timeout=2000",
            },
        )
        with engine.connect() as connection:
            return connection.execute(text("SELECT 1")).scalar_one() == 1
    except (SQLAlchemyError, KeyError, ValueError, OSError):
        return False
    finally:
        if engine is not None:
            engine.dispose()

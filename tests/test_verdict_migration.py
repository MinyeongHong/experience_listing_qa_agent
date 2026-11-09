from io import StringIO
from uuid import uuid4

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from alembic.script import ScriptDirectory
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db import models

OLD_REVISION = "ff726829f634"
NEW_REVISION = "a91c3e7d2b40"


def new_revision():
    scripts = ScriptDirectory("migrations")
    revision = scripts.get_revision(NEW_REVISION)
    assert revision.down_revision == OLD_REVISION
    return revision.module


def migration_sql(direction="upgrade"):
    output = StringIO()
    context = MigrationContext.configure(
        dialect_name="postgresql", opts={"as_sql": True, "output_buffer": output}
    )
    with Operations.context(context):
        getattr(new_revision(), direction)()
    return output.getvalue()


def test_upgrade_adds_schema_and_downgrade_restores_old_schema():
    upgrade = migration_sql()
    assert "CREATE TABLE review_candidates" in upgrade
    assert "ALTER COLUMN confidence DROP NOT NULL" in upgrade
    for name in ("verdict", "review_required", "reason"):
        assert f"ADD COLUMN {name}" in upgrade
    assert "FOREIGN KEY(analysis_id) REFERENCES analyses (id)" in upgrade
    downgrade = migration_sql("downgrade")
    assert "DROP TABLE review_candidates" in downgrade
    assert "ALTER COLUMN confidence SET NOT NULL" in downgrade


@pytest.mark.parametrize(
    ("codes", "verdict"),
    [
        ([], "review"),
        (["MISSING_CITY"], "review"),
        (["ZERO_PRICE"], "review"),
        (["POSSIBLE_INCLUSION_CONFLICT"], "review"),
        (["CONTRADICTORY_INCLUSION"], "fail"),
        (["DURATION_CONTRADICTION"], "fail"),
        (["MISSING_CITY", "DURATION_CONTRADICTION"], "fail"),
    ],
)
def test_legacy_backfill_uses_only_deterministic_contradictions(codes, verdict):
    # Execute the emitted backfill SQL, rather than a copy of its decision logic.
    statements = [part.strip() for part in migration_sql().split(";")]
    updates = [
        part for part in statements if part.upper().startswith("UPDATE ANALYSES")
    ]
    assert updates
    engine = sa.create_engine("sqlite://")
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(
                "CREATE TABLE analyses (id INTEGER PRIMARY KEY, confidence FLOAT, "
                "verdict TEXT, review_required BOOLEAN, reason TEXT)"
            )
            connection.exec_driver_sql(
                "CREATE TABLE issues (analysis_id INTEGER, code TEXT)"
            )
            connection.exec_driver_sql(
                "INSERT INTO analyses (id, confidence) VALUES (1, 1.0), (2, 0.5)"
            )
            for code in codes:
                connection.execute(
                    sa.text("INSERT INTO issues VALUES (1, :code)"), {"code": code}
                )
            for statement in updates:
                connection.execute(sa.text(statement))
            records = (
                connection.execute(sa.text("SELECT * FROM analyses ORDER BY id"))
                .mappings()
                .all()
            )
            assert records[0]["verdict"] == verdict
            assert (
                records[1]["verdict"] == "review"
            )  # Do not leak another analysis's issues.
            for record in records:
                assert record["review_required"]
                assert "Semantic QA was not performed" in record["reason"]
            assert [row["confidence"] for row in records] == [1.0, 0.5]
    finally:
        engine.dispose()


@pytest.mark.integration
def test_postgresql_migration_round_trip_and_orm(test_engine):
    # Transactional private schema: never upgrade/downgrade or truncate existing tables.
    schema = f"migration_test_{uuid4().hex}"
    with test_engine.connect() as connection:
        transaction = connection.begin()
        try:
            connection.execute(sa.schema.CreateSchema(schema))
            connection.exec_driver_sql(f'SET LOCAL search_path TO "{schema}"')
            context = MigrationContext.configure(connection)
            old = ScriptDirectory("migrations").get_revision(OLD_REVISION).module
            migration = new_revision()
            with Operations.context(context):
                old.upgrade()

            legacy_ids = [uuid4(), uuid4()]
            for analysis_id in legacy_ids:
                connection.execute(
                    sa.text(
                        "INSERT INTO analyses (id, raw_text, extracted_fields, confidence, status, created_at) "
                        "VALUES (:id, 'Legacy tour', '{\"title\": \"Legacy tour\"}', 1.0, 'completed', now())"
                    ),
                    {"id": analysis_id},
                )
            connection.execute(
                sa.text(
                    "INSERT INTO issues (id, analysis_id, code, field, message) "
                    "VALUES (:id, :analysis_id, 'DURATION_CONTRADICTION', 'duration_minutes', 'Conflicting durations')"
                ),
                {"id": uuid4(), "analysis_id": legacy_ids[0]},
            )
            with Operations.context(context):
                migration.upgrade()

            with Session(
                bind=connection, join_transaction_mode="create_savepoint"
            ) as session:
                for analysis_id, verdict in zip(
                    legacy_ids, ["fail", "review"], strict=True
                ):
                    record = session.get(models.AnalysisRecord, analysis_id)
                    assert record.verdict == verdict
                    assert record.review_required is True
                    assert "Semantic QA was not performed" in record.reason
                    assert record.confidence == 1.0
                    assert record.review_candidates == []

                record = models.AnalysisRecord(
                    raw_text="New tour",
                    extracted_fields={"title": "New tour"},
                    confidence=None,
                    review_candidates=[
                        models.ReviewCandidateRecord(
                            position=0,
                            code="SEMANTIC_CLASSIFICATION_FAILED",
                            field="inclusions",
                            included_item="Lunch",
                            excluded_item="Meals",
                            reason="Timeout",
                        )
                    ],
                )
                session.add(record)
                session.commit()
                session.refresh(record)
                assert record.confidence is None
                assert record.verdict == "review"
                assert record.review_candidates[0].status == "pending"
                assert record.review_candidates[0].analysis_id == record.id
                record_id = record.id

            # Downgrade must fail without discarding data or inventing confidence.
            with (
                pytest.raises(IntegrityError),
                connection.begin_nested(),
                Operations.context(context),
            ):
                migration.downgrade()
            assert connection.scalar(sa.text("SELECT COUNT(*) FROM analyses")) == 3

            with Session(
                bind=connection, join_transaction_mode="create_savepoint"
            ) as session:
                session.delete(session.get(models.AnalysisRecord, record_id))
                session.commit()
            with Operations.context(context):
                migration.downgrade()
            assert "review_candidates" not in sa.inspect(connection).get_table_names()
            assert "verdict" not in {
                column["name"]
                for column in sa.inspect(connection).get_columns("analyses")
            }
            assert connection.scalar(sa.text("SELECT COUNT(*) FROM analyses")) == 2
        finally:
            transaction.rollback()

"""Add analysis verdict and separately stored semantic review candidates.

Revision ID: a91c3e7d2b40
Revises: ff726829f634
"""

import sqlalchemy as sa
from alembic import op

revision: str = "a91c3e7d2b40"
down_revision: str = "ff726829f634"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "analyses",
        sa.Column("verdict", sa.String(10), nullable=False, server_default="review"),
    )
    op.add_column(
        "analyses",
        sa.Column(
            "review_required", sa.Boolean(), nullable=False, server_default=sa.true()
        ),
    )
    op.add_column(
        "analyses",
        sa.Column(
            "reason",
            sa.Text(),
            nullable=False,
            server_default="Semantic QA has not been performed.",
        ),
    )
    op.alter_column("analyses", "confidence", existing_type=sa.Float(), nullable=True)

    # Use persisted deterministic issues only; historical absence of semantic
    # candidates is not evidence that semantic checks passed.
    op.execute(
        sa.text("""
        UPDATE analyses
        SET verdict = CASE WHEN EXISTS (
            SELECT 1 FROM issues
            WHERE issues.analysis_id = analyses.id
              AND issues.code IN ('CONTRADICTORY_INCLUSION', 'DURATION_CONTRADICTION')
        ) THEN 'fail' ELSE 'review' END,
        review_required = true,
        reason = CASE WHEN EXISTS (
            SELECT 1 FROM issues
            WHERE issues.analysis_id = analyses.id
              AND issues.code IN ('CONTRADICTORY_INCLUSION', 'DURATION_CONTRADICTION')
        ) THEN 'Deterministic contradiction detected. Semantic QA was not performed for this legacy analysis.'
          ELSE 'Legacy analysis requires review. Semantic QA was not performed for this legacy analysis.' END
    """)
    )

    op.create_table(
        "review_candidates",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("analysis_id", sa.UUID(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("code", sa.String(100), nullable=False),
        sa.Column("field", sa.String(100), nullable=False),
        sa.Column("included_item", sa.Text(), nullable=False),
        sa.Column("excluded_item", sa.Text(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("status", sa.String(30), nullable=False, server_default="pending"),
        sa.ForeignKeyConstraint(["analysis_id"], ["analyses.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_review_candidates_analysis_id", "review_candidates", ["analysis_id"]
    )


def downgrade() -> None:
    # PostgreSQL rejects this if null confidences exist: do not fabricate scores
    # or delete analyses to satisfy the old NOT NULL constraint.
    op.alter_column("analyses", "confidence", existing_type=sa.Float(), nullable=False)
    op.drop_index("ix_review_candidates_analysis_id", table_name="review_candidates")
    op.drop_table("review_candidates")
    op.drop_column("analyses", "reason")
    op.drop_column("analyses", "review_required")
    op.drop_column("analyses", "verdict")

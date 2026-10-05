"""Create durable application persistence tables.

Revision ID: 0001_initial_persistence
Revises:
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0001_initial_persistence"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("provider", sa.String(length=50), nullable=False),
        sa.Column("provider_subject", sa.String(length=255), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=True),
        sa.Column("display_name", sa.String(length=255), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_users"),
        sa.UniqueConstraint(
            "provider",
            "provider_subject",
            name="uq_users_provider_subject",
        ),
    )
    op.create_table(
        "sessions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name="fk_sessions_user_id_users", ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_sessions"),
    )
    op.create_index("ix_sessions_user_id", "sessions", ["user_id"], unique=False)
    op.create_table(
        "messages",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.Uuid(), nullable=False),
        sa.Column("role", sa.String(length=20), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "role IN ('system', 'user', 'assistant', 'tool')",
            name="ck_messages_valid_role",
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["sessions.id"],
            name="fk_messages_session_id_sessions",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_messages"),
    )
    op.create_index(
        "ix_messages_session_id_id", "messages", ["session_id", "id"], unique=False
    )
    op.create_table(
        "papers",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column(
            "authors",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("published_year", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_papers"),
    )
    op.create_index(
        "uq_papers_source_url",
        "papers",
        ["source_url"],
        unique=True,
        postgresql_where=sa.text("source_url IS NOT NULL"),
    )
    op.create_table(
        "session_papers",
        sa.Column("session_id", sa.Uuid(), nullable=False),
        sa.Column("paper_id", sa.Uuid(), nullable=False),
        sa.Column(
            "indexed_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["paper_id"],
            ["papers.id"],
            name="fk_session_papers_paper_id_papers",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["sessions.id"],
            name="fk_session_papers_session_id_sessions",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("session_id", "paper_id", name="pk_session_papers"),
    )
    op.create_table(
        "paper_chunks",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.Uuid(), nullable=False),
        sa.Column("paper_id", sa.Uuid(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("total_chunks", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("chunk_index >= 0", name="ck_paper_chunks_nonnegative_chunk_index"),
        sa.CheckConstraint("total_chunks > 0", name="ck_paper_chunks_positive_total_chunks"),
        sa.ForeignKeyConstraint(
            ["paper_id"],
            ["papers.id"],
            name="fk_paper_chunks_paper_id_papers",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["sessions.id"],
            name="fk_paper_chunks_session_id_sessions",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_paper_chunks"),
        sa.UniqueConstraint(
            "session_id",
            "paper_id",
            "chunk_index",
            name="uq_paper_chunks_session_paper_index",
        ),
    )
    op.create_index(
        "ix_paper_chunks_session_id", "paper_chunks", ["session_id"], unique=False
    )
    op.create_table(
        "feedback",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("session_id", sa.Uuid(), nullable=False),
        sa.Column("message_id", sa.BigInteger(), nullable=True),
        sa.Column("rating", sa.SmallInteger(), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("rating IN (-1, 1)", name="ck_feedback_valid_rating"),
        sa.ForeignKeyConstraint(
            ["message_id"],
            ["messages.id"],
            name="fk_feedback_message_id_messages",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["sessions.id"],
            name="fk_feedback_session_id_sessions",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_feedback"),
    )
    op.create_index("ix_feedback_session_id", "feedback", ["session_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_feedback_session_id", table_name="feedback")
    op.drop_table("feedback")
    op.drop_index("ix_paper_chunks_session_id", table_name="paper_chunks")
    op.drop_table("paper_chunks")
    op.drop_table("session_papers")
    op.drop_index("uq_papers_source_url", table_name="papers")
    op.drop_table("papers")
    op.drop_index("ix_messages_session_id_id", table_name="messages")
    op.drop_table("messages")
    op.drop_index("ix_sessions_user_id", table_name="sessions")
    op.drop_table("sessions")
    op.drop_table("users")

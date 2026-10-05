"""SQLAlchemy Core schema shared by repositories and Alembic."""

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    SmallInteger,
    String,
    Table,
    Text,
    UniqueConstraint,
    Uuid,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB


NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=NAMING_CONVENTION)

users = Table(
    "users",
    metadata,
    Column("id", Uuid(as_uuid=True), primary_key=True),
    Column("provider", String(50), nullable=False),
    Column("provider_subject", String(255), nullable=False),
    Column("email", String(320), nullable=True),
    Column("display_name", String(255), nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    UniqueConstraint("provider", "provider_subject", name="uq_users_provider_subject"),
)

sessions = Table(
    "sessions",
    metadata,
    Column("id", Uuid(as_uuid=True), primary_key=True),
    Column(
        "user_id",
        Uuid(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
    ),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("expires_at", DateTime(timezone=True), nullable=True),
)
Index("ix_sessions_user_id", sessions.c.user_id)

messages = Table(
    "messages",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column(
        "session_id",
        Uuid(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("role", String(20), nullable=False),
    Column("content", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    CheckConstraint(
        "role IN ('system', 'user', 'assistant', 'tool')",
        name="valid_role",
    ),
)
Index("ix_messages_session_id_id", messages.c.session_id, messages.c.id)

papers = Table(
    "papers",
    metadata,
    Column("id", Uuid(as_uuid=True), primary_key=True),
    Column("title", String(500), nullable=False),
    Column("source_url", Text, nullable=True),
    Column("authors", JSONB, nullable=False, server_default=text("'[]'::jsonb")),
    Column("published_year", Integer, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)
Index(
    "uq_papers_source_url",
    papers.c.source_url,
    unique=True,
    postgresql_where=papers.c.source_url.is_not(None),
)

session_papers = Table(
    "session_papers",
    metadata,
    Column(
        "session_id",
        Uuid(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "paper_id",
        Uuid(as_uuid=True),
        ForeignKey("papers.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("indexed_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)

paper_chunks = Table(
    "paper_chunks",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column(
        "session_id",
        Uuid(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column(
        "paper_id",
        Uuid(as_uuid=True),
        ForeignKey("papers.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("chunk_index", Integer, nullable=False),
    Column("total_chunks", Integer, nullable=False),
    Column("content", Text, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    CheckConstraint("chunk_index >= 0", name="nonnegative_chunk_index"),
    CheckConstraint("total_chunks > 0", name="positive_total_chunks"),
    UniqueConstraint(
        "session_id",
        "paper_id",
        "chunk_index",
        name="uq_paper_chunks_session_paper_index",
    ),
)
Index("ix_paper_chunks_session_id", paper_chunks.c.session_id)

feedback = Table(
    "feedback",
    metadata,
    Column("id", Uuid(as_uuid=True), primary_key=True),
    Column(
        "session_id",
        Uuid(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column(
        "message_id",
        BigInteger,
        ForeignKey("messages.id", ondelete="SET NULL"),
        nullable=True,
    ),
    Column("rating", SmallInteger, nullable=False),
    Column("comment", Text, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    CheckConstraint("rating IN (-1, 1)", name="valid_rating"),
)
Index("ix_feedback_session_id", feedback.c.session_id)

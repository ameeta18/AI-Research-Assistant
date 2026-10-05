"""Small persistence boundary used by the UI, API, and retrieval tools."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Protocol, Sequence
from uuid import UUID, uuid4

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import create_engine, text

from src.config import get_settings
from src.observability import get_logger
from src.persistence.schema import (
    feedback,
    messages,
    paper_chunks,
    papers,
    session_papers,
    sessions,
)


logger = get_logger("persistence")
VALID_ROLES = frozenset({"system", "user", "assistant", "tool"})


class PersistenceError(RuntimeError):
    """Raised when durable storage cannot complete an operation."""


@dataclass(frozen=True)
class StoredMessage:
    id: int
    role: str
    content: str
    created_at: datetime


@dataclass(frozen=True)
class StoredChunk:
    paper_id: UUID
    title: str
    source_url: str
    chunk_index: int
    total_chunks: int
    content: str


class Persistence(Protocol):
    enabled: bool

    def ensure_session(self, session_id: str, user_id: str | None = None) -> None: ...

    def save_message(self, session_id: str, role: str, content: str) -> int | None: ...

    def list_messages(self, session_id: str) -> list[StoredMessage]: ...

    def save_paper_chunks(
        self,
        session_id: str,
        title: str,
        source_url: str,
        chunks: Sequence[str],
    ) -> UUID | None: ...

    def load_paper_chunks(self, session_id: str) -> list[StoredChunk]: ...

    def save_feedback(
        self,
        session_id: str,
        rating: int,
        message_id: int | None = None,
        comment: str | None = None,
    ) -> UUID | None: ...

    def healthcheck(self) -> bool: ...

    def close(self) -> None: ...


class NoOpPersistence:
    """Disabled persistence implementation that preserves lightweight local use."""

    enabled = False

    def ensure_session(self, session_id: str, user_id: str | None = None) -> None:
        _as_uuid(session_id, "session_id")

    def save_message(self, session_id: str, role: str, content: str) -> None:
        _validate_message(session_id, role, content)
        return None

    def list_messages(self, session_id: str) -> list[StoredMessage]:
        _as_uuid(session_id, "session_id")
        return []

    def save_paper_chunks(
        self,
        session_id: str,
        title: str,
        source_url: str,
        chunks: Sequence[str],
    ) -> None:
        _validate_chunks(session_id, title, chunks)
        return None

    def load_paper_chunks(self, session_id: str) -> list[StoredChunk]:
        _as_uuid(session_id, "session_id")
        return []

    def save_feedback(
        self,
        session_id: str,
        rating: int,
        message_id: int | None = None,
        comment: str | None = None,
    ) -> None:
        _as_uuid(session_id, "session_id")
        _validate_rating(rating)
        return None

    def healthcheck(self) -> bool:
        return True

    def close(self) -> None:
        return None


class PostgresPersistence:
    """Synchronous PostgreSQL repository for durable application records."""

    enabled = True

    def __init__(
        self,
        database_url: str,
        *,
        engine: Engine | None = None,
        session_ttl_minutes: int | None = None,
    ) -> None:
        settings = get_settings()
        self._session_ttl_minutes = (
            session_ttl_minutes
            if session_ttl_minutes is not None
            else settings.session_ttl_minutes
        )
        self._engine = engine or create_engine(
            database_url,
            pool_pre_ping=True,
            pool_size=settings.database_pool_size,
            max_overflow=settings.database_max_overflow,
            pool_timeout=settings.database_pool_timeout_seconds,
            connect_args={
                "connect_timeout": int(settings.database_connect_timeout_seconds),
                "application_name": "ai-researcher",
            },
        )

    def _session_values(self, session_id: UUID, user_id: UUID | None) -> dict:
        expires_at = datetime.now(timezone.utc) + timedelta(
            minutes=self._session_ttl_minutes
        )
        return {"id": session_id, "user_id": user_id, "expires_at": expires_at}

    def _ensure_session(self, connection, session_id: UUID, user_id: UUID | None) -> None:
        values = self._session_values(session_id, user_id)
        update_values = {
            "updated_at": datetime.now(timezone.utc),
            "expires_at": values["expires_at"],
        }
        if user_id is not None:
            update_values["user_id"] = user_id
        connection.execute(
            pg_insert(sessions)
            .values(**values)
            .on_conflict_do_update(
                index_elements=[sessions.c.id],
                set_=update_values,
            )
        )

    def ensure_session(self, session_id: str, user_id: str | None = None) -> None:
        session_uuid = _as_uuid(session_id, "session_id")
        user_uuid = _as_uuid(user_id, "user_id") if user_id else None
        try:
            with self._engine.begin() as connection:
                self._ensure_session(connection, session_uuid, user_uuid)
        except SQLAlchemyError as exc:
            self._raise("ensure_session", exc)

    def save_message(self, session_id: str, role: str, content: str) -> int:
        session_uuid = _validate_message(session_id, role, content)
        try:
            with self._engine.begin() as connection:
                self._ensure_session(connection, session_uuid, None)
                message_id = connection.execute(
                    messages.insert()
                    .values(session_id=session_uuid, role=role, content=content)
                    .returning(messages.c.id)
                ).scalar_one()
                return int(message_id)
        except SQLAlchemyError as exc:
            self._raise("save_message", exc)

    def list_messages(self, session_id: str) -> list[StoredMessage]:
        session_uuid = _as_uuid(session_id, "session_id")
        try:
            with self._engine.connect() as connection:
                rows = connection.execute(
                    select(
                        messages.c.id,
                        messages.c.role,
                        messages.c.content,
                        messages.c.created_at,
                    )
                    .where(messages.c.session_id == session_uuid)
                    .order_by(messages.c.id)
                ).mappings()
                return [StoredMessage(**dict(row)) for row in rows]
        except SQLAlchemyError as exc:
            self._raise("list_messages", exc)

    def save_paper_chunks(
        self,
        session_id: str,
        title: str,
        source_url: str,
        chunks: Sequence[str],
    ) -> UUID:
        session_uuid = _validate_chunks(session_id, title, chunks)
        normalized_url = source_url.strip() or None
        paper_uuid = uuid4()
        try:
            with self._engine.begin() as connection:
                self._ensure_session(connection, session_uuid, None)
                if normalized_url:
                    paper_uuid = connection.execute(
                        pg_insert(papers)
                        .values(
                            id=paper_uuid,
                            title=title.strip(),
                            source_url=normalized_url,
                        )
                        .on_conflict_do_update(
                            index_elements=[papers.c.source_url],
                            index_where=papers.c.source_url.is_not(None),
                            set_={
                                "title": title.strip(),
                                "updated_at": datetime.now(timezone.utc),
                            },
                        )
                        .returning(papers.c.id)
                    ).scalar_one()
                else:
                    connection.execute(
                        papers.insert().values(id=paper_uuid, title=title.strip())
                    )

                connection.execute(
                    pg_insert(session_papers)
                    .values(session_id=session_uuid, paper_id=paper_uuid)
                    .on_conflict_do_nothing()
                )
                connection.execute(
                    delete(paper_chunks).where(
                        paper_chunks.c.session_id == session_uuid,
                        paper_chunks.c.paper_id == paper_uuid,
                    )
                )
                connection.execute(
                    paper_chunks.insert(),
                    [
                        {
                            "session_id": session_uuid,
                            "paper_id": paper_uuid,
                            "chunk_index": index,
                            "total_chunks": len(chunks),
                            "content": chunk,
                        }
                        for index, chunk in enumerate(chunks)
                    ],
                )
                return paper_uuid
        except SQLAlchemyError as exc:
            self._raise("save_paper_chunks", exc)

    def load_paper_chunks(self, session_id: str) -> list[StoredChunk]:
        session_uuid = _as_uuid(session_id, "session_id")
        try:
            with self._engine.connect() as connection:
                rows = connection.execute(
                    select(
                        paper_chunks.c.paper_id,
                        papers.c.title,
                        papers.c.source_url,
                        paper_chunks.c.chunk_index,
                        paper_chunks.c.total_chunks,
                        paper_chunks.c.content,
                    )
                    .join(papers, papers.c.id == paper_chunks.c.paper_id)
                    .where(paper_chunks.c.session_id == session_uuid)
                    .order_by(paper_chunks.c.id)
                ).mappings()
                return [
                    StoredChunk(
                        paper_id=row["paper_id"],
                        title=row["title"],
                        source_url=row["source_url"] or "",
                        chunk_index=row["chunk_index"],
                        total_chunks=row["total_chunks"],
                        content=row["content"],
                    )
                    for row in rows
                ]
        except SQLAlchemyError as exc:
            self._raise("load_paper_chunks", exc)

    def save_feedback(
        self,
        session_id: str,
        rating: int,
        message_id: int | None = None,
        comment: str | None = None,
    ) -> UUID:
        session_uuid = _as_uuid(session_id, "session_id")
        _validate_rating(rating)
        feedback_uuid = uuid4()
        try:
            with self._engine.begin() as connection:
                self._ensure_session(connection, session_uuid, None)
                connection.execute(
                    feedback.insert().values(
                        id=feedback_uuid,
                        session_id=session_uuid,
                        message_id=message_id,
                        rating=rating,
                        comment=comment,
                    )
                )
            return feedback_uuid
        except SQLAlchemyError as exc:
            self._raise("save_feedback", exc)

    def healthcheck(self) -> bool:
        try:
            with self._engine.connect() as connection:
                return connection.execute(text("SELECT 1")).scalar_one() == 1
        except SQLAlchemyError:
            return False

    def close(self) -> None:
        self._engine.dispose()

    @staticmethod
    def _raise(operation: str, exc: SQLAlchemyError):
        logger.error(
            "database_operation_failed",
            extra={
                "event_data": {
                    "operation": operation,
                    "exception_type": type(exc).__name__,
                }
            },
        )
        raise PersistenceError("Persistent storage is temporarily unavailable.") from exc


def _as_uuid(value: str, field_name: str) -> UUID:
    try:
        return UUID(str(value))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"{field_name} must be a valid UUID") from exc


def _validate_message(session_id: str, role: str, content: str) -> UUID:
    session_uuid = _as_uuid(session_id, "session_id")
    if role not in VALID_ROLES:
        raise ValueError(f"role must be one of: {', '.join(sorted(VALID_ROLES))}")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("content cannot be empty")
    return session_uuid


def _validate_chunks(session_id: str, title: str, chunks: Sequence[str]) -> UUID:
    session_uuid = _as_uuid(session_id, "session_id")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title cannot be empty")
    if not chunks or any(not isinstance(chunk, str) or not chunk for chunk in chunks):
        raise ValueError("chunks must contain non-empty text")
    return session_uuid


def _validate_rating(rating: int) -> None:
    if rating not in {-1, 1}:
        raise ValueError("rating must be -1 or 1")


@lru_cache(maxsize=1)
def get_persistence() -> Persistence:
    """Return the configured process-wide persistence implementation."""
    settings = get_settings()
    if not settings.database_enabled:
        return NoOpPersistence()
    database_url = settings.get_database_url()
    if database_url is None:  # protected by Settings validation
        raise RuntimeError("DATABASE_URL is required")
    return PostgresPersistence(database_url)


def close_persistence() -> None:
    """Dispose database connections during controlled shutdown."""
    if get_persistence.cache_info().currsize:
        get_persistence().close()
    get_persistence.cache_clear()

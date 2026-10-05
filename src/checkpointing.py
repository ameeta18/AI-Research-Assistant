"""LangGraph checkpointer selection and PostgreSQL connection lifecycle."""

from __future__ import annotations

import os
from threading import RLock

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from src.config import get_settings


os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")

_lock = RLock()
_pool: ConnectionPool | None = None
_checkpointer = None


def _checkpoint_url(database_url: str) -> str:
    """Convert SQLAlchemy's explicit driver URL into a psycopg URL."""
    return database_url.replace("postgresql+psycopg://", "postgresql://", 1)


def get_checkpointer():
    """Return in-memory checkpoints locally or durable checkpoints in production."""
    global _pool, _checkpointer

    settings = get_settings()
    if not settings.database_enabled:
        return MemorySaver()

    with _lock:
        if _checkpointer is None:
            database_url = settings.get_database_url()
            if database_url is None:  # protected by Settings validation
                raise RuntimeError("DATABASE_URL is required")
            _pool = ConnectionPool(
                conninfo=_checkpoint_url(database_url),
                min_size=1,
                max_size=settings.database_pool_size + settings.database_max_overflow,
                timeout=settings.database_pool_timeout_seconds,
                open=True,
                kwargs={
                    "autocommit": True,
                    "prepare_threshold": 0,
                    "row_factory": dict_row,
                    "connect_timeout": int(settings.database_connect_timeout_seconds),
                    "application_name": "ai-researcher-checkpoints",
                },
            )
            _checkpointer = PostgresSaver(_pool)
        return _checkpointer


def close_checkpointer() -> None:
    """Close the shared PostgreSQL pool during controlled shutdown."""
    global _pool, _checkpointer
    with _lock:
        if _pool is not None:
            _pool.close()
        _pool = None
        _checkpointer = None

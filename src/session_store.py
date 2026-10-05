"""Session-scoped research resources for the single-process application."""

from collections.abc import Callable
from dataclasses import dataclass, field
from threading import RLock
import time
from typing import Any, Protocol

from src.config import get_settings


class SessionCapacityError(RuntimeError):
    """Raised when the in-memory session store reaches its configured limit."""


@dataclass
class SessionResources:
    """Mutable research state owned by one conversation session."""

    session_id: str
    created_at: float
    last_accessed_at: float
    last_read_text: str = ""
    last_read_url: str = ""
    embeddings: Any | None = field(default=None, repr=False)
    vectorstore: Any | None = None
    lock: RLock = field(default_factory=RLock, repr=False)


class SessionStore(Protocol):
    """Storage boundary that can later be backed by persistent infrastructure."""

    def get_or_create(self, session_id: str) -> SessionResources: ...

    def get(self, session_id: str) -> SessionResources | None: ...

    def delete(self, session_id: str) -> bool: ...

    def cleanup_expired(self) -> int: ...

    def clear(self) -> None: ...


class InMemorySessionStore:
    """Thread-safe, bounded store for session-specific PDF and FAISS state."""

    def __init__(
        self,
        ttl_seconds: float,
        max_sessions: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be greater than zero")
        if max_sessions <= 0:
            raise ValueError("max_sessions must be greater than zero")

        self._ttl_seconds = ttl_seconds
        self._max_sessions = max_sessions
        self._clock = clock
        self._sessions: dict[str, SessionResources] = {}
        self._lock = RLock()

    @staticmethod
    def _validate_session_id(session_id: str) -> str:
        normalized = session_id.strip() if isinstance(session_id, str) else ""
        if not normalized:
            raise ValueError("A non-empty session_id is required")
        return normalized

    def _cleanup_expired_locked(self, now: float) -> int:
        expired = [
            session_id
            for session_id, resources in self._sessions.items()
            if now - resources.last_accessed_at >= self._ttl_seconds
        ]
        for session_id in expired:
            del self._sessions[session_id]
        return len(expired)

    def get_or_create(self, session_id: str) -> SessionResources:
        session_id = self._validate_session_id(session_id)
        now = self._clock()

        with self._lock:
            self._cleanup_expired_locked(now)
            resources = self._sessions.get(session_id)
            if resources is None:
                if len(self._sessions) >= self._max_sessions:
                    raise SessionCapacityError(
                        "The maximum number of active sessions has been reached"
                    )
                resources = SessionResources(
                    session_id=session_id,
                    created_at=now,
                    last_accessed_at=now,
                )
                self._sessions[session_id] = resources
            else:
                resources.last_accessed_at = now
            return resources

    def get(self, session_id: str) -> SessionResources | None:
        session_id = self._validate_session_id(session_id)
        now = self._clock()

        with self._lock:
            self._cleanup_expired_locked(now)
            resources = self._sessions.get(session_id)
            if resources is not None:
                resources.last_accessed_at = now
            return resources

    def delete(self, session_id: str) -> bool:
        session_id = self._validate_session_id(session_id)
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def cleanup_expired(self) -> int:
        with self._lock:
            return self._cleanup_expired_locked(self._clock())

    def clear(self) -> None:
        """Remove all sessions. Intended for controlled shutdown and tests."""
        with self._lock:
            self._sessions.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)


_settings = get_settings()
_session_store: SessionStore = InMemorySessionStore(
    ttl_seconds=_settings.session_ttl_minutes * 60,
    max_sessions=_settings.max_active_sessions,
)


def get_session_store() -> SessionStore:
    """Return the configured session resource store."""
    return _session_store

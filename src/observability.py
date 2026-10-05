"""Structured application logging and optional LangSmith tracing."""

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import lru_cache
import json
import logging
import sys
from typing import Any, Iterator, TextIO

from langsmith import Client, tracing_context

from src.config import Settings, get_settings


_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_session_id: ContextVar[str | None] = ContextVar("session_id", default=None)
_HANDLER_MARKER = "_ai_researcher_json_handler"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


class JsonFormatter(logging.Formatter):
    """Format a controlled set of operational fields as one JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }

        request_id = _request_id.get()
        session_id = _session_id.get()
        if request_id:
            payload["request_id"] = request_id
        if session_id:
            payload["session_id"] = session_id

        event_data = getattr(record, "event_data", None)
        if isinstance(event_data, dict):
            payload.update(
                {
                    str(key): _json_safe(value)
                    for key, value in event_data.items()
                }
            )
        if record.exc_info and record.exc_info[0]:
            payload["exception_type"] = record.exc_info[0].__name__
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def configure_logging(
    settings: Settings | None = None,
    stream: TextIO | None = None,
) -> logging.Logger:
    """Configure one process-wide JSON handler without duplicating handlers."""
    settings = settings or get_settings()
    logger = logging.getLogger("ai_researcher")
    logger.setLevel(settings.log_level)
    logger.propagate = False

    if not any(getattr(handler, _HANDLER_MARKER, False) for handler in logger.handlers):
        handler = logging.StreamHandler(stream or sys.stdout)
        handler.setFormatter(JsonFormatter())
        setattr(handler, _HANDLER_MARKER, True)
        logger.addHandler(handler)
    return logger


def get_logger(component: str) -> logging.Logger:
    """Return an application logger for one component."""
    configure_logging()
    return logging.getLogger(f"ai_researcher.{component}")


@contextmanager
def log_context(
    *,
    request_id: str | None = None,
    session_id: str | None = None,
) -> Iterator[None]:
    """Attach correlation identifiers to every log emitted in this context."""
    request_token = _request_id.set(request_id) if request_id is not None else None
    session_token = _session_id.set(session_id) if session_id is not None else None
    try:
        yield
    finally:
        if session_token is not None:
            _session_id.reset(session_token)
        if request_token is not None:
            _request_id.reset(request_token)


def current_request_id() -> str | None:
    return _request_id.get()


@lru_cache(maxsize=1)
def get_langsmith_client() -> Client | None:
    """Create the trace client only when tracing is explicitly enabled."""
    settings = get_settings()
    if not settings.langsmith_tracing:
        return None
    return Client(
        api_key=settings.get_langsmith_api_key(),
        hide_inputs=settings.langsmith_hide_inputs,
        hide_outputs=settings.langsmith_hide_outputs,
        tracing_sampling_rate=settings.langsmith_tracing_sampling_rate,
    )


@contextmanager
def trace_research_run(*, interface: str, session_id: str) -> Iterator[None]:
    """Apply one trace policy consistently to API, Streamlit, and MCP work."""
    settings = get_settings()
    with tracing_context(
        project_name=settings.langsmith_project,
        tags=[interface],
        metadata={
            "interface": interface,
            "session_id": session_id,
            "request_id": current_request_id(),
        },
        enabled=settings.langsmith_tracing,
        client=get_langsmith_client(),
    ):
        yield

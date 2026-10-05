"""Optional Redis-backed JSON cache for reusable public search results."""

from collections.abc import Mapping
from functools import lru_cache
import hashlib
import json
from typing import Any, Protocol

from src.config import get_settings
from src.observability import get_logger


logger = get_logger("cache")
CACHE_SCHEMA_VERSION = "v1"


class JsonCache(Protocol):
    """Small cache boundary used by external search adapters."""

    enabled: bool

    def get_json(self, namespace: str, identity: Mapping[str, Any]) -> Any | None: ...

    def set_json(
        self,
        namespace: str,
        identity: Mapping[str, Any],
        value: Any,
    ) -> bool: ...


def build_cache_key(
    prefix: str,
    namespace: str,
    identity: Mapping[str, Any],
) -> str:
    """Build a deterministic key without exposing raw queries or user input."""
    canonical_identity = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    digest = hashlib.sha256(canonical_identity.encode("utf-8")).hexdigest()
    return f"{prefix}:{CACHE_SCHEMA_VERSION}:{namespace}:{digest}"


class NoOpJsonCache:
    """Disabled cache implementation that performs no external operations."""

    enabled = False

    def get_json(self, namespace: str, identity: Mapping[str, Any]) -> None:
        return None

    def set_json(
        self,
        namespace: str,
        identity: Mapping[str, Any],
        value: Any,
    ) -> bool:
        return False


class RedisJsonCache:
    """Failure-tolerant Redis cache with JSON serialization and TTL expiry."""

    enabled = True

    def __init__(
        self,
        client: Any,
        *,
        prefix: str,
        ttl_seconds: int,
        recoverable_errors: tuple[type[BaseException], ...] = (
            ConnectionError,
            TimeoutError,
        ),
    ) -> None:
        self._client = client
        self._prefix = prefix
        self._ttl_seconds = ttl_seconds
        self._recoverable_errors = recoverable_errors

    @classmethod
    def from_url(cls, redis_url: str) -> "RedisJsonCache":
        """Create a lazy Redis client; no network call occurs during construction."""
        from redis import Redis
        from redis.exceptions import RedisError

        settings = get_settings()
        client = Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=settings.redis_connect_timeout_seconds,
            socket_timeout=settings.redis_socket_timeout_seconds,
            health_check_interval=30,
        )
        return cls(
            client,
            prefix=settings.cache_key_prefix,
            ttl_seconds=settings.cache_ttl_seconds,
            recoverable_errors=(RedisError,),
        )

    def _key(self, namespace: str, identity: Mapping[str, Any]) -> str:
        return build_cache_key(self._prefix, namespace, identity)

    def get_json(self, namespace: str, identity: Mapping[str, Any]) -> Any | None:
        key = self._key(namespace, identity)
        try:
            cached = self._client.get(key)
        except self._recoverable_errors as exc:
            logger.warning(
                "cache_read_failed",
                extra={
                    "event_data": {
                        "namespace": namespace,
                        "exception_type": type(exc).__name__,
                    }
                },
            )
            return None

        if cached is None:
            logger.info(
                "cache_miss",
                extra={"event_data": {"namespace": namespace}},
            )
            return None

        try:
            value = json.loads(cached)
        except (TypeError, json.JSONDecodeError):
            logger.warning(
                "cache_entry_invalid",
                extra={"event_data": {"namespace": namespace}},
            )
            self._delete_invalid(key, namespace)
            return None

        logger.info(
            "cache_hit",
            extra={"event_data": {"namespace": namespace}},
        )
        return value

    def _delete_invalid(self, key: str, namespace: str) -> None:
        try:
            self._client.delete(key)
        except self._recoverable_errors as exc:
            logger.warning(
                "cache_delete_failed",
                extra={
                    "event_data": {
                        "namespace": namespace,
                        "exception_type": type(exc).__name__,
                    }
                },
            )

    def set_json(
        self,
        namespace: str,
        identity: Mapping[str, Any],
        value: Any,
    ) -> bool:
        key = self._key(namespace, identity)
        payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        try:
            self._client.set(key, payload, ex=self._ttl_seconds)
        except self._recoverable_errors as exc:
            logger.warning(
                "cache_write_failed",
                extra={
                    "event_data": {
                        "namespace": namespace,
                        "exception_type": type(exc).__name__,
                    }
                },
            )
            return False

        logger.info(
            "cache_write_completed",
            extra={
                "event_data": {
                    "namespace": namespace,
                    "ttl_seconds": self._ttl_seconds,
                }
            },
        )
        return True


@lru_cache(maxsize=1)
def get_cache() -> JsonCache:
    """Return the configured cache without connecting when caching is disabled."""
    settings = get_settings()
    if not settings.cache_enabled:
        return NoOpJsonCache()

    redis_url = settings.get_redis_url()
    if redis_url is None:
        return NoOpJsonCache()
    return RedisJsonCache.from_url(redis_url)

"""Shared validation and retry boundaries for external services."""

from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
import random
import time
from typing import Any
from urllib.parse import urlparse

import requests

from src.config import get_settings
from src.observability import get_logger


RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
logger = get_logger("http")


class UserFacingError(RuntimeError):
    """Base error whose message is safe to return to an API or tool caller."""


class InputValidationError(UserFacingError):
    """Raised when caller-controlled input is invalid or outside safe limits."""


class ExternalServiceError(UserFacingError):
    """Raised when an external service cannot complete a request safely."""


class ExternalServiceTimeout(ExternalServiceError):
    """Raised when an external service exhausts all timeout retries."""


class DownloadTooLargeError(InputValidationError):
    """Raised when a download exceeds its configured size limit."""


class InvalidPDFError(InputValidationError):
    """Raised when downloaded content is not a readable PDF."""


def validate_text(value: str, field_name: str, max_length: int) -> str:
    """Return trimmed non-empty text that fits within a configured boundary."""
    if not isinstance(value, str):
        raise InputValidationError(f"{field_name} must be text.")

    normalized = value.strip()
    if not normalized:
        raise InputValidationError(f"{field_name} cannot be empty.")
    if len(normalized) > max_length:
        raise InputValidationError(
            f"{field_name} is too long; the maximum is {max_length} characters."
        )
    return normalized


def validate_int_range(value: int, field_name: str, minimum: int, maximum: int) -> int:
    """Validate an integer without accepting booleans as integers."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputValidationError(f"{field_name} must be an integer.")
    if not minimum <= value <= maximum:
        raise InputValidationError(
            f"{field_name} must be between {minimum} and {maximum}."
        )
    return value


def validate_http_url(url: str, field_name: str = "URL") -> str:
    """Accept only absolute HTTP(S) URLs without embedded credentials."""
    normalized = validate_text(url, field_name, max_length=2_048)
    parsed = urlparse(normalized)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise InputValidationError(f"{field_name} must be an absolute HTTP or HTTPS URL.")
    if parsed.username or parsed.password:
        raise InputValidationError(f"{field_name} must not contain embedded credentials.")
    return normalized


def _retry_after_seconds(response: requests.Response, maximum: float) -> float | None:
    value = response.headers.get("Retry-After")
    if not value:
        return None

    try:
        delay = float(value)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
            now = datetime.now(retry_at.tzinfo or timezone.utc)
            delay = (retry_at - now).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    return min(max(delay, 0.0), maximum)


def _backoff_delay(attempt_index: int, response: requests.Response | None) -> float:
    settings = get_settings()
    if response is not None:
        retry_after = _retry_after_seconds(
            response,
            settings.http_max_retry_delay_seconds,
        )
        if retry_after is not None:
            return retry_after

    delay = settings.http_backoff_seconds * (2**attempt_index)
    delay += random.uniform(0.0, settings.http_jitter_seconds)
    return min(delay, settings.http_max_retry_delay_seconds)


def request_with_retries(
    method: str,
    url: str,
    *,
    service_name: str,
    **kwargs: Any,
) -> requests.Response:
    """Make an HTTP request with bounded retries for transient failures only."""
    settings = get_settings()
    kwargs.setdefault(
        "timeout",
        (
            settings.http_connect_timeout_seconds,
            settings.http_read_timeout_seconds,
        ),
    )

    last_was_timeout = False
    for attempt_index in range(settings.http_max_attempts):
        response = None
        attempt_started_at = time.perf_counter()
        failure_type = "retryable_status"
        try:
            response = requests.request(method, url, **kwargs)
            last_was_timeout = False
            if response.status_code not in RETRYABLE_STATUS_CODES:
                response.raise_for_status()
                logger.info(
                    "external_request_completed",
                    extra={
                        "event_data": {
                            "service": service_name,
                            "status_code": response.status_code,
                            "attempt": attempt_index + 1,
                            "duration_ms": round(
                                (time.perf_counter() - attempt_started_at) * 1000,
                                2,
                            ),
                        }
                    },
                )
                return response
        except requests.Timeout:
            last_was_timeout = True
            failure_type = "timeout"
        except requests.ConnectionError:
            last_was_timeout = False
            failure_type = "connection_error"
        except requests.RequestException as exc:
            logger.warning(
                "external_request_rejected",
                extra={
                    "event_data": {
                        "service": service_name,
                        "status_code": (
                            response.status_code if response is not None else None
                        ),
                        "attempt": attempt_index + 1,
                    }
                },
            )
            raise ExternalServiceError(
                f"{service_name} could not complete the request."
            ) from exc

        if attempt_index == settings.http_max_attempts - 1:
            break

        delay = _backoff_delay(attempt_index, response)
        if response is not None:
            response.close()
        logger.warning(
            "external_request_retrying",
            extra={
                "event_data": {
                    "service": service_name,
                    "status_code": (
                        response.status_code if response is not None else None
                    ),
                    "failure_type": failure_type,
                    "attempt": attempt_index + 1,
                    "next_attempt": attempt_index + 2,
                    "delay_seconds": round(delay, 3),
                }
            },
        )
        time.sleep(delay)

    if response is not None:
        response.close()
    logger.error(
        "external_request_failed",
        extra={
            "event_data": {
                "service": service_name,
                "status_code": response.status_code if response is not None else None,
                "attempts": settings.http_max_attempts,
                "failure_type": "timeout" if last_was_timeout else failure_type,
            }
        },
    )
    if last_was_timeout:
        raise ExternalServiceTimeout(
            f"{service_name} timed out. Please try again shortly."
        )
    raise ExternalServiceError(
        f"{service_name} is temporarily unavailable. Please try again shortly."
    )

"""Select readable PDF links without downloading an entire paper."""

from collections.abc import Iterable

import requests

from src.reliability import UserFacingError, request_with_retries, validate_http_url


PDF_SIGNATURE = b"%PDF-"


def is_readable_pdf_url(url: str) -> bool:
    """Return whether an HTTP(S) URL responds with PDF content."""
    try:
        normalized_url = validate_http_url(url, "PDF candidate URL")
        response = request_with_retries(
            "GET",
            normalized_url,
            service_name="PDF link validation",
            headers={
                "Accept": "application/pdf",
                "Range": "bytes=0-1023",
            },
            stream=True,
        )
    except UserFacingError:
        return False

    prefix = bytearray()
    try:
        for chunk in response.iter_content(chunk_size=1024):
            if not chunk:
                continue
            prefix.extend(chunk)
            break
    except requests.RequestException:
        return False
    finally:
        response.close()

    return bytes(prefix).lstrip().startswith(PDF_SIGNATURE)


def select_working_pdf(
    candidates: Iterable[tuple[str, str | None]],
) -> tuple[str, str] | None:
    """Return the first readable PDF URL and its source, preserving priority."""
    seen_urls: set[str] = set()
    for source, candidate_url in candidates:
        if not isinstance(candidate_url, str):
            continue
        candidate_url = candidate_url.strip()
        if not candidate_url or candidate_url in seen_urls:
            continue
        seen_urls.add(candidate_url)
        if is_readable_pdf_url(candidate_url):
            return candidate_url, source
    return None

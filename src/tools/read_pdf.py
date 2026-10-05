# src/tools/read_pdf.py
import io
from typing import Annotated

import PyPDF2
import requests
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from src.config import MAX_PDF_BYTES, MAX_PDF_PAGES, MAX_PDF_TEXT_CHARS
from src.reliability import (
    DownloadTooLargeError,
    ExternalServiceError,
    InvalidPDFError,
    UserFacingError,
    request_with_retries,
    validate_http_url,
)
from src.session_store import get_session_store


def get_last_read_text(session_id: str) -> str:
    """Access the last successfully read paper for one session."""
    resources = get_session_store().get(session_id)
    if resources is None:
        return ""
    with resources.lock:
        return resources.last_read_text


def read_pdf_for_session(url: str, session_id: str) -> str:
    """Download a PDF and store its extracted text in one session."""
    url = validate_http_url(url, "PDF URL")
    response = request_with_retries(
        "GET",
        url,
        service_name="PDF download",
        stream=True,
    )

    content_length = response.headers.get("Content-Length")
    if content_length:
        try:
            if int(content_length) > MAX_PDF_BYTES:
                response.close()
                raise DownloadTooLargeError(
                    f"PDF exceeds the {MAX_PDF_BYTES // 1_000_000} MB download limit."
                )
        except ValueError:
            pass

    content = bytearray()
    try:
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            content.extend(chunk)
            if len(content) > MAX_PDF_BYTES:
                raise DownloadTooLargeError(
                    f"PDF exceeds the {MAX_PDF_BYTES // 1_000_000} MB download limit."
                )
    except requests.RequestException as exc:
        raise ExternalServiceError(
            "PDF download was interrupted. Please try again shortly."
        ) from exc
    finally:
        response.close()

    if not bytes(content).lstrip().startswith(b"%PDF-"):
        raise InvalidPDFError("The downloaded file is not a valid PDF.")

    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
        page_count = len(pdf_reader.pages)
    except Exception as exc:
        raise InvalidPDFError("The downloaded PDF could not be read.") from exc
    if page_count > MAX_PDF_PAGES:
        raise DownloadTooLargeError(
            f"PDF exceeds the {MAX_PDF_PAGES}-page processing limit."
        )

    pages = []
    extracted_characters = 0
    for page in pdf_reader.pages:
        try:
            extracted = page.extract_text()
        except Exception as exc:
            raise InvalidPDFError("Text could not be extracted from the PDF.") from exc
        if extracted:
            pages.append(extracted)
            extracted_characters += len(extracted)
            if extracted_characters > MAX_PDF_TEXT_CHARS:
                raise DownloadTooLargeError(
                    "PDF contains more text than the processing limit allows."
                )

    full_text = "\n".join(pages).strip()
    if not full_text:
        return "Error: Could not extract any text from this PDF."

    resources = get_session_store().get_or_create(session_id)
    with resources.lock:
        resources.last_read_text = full_text
        resources.last_read_url = url

    return full_text


@tool
def read_pdf(
    url: str,
    session_id: Annotated[str, InjectedState("session_id")],
) -> str:
    """Read and extract text from a PDF file given its URL.

    Args:
        url: The URL of the PDF file to read

    Returns:
        The extracted text content from the PDF
    """
    try:
        return read_pdf_for_session(url, session_id)
    except UserFacingError as exc:
        return f"Error: {exc}"

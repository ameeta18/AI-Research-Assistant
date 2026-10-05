import subprocess
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from src.config import MAX_PDF_BYTES
from src.reliability import (
    DownloadTooLargeError,
    ExternalServiceError,
    ExternalServiceTimeout,
    InputValidationError,
    InvalidPDFError,
    request_with_retries,
    validate_http_url,
    validate_int_range,
    validate_text,
)
from src.tools.read_pdf import read_pdf_for_session
from src.tools.write_pdf import _compile_with_tectonic


def make_response(status_code: int, headers: dict | None = None) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response.headers.update(headers or {})
    response._content = b"{}"
    response.url = "https://service.example/resource"
    response.raw = Mock()
    return response


class HttpReliabilityTests(unittest.TestCase):
    @patch("src.reliability.random.uniform", return_value=0.0)
    @patch("src.reliability.time.sleep")
    @patch("src.reliability.requests.request")
    def test_timeout_is_retried_with_backoff(self, request, sleep, _jitter):
        request.side_effect = [requests.Timeout(), make_response(200)]

        result = request_with_retries(
            "GET",
            "https://service.example/resource",
            service_name="Test service",
        )

        self.assertEqual(result.status_code, 200)
        self.assertEqual(request.call_count, 2)
        sleep.assert_called_once_with(1.0)
        self.assertEqual(request.call_args.kwargs["timeout"], (5.0, 30.0))

    @patch("src.reliability.time.sleep")
    @patch("src.reliability.requests.request")
    def test_retry_after_header_is_respected(self, request, sleep):
        request.side_effect = [
            make_response(429, {"Retry-After": "4"}),
            make_response(200),
        ]

        request_with_retries(
            "GET",
            "https://service.example/resource",
            service_name="Test service",
        )

        sleep.assert_called_once_with(4.0)

    @patch("src.reliability.requests.request")
    def test_permanent_client_error_is_not_retried_or_leaked(self, request):
        request.return_value = make_response(404)

        with self.assertRaisesRegex(
            ExternalServiceError,
            "Test service could not complete the request",
        ):
            request_with_retries(
                "GET",
                "https://service.example/private?token=secret",
                service_name="Test service",
            )

        self.assertEqual(request.call_count, 1)

    @patch("src.reliability.random.uniform", return_value=0.0)
    @patch("src.reliability.time.sleep")
    @patch("src.reliability.requests.request", side_effect=requests.Timeout())
    def test_timeout_stops_after_configured_attempts(self, request, sleep, _jitter):
        with self.assertRaises(ExternalServiceTimeout):
            request_with_retries(
                "GET",
                "https://service.example/resource",
                service_name="Test service",
            )

        self.assertEqual(request.call_count, 3)
        self.assertEqual(sleep.call_count, 2)


class BoundaryValidationTests(unittest.TestCase):
    def test_text_and_integer_boundaries(self):
        self.assertEqual(validate_text("  useful query  ", "Query", 20), "useful query")
        self.assertEqual(validate_int_range(5, "k", 1, 10), 5)

        with self.assertRaises(InputValidationError):
            validate_text("   ", "Query", 20)
        with self.assertRaises(InputValidationError):
            validate_text("too long", "Query", 3)
        with self.assertRaises(InputValidationError):
            validate_int_range(True, "k", 1, 10)
        with self.assertRaises(InputValidationError):
            validate_int_range(11, "k", 1, 10)

    def test_pdf_url_requires_http_without_credentials(self):
        self.assertEqual(
            validate_http_url("https://example.com/paper.pdf"),
            "https://example.com/paper.pdf",
        )
        for invalid_url in (
            "file:///etc/passwd",
            "example.com/paper.pdf",
            "https://user:password@example.com/paper.pdf",
        ):
            with self.subTest(url=invalid_url):
                with self.assertRaises(InputValidationError):
                    validate_http_url(invalid_url)

    def test_pdf_content_length_is_bounded(self):
        response = Mock()
        response.headers = {"Content-Length": str(MAX_PDF_BYTES + 1)}

        with (
            patch("src.tools.read_pdf.request_with_retries", return_value=response),
            self.assertRaises(DownloadTooLargeError),
        ):
            read_pdf_for_session("https://example.com/large.pdf", "session-1")

        response.close.assert_called_once()

    def test_non_pdf_download_is_rejected(self):
        response = Mock()
        response.headers = {}
        response.iter_content.return_value = [b"<html>not a pdf</html>"]

        with (
            patch("src.tools.read_pdf.request_with_retries", return_value=response),
            self.assertRaises(InvalidPDFError),
        ):
            read_pdf_for_session("https://example.com/not-pdf", "session-1")

        response.close.assert_called_once()

    @patch(
        "src.tools.write_pdf.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="tectonic", timeout=120),
    )
    def test_latex_compilation_timeout_is_controlled(self, _run):
        success, log = _compile_with_tectonic(Path("paper.tex"), Path("output"))

        self.assertFalse(success)
        self.assertIn("timed out", log)


if __name__ == "__main__":
    unittest.main()

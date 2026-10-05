import unittest
from unittest.mock import Mock, call, patch

from src.cache import NoOpJsonCache
from src.reliability import ExternalServiceError
from src.tools.pdf_sources import is_readable_pdf_url, select_working_pdf
from src.tools.semantic_scholar_tool import _search_semantic_scholar_papers


class PdfLinkValidationTests(unittest.TestCase):
    def test_pdf_signature_is_accepted_and_response_is_closed(self):
        response = Mock()
        response.iter_content.return_value = [b"%PDF-1.7 content"]

        with patch(
            "src.tools.pdf_sources.request_with_retries",
            return_value=response,
        ) as request:
            result = is_readable_pdf_url("https://example.com/paper")

        self.assertTrue(result)
        response.close.assert_called_once()
        request.assert_called_once_with(
            "GET",
            "https://example.com/paper",
            service_name="PDF link validation",
            headers={"Accept": "application/pdf", "Range": "bytes=0-1023"},
            stream=True,
        )

    def test_html_page_is_rejected_even_when_url_looks_like_a_pdf(self):
        response = Mock()
        response.iter_content.return_value = [b"<html>Access denied</html>"]

        with patch(
            "src.tools.pdf_sources.request_with_retries",
            return_value=response,
        ):
            result = is_readable_pdf_url("https://example.com/paper.pdf")

        self.assertFalse(result)
        response.close.assert_called_once()

    def test_failed_validation_is_treated_as_unavailable(self):
        with patch(
            "src.tools.pdf_sources.request_with_retries",
            side_effect=ExternalServiceError("blocked"),
        ):
            self.assertFalse(is_readable_pdf_url("https://example.com/paper.pdf"))

    def test_arxiv_is_preferred_then_open_access_is_used_as_fallback(self):
        arxiv_url = "https://arxiv.org/pdf/1234.5678"
        open_access_url = "https://repository.example/paper.pdf"

        with patch(
            "src.tools.pdf_sources.is_readable_pdf_url",
            side_effect=[False, True],
        ) as validate:
            selected = select_working_pdf([
                ("arXiv", arxiv_url),
                ("Semantic Scholar open access", open_access_url),
            ])

        self.assertEqual(
            selected,
            (open_access_url, "Semantic Scholar open access"),
        )
        self.assertEqual(validate.call_args_list, [call(arxiv_url), call(open_access_url)])


class SemanticScholarPdfSelectionTests(unittest.TestCase):
    def test_requests_open_access_pdf_and_never_uses_landing_page_as_pdf(self):
        response = Mock()
        response.json.return_value = {
            "data": [
                {
                    "title": "Open Paper",
                    "abstract": "Summary",
                    "authors": [{"name": "Researcher"}],
                    "year": 2025,
                    "url": "https://www.semanticscholar.org/paper/landing-page",
                    "externalIds": {},
                    "openAccessPdf": {
                        "url": "https://repository.example/open-paper.pdf"
                    },
                }
            ]
        }

        with (
            patch(
                "src.tools.semantic_scholar_tool.get_cache",
                return_value=NoOpJsonCache(),
            ),
            patch(
                "src.tools.semantic_scholar_tool.request_with_retries",
                return_value=response,
            ) as request,
            patch(
                "src.tools.semantic_scholar_tool.select_working_pdf",
                return_value=(
                    "https://repository.example/open-paper.pdf",
                    "Semantic Scholar open access",
                ),
            ) as select,
        ):
            papers = _search_semantic_scholar_papers("open research", 1)

        fields = request.call_args.kwargs["params"]["fields"]
        self.assertIn("openAccessPdf", fields)
        candidates = select.call_args.args[0]
        self.assertNotIn(
            "https://www.semanticscholar.org/paper/landing-page",
            [url for _, url in candidates],
        )
        self.assertEqual(papers[0]["pdf"], "https://repository.example/open-paper.pdf")
        self.assertEqual(
            papers[0]["paper_url"],
            "https://www.semanticscholar.org/paper/landing-page",
        )

    def test_paper_without_a_working_pdf_is_not_returned(self):
        response = Mock()
        response.json.return_value = {
            "data": [
                {
                    "title": "Blocked Paper",
                    "abstract": "Summary",
                    "authors": [],
                    "year": 2025,
                    "url": "https://www.semanticscholar.org/paper/blocked",
                    "externalIds": {},
                    "openAccessPdf": None,
                }
            ]
        }

        with (
            patch(
                "src.tools.semantic_scholar_tool.get_cache",
                return_value=NoOpJsonCache(),
            ),
            patch(
                "src.tools.semantic_scholar_tool.request_with_retries",
                return_value=response,
            ),
            patch(
                "src.tools.semantic_scholar_tool.select_working_pdf",
                return_value=None,
            ),
        ):
            papers = _search_semantic_scholar_papers("blocked research", 1)

        self.assertEqual(papers, [])


if __name__ == "__main__":
    unittest.main()

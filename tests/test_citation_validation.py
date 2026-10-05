from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.tools.write_pdf import (
    find_unresolved_citation_warnings,
    render_latex_pdf,
    validate_citations,
)


VALID_LATEX = r"""
\documentclass{article}
\usepackage{hyperref}
\begin{document}
Quantum algorithms can support reaction simulations \cite{weiss2025}.

\begin{thebibliography}{99}
\bibitem{weiss2025}
R. Weiss et al. \textit{Solving Reaction Dynamics with Quantum Computing
Algorithms}. 2025. \url{https://arxiv.org/pdf/2404.00202}
\end{thebibliography}
\end{document}
"""


class CitationValidationTests(unittest.TestCase):
    def test_valid_self_contained_citations_pass(self):
        self.assertEqual(validate_citations(VALID_LATEX), [])

    def test_missing_in_text_citation_is_rejected(self):
        latex = VALID_LATEX.replace(r" \cite{weiss2025}", "")

        errors = validate_citations(latex)

        self.assertTrue(any("No in-text citation" in error for error in errors))
        self.assertTrue(any("not cited" in error for error in errors))

    def test_unknown_duplicate_and_uncited_keys_are_reported(self):
        latex = VALID_LATEX.replace(
            r"\cite{weiss2025}",
            r"\cite{missing-key}",
        ).replace(
            r"\end{thebibliography}",
            r"""
\bibitem{weiss2025}
R. Weiss et al. \textit{Duplicate title}. 2025.
\url{https://arxiv.org/pdf/2404.00202}
\end{thebibliography}
""",
        )

        errors = validate_citations(latex)

        self.assertTrue(any("Duplicate bibliography keys" in error for error in errors))
        self.assertTrue(any("without matching" in error for error in errors))
        self.assertTrue(any("not cited" in error for error in errors))

    def test_incomplete_reference_metadata_is_rejected(self):
        latex = VALID_LATEX.replace(
            r"""R. Weiss et al. \textit{Solving Reaction Dynamics with Quantum Computing
Algorithms}. 2025. \url{https://arxiv.org/pdf/2404.00202}""",
            "Incomplete reference",
        )

        errors = validate_citations(latex)
        combined = " ".join(errors)

        self.assertIn("author", combined)
        self.assertIn("title", combined)
        self.assertIn("four-digit year", combined)
        self.assertIn("HTTP(S) URL", combined)

    def test_citations_inside_comments_do_not_count(self):
        latex = VALID_LATEX.replace(
            r"Quantum algorithms can support reaction simulations \cite{weiss2025}.",
            r"Quantum algorithms can support reaction simulations. % \cite{weiss2025}",
        )

        errors = validate_citations(latex)

        self.assertTrue(any("No in-text citation" in error for error in errors))

    def test_compiler_citation_warnings_are_detected(self):
        log = """
LaTeX Warning: Citation `weiss2025' on page 1 undefined.
LaTeX Warning: There were undefined references.
"""

        warnings = find_unresolved_citation_warnings(log)

        self.assertEqual(len(warnings), 2)

    def test_renderer_does_not_compile_invalid_citations(self):
        invalid = VALID_LATEX.replace(r"\cite{weiss2025}", r"\cite{missing}")

        with patch("src.tools.write_pdf.shutil.which") as which:
            result = render_latex_pdf.func(invalid)

        self.assertIn("Citation validation failed", result)
        which.assert_not_called()

    def test_renderer_rejects_unresolved_compiler_warnings(self):
        warning = "LaTeX Warning: Citation `weiss2025' undefined."

        with (
            TemporaryDirectory() as directory,
            patch("src.tools.write_pdf.OUTPUT_DIR", Path(directory)),
            patch(
                "src.tools.write_pdf.shutil.which",
                side_effect=lambda command: "tectonic" if command == "tectonic" else None,
            ),
            patch(
                "src.tools.write_pdf._compile_with_tectonic",
                return_value=(True, warning),
            ),
        ):
            result = render_latex_pdf.func(VALID_LATEX)

        self.assertIn("PDF compilation failed", result)
        self.assertIn("Citation", result)

    def test_renderer_accepts_valid_resolved_document(self):
        def compile_successfully(tex_path: Path, _output_dir: Path):
            tex_path.with_suffix(".pdf").write_bytes(b"%PDF-valid")
            return True, "Compilation completed without warnings."

        with (
            TemporaryDirectory() as directory,
            patch("src.tools.write_pdf.OUTPUT_DIR", Path(directory)),
            patch("src.tools.write_pdf.shutil.which", return_value="tectonic"),
            patch(
                "src.tools.write_pdf._compile_with_tectonic",
                side_effect=compile_successfully,
            ),
        ):
            result = render_latex_pdf.func(VALID_LATEX)

        self.assertIn("PDF generated successfully", result)


if __name__ == "__main__":
    unittest.main()

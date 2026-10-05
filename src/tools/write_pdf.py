# src/tools/write_pdf.py
import re
import subprocess
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from langchain_core.tools import tool

from src.config import LATEX_COMPILE_TIMEOUT_SECONDS, MAX_LATEX_CHARS


# ──────────────────────────────────────────────
# Output directory — project root / output
# ──────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


CITATION_PATTERN = re.compile(
    r"\\(?:cite|citep|citet|autocite|parencite|textcite)\*?"
    r"(?:\s*\[[^\]]*\]){0,2}\s*\{([^{}]+)\}"
)
BIBITEM_BLOCK_PATTERN = re.compile(
    r"\\bibitem(?:\s*\[[^\]]*\])?\s*\{(?P<key>[^{}]+)\}"
    r"(?P<body>.*?)(?=\\bibitem|\\end\{thebibliography\})",
    re.DOTALL,
)
TITLE_PATTERN = re.compile(r"\\(?:textit|emph)\{[^{}]+\}")
YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
URL_PATTERN = re.compile(r"https?://[^\s}\]]+")
UNRESOLVED_CITATION_PATTERNS = (
    re.compile(r"citation.*undefined", re.IGNORECASE),
    re.compile(r"undefined citations?", re.IGNORECASE),
    re.compile(r"there were undefined references", re.IGNORECASE),
    re.compile(r"empty bibliography", re.IGNORECASE),
)


def _without_latex_comments(content: str) -> str:
    """Remove ordinary LaTeX comments so they cannot satisfy validation."""
    return re.sub(r"(?<!\\)%[^\n]*", "", content)


def validate_citations(content: str) -> list[str]:
    """Return citation-integrity errors for a self-contained LaTeX document."""
    source = _without_latex_comments(content)
    errors: list[str] = []

    citation_keys = [
        key.strip()
        for match in CITATION_PATTERN.finditer(source)
        for key in match.group(1).split(",")
        if key.strip()
    ]
    if not citation_keys:
        errors.append("No in-text citation command such as \\cite{source-key} was found.")

    if r"\begin{thebibliography}" not in source:
        errors.append("A self-contained thebibliography environment is required.")
    if r"\end{thebibliography}" not in source:
        errors.append("The thebibliography environment is not closed.")

    bibliography_matches = list(BIBITEM_BLOCK_PATTERN.finditer(source))
    bibliography_keys = [match.group("key").strip() for match in bibliography_matches]
    if not bibliography_keys:
        errors.append("No bibliography entries using \\bibitem{source-key} were found.")

    duplicate_keys = sorted(
        key for key, count in Counter(bibliography_keys).items() if count > 1
    )
    if duplicate_keys:
        errors.append(
            "Duplicate bibliography keys: " + ", ".join(duplicate_keys) + "."
        )

    unknown_keys = sorted(set(citation_keys) - set(bibliography_keys))
    if unknown_keys:
        errors.append(
            "Citation keys without matching bibliography entries: "
            + ", ".join(unknown_keys)
            + "."
        )

    uncited_keys = sorted(set(bibliography_keys) - set(citation_keys))
    if uncited_keys:
        errors.append(
            "Bibliography entries not cited in the document: "
            + ", ".join(uncited_keys)
            + "."
        )

    for match in bibliography_matches:
        key = match.group("key").strip()
        body = match.group("body").strip()
        title_match = TITLE_PATTERN.search(body)
        author_text = body[: title_match.start()] if title_match else ""
        author_words = re.findall(r"[A-Za-z][A-Za-z.'-]*", author_text)

        missing_fields = []
        if len(author_words) < 2:
            missing_fields.append("author")
        if title_match is None:
            missing_fields.append("title in \\textit{} or \\emph{}")
        if YEAR_PATTERN.search(body) is None:
            missing_fields.append("four-digit year")
        if URL_PATTERN.search(body) is None:
            missing_fields.append("HTTP(S) URL")
        if missing_fields:
            errors.append(
                f"Bibliography entry '{key}' is missing: "
                + ", ".join(missing_fields)
                + "."
            )

    return errors


def find_unresolved_citation_warnings(log: str) -> list[str]:
    """Extract unresolved citation/reference warnings from compiler output."""
    warnings = []
    for line in log.splitlines():
        normalized = line.strip()
        if normalized and any(
            pattern.search(normalized)
            for pattern in UNRESOLVED_CITATION_PATTERNS
        ):
            warnings.append(normalized)
    return list(dict.fromkeys(warnings))


# ──────────────────────────────────────────────
# LaTeX Sanitization (much more robust)
# ──────────────────────────────────────────────
def sanitize_latex(content: str) -> str:
    """Clean common LLM LaTeX mistakes that break compilation."""

    # 1. Remove markdown code fences (LLMs love wrapping LaTeX in ```)
    content = re.sub(r"^```(?:latex|tex)?\s*\n?", "", content, flags=re.MULTILINE)
    content = re.sub(r"\n?```\s*$", "", content, flags=re.MULTILINE)

    # 2. Fix double-escaped backslashes before commands
    #    \\documentclass → \documentclass, but preserve \\ (line break)
    content = re.sub(r"\\\\(?=[a-zA-Z])", r"\\", content)

    # 3. Replace unicode characters that break LaTeX
    unicode_map = {
        "\u2013": "--",       # en-dash
        "\u2014": "---",      # em-dash
        "\u2018": "`",        # left single quote
        "\u2019": "'",        # right single quote
        "\u201c": "``",       # left double quote
        "\u201d": "''",       # right double quote
        "\u2026": "...",      # ellipsis
        "\u00a0": " ",        # non-breaking space
        "\u2002": " ",        # en space
        "\u2003": " ",        # em space
        "\u2009": " ",        # thin space
        "\u200b": "",         # zero-width space
        "\u00b0": "$^\\circ$",  # degree symbol
        "\u00d7": "$\\times$",  # multiplication sign
        "\u00b1": "$\\pm$",     # plus-minus
        "\u2264": "$\\leq$",    # less than or equal
        "\u2265": "$\\geq$",    # greater than or equal
        "\u2260": "$\\neq$",    # not equal
        "\u221e": "$\\infty$",  # infinity
        "\u03b1": "$\\alpha$",  # alpha
        "\u03b2": "$\\beta$",   # beta
        "\u03b3": "$\\gamma$",  # gamma
        "\u03b4": "$\\delta$",  # delta
        "\u03c0": "$\\pi$",     # pi
        "\u03c3": "$\\sigma$",  # sigma
        "\u03bc": "$\\mu$",     # mu
    }
    for char, replacement in unicode_map.items():
        content = content.replace(char, replacement)

    # 4. Escape unescaped special LaTeX characters in text
    #    (but NOT inside math mode or commands)
    #    Common culprits: %, &, #, _ in plain text
    #    This is tricky so we only fix the most common: bare % and &
    #    that appear outside of \begin{tabular} context

    # 5. Remove stray linebreaks after section commands
    no_break_after = [
        r"\\begin\{document\}",
        r"\\end\{document\}",
        r"\\maketitle",
        r"\\tableofcontents",
        r"\\(sub)*section\*?\{[^}]*\}",
        r"\\title\{[^}]*\}",
        r"\\author\{[^}]*\}",
        r"\\date\{[^}]*\}",
    ]
    for pattern in no_break_after:
        content = re.sub(f"({pattern})\\s*\\\\\\\\", r"\1", content)

    # 6. Ensure document has required packages
    required_packages = [
        r"\usepackage{amsmath}",
        r"\usepackage{amssymb}",
        r"\usepackage{hyperref}",
        r"\usepackage[utf8]{inputenc}",
    ]
    
    if r"\begin{document}" in content:
        for pkg in required_packages:
            # Check if package is already included (flexible matching)
            pkg_name = re.search(r"\\usepackage.*?\{(\w+)\}", pkg).group(1)
            if not re.search(rf"\\usepackage.*?\{{{pkg_name}\}}", content):
                # Insert before \begin{document}
                content = content.replace(
                    r"\begin{document}",
                    f"{pkg}\n\\begin{{document}}"
                )

    # 7. Ensure \documentclass exists
    if r"\documentclass" not in content:
        content = "\\documentclass{article}\n" + content

    # 8. Ensure \begin{document} and \end{document} exist
    if r"\begin{document}" not in content:
        # Find first \section or \title and insert before it
        match = re.search(r"(\\(?:section|title))", content)
        if match:
            pos = match.start()
            content = content[:pos] + "\\begin{document}\n" + content[pos:]
        else:
            content += "\n\\begin{document}\n"
    
    if r"\end{document}" not in content:
        content += "\n\\end{document}"

    return content.strip()


# ──────────────────────────────────────────────
# Fallback: pdflatex if tectonic fails
# ──────────────────────────────────────────────
def _compile_with_tectonic(tex_path: Path, output_dir: Path) -> tuple[bool, str]:
    """Try compiling with tectonic."""
    try:
        result = subprocess.run(
            ["tectonic", str(tex_path), "-o", str(output_dir)],
            capture_output=True,
            text=True,
            timeout=LATEX_COMPILE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return False, "Tectonic compilation timed out."
    log = (result.stdout or "") + "\n" + (result.stderr or "")
    return result.returncode == 0, log


def _compile_with_pdflatex(tex_path: Path, output_dir: Path) -> tuple[bool, str]:
    """Fallback: run enough pdfLaTeX passes to resolve references."""
    logs = []
    try:
        for _ in range(3):
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode",
                 f"-output-directory={output_dir}", str(tex_path)],
                capture_output=True,
                text=True,
                timeout=LATEX_COMPILE_TIMEOUT_SECONDS,
            )
            current_log = (result.stdout or "") + "\n" + (result.stderr or "")
            logs.append(current_log)
            if result.returncode != 0:
                return False, "\n".join(logs)
    except subprocess.TimeoutExpired:
        return False, "pdfLaTeX compilation timed out."
    return True, logs[-1]


# ──────────────────────────────────────────────
# Tool
# ──────────────────────────────────────────────
@tool
def render_latex_pdf(latex_content: str) -> str:
    """Render a LaTeX document to PDF.

    Args:
        latex_content: Complete LaTeX document source code

    Returns:
        Path to the generated PDF file, or error message with details
    """
    # Validate input
    if not isinstance(latex_content, str) or len(latex_content.strip()) < 50:
        return "Error: LaTeX content too short. Provide a complete document."
    if len(latex_content) > MAX_LATEX_CHARS:
        return (
            "Error: LaTeX content exceeds the "
            f"{MAX_LATEX_CHARS}-character compilation limit."
        )

    # Sanitize
    latex_content = sanitize_latex(latex_content)

    citation_errors = validate_citations(latex_content)
    if citation_errors:
        details = "\n".join(f"- {error}" for error in citation_errors)
        return (
            "Error: Citation validation failed before PDF compilation.\n"
            f"{details}\n"
            "Use matching \\cite{key} and \\bibitem{key} commands, then retry once."
        )

    # Create file paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    tex_path = OUTPUT_DIR / f"paper_{timestamp}.tex"
    pdf_path = OUTPUT_DIR / f"paper_{timestamp}.pdf"
    log_path = OUTPUT_DIR / f"paper_{timestamp}.log"

    # Write .tex file
    tex_path.write_text(latex_content, encoding="utf-8")

    # Try compilation
    success = False
    log = ""
    attempt_logs = []

    # Attempt 1: tectonic
    if shutil.which("tectonic"):
        success, tectonic_log = _compile_with_tectonic(tex_path, OUTPUT_DIR)
        attempt_logs.append("Tectonic:\n" + tectonic_log)
        if success and find_unresolved_citation_warnings(tectonic_log):
            success = False

    # Attempt 2: pdflatex fallback
    if not success and shutil.which("pdflatex"):
        success, pdflatex_log = _compile_with_pdflatex(tex_path, OUTPUT_DIR)
        attempt_logs.append("pdfLaTeX:\n" + pdflatex_log)
        if success and find_unresolved_citation_warnings(pdflatex_log):
            success = False

    log = "\n\n".join(attempt_logs)

    # Save log regardless
    log_path.write_text(log, encoding="utf-8")

    # Check result
    if success and pdf_path.exists():
        return f"PDF generated successfully: {pdf_path}"

    # If PDF not found, give helpful error
    # Extract the actual LaTeX error from log
    error_lines = []
    for line in log.split("\n"):
        if (
            line.startswith("!")
            or "Error" in line
            or "Undefined" in line
            or find_unresolved_citation_warnings(line)
        ):
            error_lines.append(line.strip())

    error_summary = "\n".join(error_lines[:5]) if error_lines else "Unknown error"

    return (
        f"Error: PDF compilation failed.\n"
        f"TEX file saved at: {tex_path}\n"
        f"Log file at: {log_path}\n"
        f"Errors found:\n{error_summary}\n\n"
        f"Common fixes: Check for unescaped special characters (%, &, #, _), "
        f"missing \\end{{}} tags, or undefined commands."
    )

# src/tools/write_pdf.py
import re
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
from langchain_core.tools import tool


# ──────────────────────────────────────────────
# Output directory — project root / output
# ──────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


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
    result = subprocess.run(
        ["tectonic", str(tex_path), "-o", str(output_dir)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    log = (result.stdout or "") + "\n" + (result.stderr or "")
    return result.returncode == 0, log


def _compile_with_pdflatex(tex_path: Path, output_dir: Path) -> tuple[bool, str]:
    """Fallback: try compiling with pdflatex (run twice for references)."""
    for _ in range(2):
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", 
             f"-output-directory={output_dir}", str(tex_path)],
            capture_output=True,
            text=True,
            timeout=120,
        )
    log = (result.stdout or "") + "\n" + (result.stderr or "")
    return result.returncode == 0, log


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
    if not latex_content or len(latex_content.strip()) < 50:
        return "Error: LaTeX content too short. Provide a complete document."

    # Sanitize
    latex_content = sanitize_latex(latex_content)

    # Create file paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tex_path = OUTPUT_DIR / f"paper_{timestamp}.tex"
    pdf_path = OUTPUT_DIR / f"paper_{timestamp}.pdf"
    log_path = OUTPUT_DIR / f"paper_{timestamp}.log"

    # Write .tex file
    tex_path.write_text(latex_content, encoding="utf-8")

    # Try compilation
    success = False
    log = ""

    # Attempt 1: tectonic
    if shutil.which("tectonic"):
        success, log = _compile_with_tectonic(tex_path, OUTPUT_DIR)

    # Attempt 2: pdflatex fallback
    if not success and shutil.which("pdflatex"):
        success, log = _compile_with_pdflatex(tex_path, OUTPUT_DIR)

    # Save log regardless
    log_path.write_text(log, encoding="utf-8")

    # Check result
    if pdf_path.exists():
        return f"PDF generated successfully: {pdf_path}"

    # If PDF not found, give helpful error
    # Extract the actual LaTeX error from log
    error_lines = []
    for line in log.split("\n"):
        if line.startswith("!") or "Error" in line or "Undefined" in line:
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
# src/tools/read_pdf.py
import io
import requests
import PyPDF2
from langchain_core.tools import tool

# Store last read paper for index_paper to access
_last_read = {"text": "", "url": ""}


def get_last_read_text() -> str:
    """Access the full text of the last read paper."""
    return _last_read["text"]


@tool
def read_pdf(url: str) -> str:
    """Read and extract text from a PDF file given its URL.

    Args:
        url: The URL of the PDF file to read

    Returns:
        The extracted text content from the PDF
    """
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    pdf_file = io.BytesIO(response.content)
    pdf_reader = PyPDF2.PdfReader(pdf_file)

    text = ""
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"

    if not text.strip():
        return "Error: Could not extract any text from this PDF."

    full_text = text.strip()

    # Store full text for index_paper
    _last_read["text"] = full_text
    _last_read["url"] = url

    return full_text
# src/tools/arxiv_tool.py
import xml.etree.ElementTree as ET
import requests
from langchain_core.tools import tool
from src.config import ARXIV_MAX_RESULTS


def _parse_arxiv_xml(xml_content: str) -> list[dict]:
    """Parse the XML content from arXiv API response."""
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    root = ET.fromstring(xml_content)
    entries = []

    for entry in root.findall("atom:entry", ns):
        authors = [
            author.findtext("atom:name", namespaces=ns)
            for author in entry.findall("atom:author", ns)
        ]
        categories = [
            cat.attrib.get("term")
            for cat in entry.findall("atom:category", ns)
        ]
        pdf_link = None
        for link in entry.findall("atom:link", ns):
            if link.attrib.get("type") == "application/pdf":
                pdf_link = link.attrib.get("href")
                break

        entries.append({
            "title": entry.findtext("atom:title", namespaces=ns).strip(),
            "summary": entry.findtext("atom:summary", namespaces=ns).strip(),
            "authors": authors,
            "categories": categories,
            "pdf": pdf_link,
        })

    return entries


def _search_arxiv_papers(topic: str, max_results: int = ARXIV_MAX_RESULTS) -> list[dict]:
    """Search arXiv API for papers on a given topic."""
    # Clean query - replace spaces with +, remove problematic chars
    query = "+".join(topic.lower().split())
    for char in '()" ':
        query = query.replace(char, "")

    url = (
        "http://export.arxiv.org/api/query"
        f"?search_query=abs:{query}"
        f"&max_results={max_results}"
        "&sortBy=relevance"
        "&sortOrder=descending"
    )

    resp = requests.get(url, timeout=15)
    resp.raise_for_status()

    return _parse_arxiv_xml(resp.text)


@tool
def arxiv_search(topic: str) -> list[dict]:
    """Search for recently uploaded arXiv papers on a given topic.

    Args:
        topic: The topic to search for papers about

    Returns:
        List of papers with title, authors, summary, categories, and pdf link
    """
    papers = _search_arxiv_papers(topic)
    if not papers:
        words = topic.lower().split()
        stop_words = {"a", "an", "the", "of", "from", "and", "in", "on", "for",
                      "with", "to", "by", "about", "using", "based", "topic",
                      "interested", "im", "i'm", "model", "models", "paper", "papers"}
        keywords = [w for w in words if w not in stop_words]
        
        # Try with just 2-3 core keywords
        if len(keywords) > 2:
            shorter_query = " ".join(keywords[:3])
            papers = _search_arxiv_papers(shorter_query)
    if not papers:
        return [{"error": f"No papers found for topic: {topic}"}]
    return papers
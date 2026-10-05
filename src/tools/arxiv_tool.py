# src/tools/arxiv_tool.py
import xml.etree.ElementTree as ET

from langchain_core.tools import tool

from src.cache import get_cache
from src.config import ARXIV_MAX_RESULTS, MAX_SEARCH_QUERY_CHARS, MAX_SEARCH_RESULTS
from src.reliability import (
    ExternalServiceError,
    UserFacingError,
    request_with_retries,
    validate_int_range,
    validate_text,
)
from src.tools.pdf_sources import select_working_pdf


def _parse_arxiv_xml(xml_content: str) -> list[dict]:
    """Parse the XML content from arXiv API response."""
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as exc:
        raise ExternalServiceError("arXiv returned an invalid response.") from exc
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
            "title": (entry.findtext("atom:title", namespaces=ns) or "Unknown").strip(),
            "summary": (entry.findtext("atom:summary", namespaces=ns) or "").strip(),
            "authors": authors,
            "categories": categories,
            "pdf": pdf_link,
        })

    return entries


def _search_arxiv_papers(topic: str, max_results: int = ARXIV_MAX_RESULTS) -> list[dict]:
    """Search arXiv API for papers on a given topic."""
    topic = validate_text(topic, "Search topic", MAX_SEARCH_QUERY_CHARS)
    max_results = validate_int_range(
        max_results,
        "max_results",
        1,
        MAX_SEARCH_RESULTS,
    )
    query = " ".join(topic.lower().split())
    cache = get_cache()
    cache_identity = {
        "query": query.casefold(),
        "max_results": max_results,
        "pdf_policy_version": 2,
    }
    cached = cache.get_json("arxiv-search", cache_identity)
    if (
        isinstance(cached, list)
        and cached
        and all(isinstance(paper, dict) and "error" not in paper for paper in cached)
    ):
        return cached

    resp = request_with_retries(
        "GET",
        "https://export.arxiv.org/api/query",
        service_name="arXiv",
        params={
            "search_query": f"abs:{query}",
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        },
    )

    papers = []
    for paper in _parse_arxiv_xml(resp.text):
        working_pdf = select_working_pdf([("arXiv", paper.get("pdf"))])
        if working_pdf is None:
            continue
        paper["pdf"], paper["pdf_source"] = working_pdf
        papers.append(paper)
    if papers:
        cache.set_json("arxiv-search", cache_identity, papers)
    return papers


@tool
def arxiv_search(topic: str) -> list[dict]:
    """Search for recently uploaded arXiv papers on a given topic.

    Args:
        topic: The topic to search for papers about

    Returns:
        List of papers with title, authors, summary, categories, and pdf link
    """
    try:
        normalized_topic = validate_text(topic, "Search topic", MAX_SEARCH_QUERY_CHARS)
        papers = _search_arxiv_papers(normalized_topic)
        if not papers:
            words = normalized_topic.lower().split()
            stop_words = {
                "a", "an", "the", "of", "from", "and", "in", "on", "for",
                "with", "to", "by", "about", "using", "based", "topic",
                "interested", "im", "i'm", "model", "models", "paper", "papers",
            }
            keywords = [word for word in words if word not in stop_words]

            if len(keywords) > 2:
                papers = _search_arxiv_papers(" ".join(keywords[:3]))
        if not papers:
            return [{"error": f"No papers found for topic: {normalized_topic}"}]
        return papers
    except UserFacingError as exc:
        return [{"error": str(exc)}]

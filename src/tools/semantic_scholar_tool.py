from langchain_core.tools import tool
from src.cache import get_cache
from src.config import (
    ARXIV_MAX_RESULTS,
    MAX_SEARCH_QUERY_CHARS,
    MAX_SEARCH_RESULTS,
    get_settings,
)
from src.reliability import (
    ExternalServiceError,
    UserFacingError,
    request_with_retries,
    validate_int_range,
    validate_text,
)
from src.tools.pdf_sources import select_working_pdf


def _search_semantic_scholar_papers(
    topic: str,
    max_results: int = ARXIV_MAX_RESULTS,
) -> list[dict]:
    """Search Semantic Scholar using the shared reliability policy."""
    topic = validate_text(topic, "Search topic", MAX_SEARCH_QUERY_CHARS)
    max_results = validate_int_range(
        max_results,
        "max_results",
        1,
        MAX_SEARCH_RESULTS,
    )
    normalized_query = " ".join(topic.casefold().split())
    cache = get_cache()
    cache_identity = {
        "query": normalized_query,
        "max_results": max_results,
        "pdf_policy_version": 2,
    }
    cached = cache.get_json("semantic-scholar-search", cache_identity)
    if (
        isinstance(cached, list)
        and cached
        and all(isinstance(paper, dict) and "error" not in paper for paper in cached)
    ):
        return cached

    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": topic,
        "limit": max_results,
        "fields": "title,authors,abstract,year,url,externalIds,openAccessPdf",
    }
    headers = {}
    api_key = get_settings().get_semantic_scholar_api_key()
    if api_key:
        headers["x-api-key"] = api_key

    response = request_with_retries(
        "GET",
        url,
        service_name="Semantic Scholar",
        params=params,
        headers=headers,
    )
    try:
        data = response.json()
    except ValueError as exc:
        raise ExternalServiceError(
            "Semantic Scholar returned an invalid response."
        ) from exc

    papers = []
    for paper in data.get("data", []):
        if not paper.get("abstract"):
            continue

        arxiv_id = (paper.get("externalIds") or {}).get("ArXiv")
        open_access_pdf = paper.get("openAccessPdf") or {}
        open_access_url = (
            open_access_pdf.get("url")
            if isinstance(open_access_pdf, dict)
            else None
        )
        working_pdf = select_working_pdf([
            (
                "arXiv",
                f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else None,
            ),
            ("Semantic Scholar open access", open_access_url),
        ])
        if working_pdf is None:
            continue
        pdf_link, pdf_source = working_pdf
        authors = [
            author.get("name", "")
            for author in (paper.get("authors") or [])
        ]
        papers.append({
            "title": paper.get("title", "Unknown"),
            "authors": authors,
            "summary": paper.get("abstract", "").strip(),
            "year": paper.get("year", "N/A"),
            "pdf": pdf_link,
            "pdf_source": pdf_source,
            "paper_url": paper.get("url"),
        })
    if papers:
        cache.set_json("semantic-scholar-search", cache_identity, papers)
    return papers


@tool
def semantic_search(topic: str) -> list[dict]:
    """Search Semantic Scholar for relevant academic papers.

    IMPORTANT: Before calling this tool, rewrite the user's topic 
    into 3-5 specific academic keywords. Examples:
    - "hallucination of LLM" → search "LLM hallucination detection mitigation"
    - "how transformers work" → search "transformer self-attention mechanism"  
    - "AI for medical images" → search "medical image classification deep learning"
    
    Always use specific technical terminology, not casual language.

    Args:
        topic: Specific academic keywords to search for

    Returns:
        List of papers with title, authors, summary, year, and URL
    """
    try:
        normalized_topic = validate_text(topic, "Search topic", MAX_SEARCH_QUERY_CHARS)
        papers = _search_semantic_scholar_papers(normalized_topic)
        if not papers:
            return [{"error": f"No papers found for: {normalized_topic}"}]
        return papers
    except UserFacingError as exc:
        return [{"error": str(exc)}]

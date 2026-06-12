import time
import requests
from langchain_core.tools import tool
from src.config import ARXIV_MAX_RESULTS


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
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": topic,
        "limit": ARXIV_MAX_RESULTS,
        "fields": "title,authors,abstract,year,url,externalIds",
    }

    for attempt in range(3):
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 429:
            wait = 3 * (attempt + 1)
            time.sleep(wait)
            continue
        resp.raise_for_status()

        data = resp.json()
        papers = []

        for paper in data.get("data", []):
            if not paper.get("abstract"):
                continue

            arxiv_id = paper.get("externalIds", {}).get("ArXiv")
            pdf_link = f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else paper.get("url", "N/A")

            authors = [a.get("name", "") for a in paper.get("authors", [])]

            papers.append({
                "title": paper.get("title", "Unknown"),
                "authors": authors,
                "summary": paper.get("abstract", "").strip(),
                "year": paper.get("year", "N/A"),
                "pdf": pdf_link,
            })

        if not papers:
            return [{"error": f"No papers found for: {topic}"}]

        return papers

    return [{"error": "Semantic Scholar rate limited. Try again in a minute."}]
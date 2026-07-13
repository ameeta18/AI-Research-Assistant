"""
MCP Server for Academic Research Tools

Exposes arXiv and Semantic Scholar paper search as Model Context Protocol (MCP)
tools, making them usable in Claude Desktop or any MCP-compatible client.

"""
import os
import time
import xml.etree.ElementTree as ET
import requests
from mcp.server.fastmcp import FastMCP


mcp = FastMCP("academic-research")

MAX_RESULTS = 5



@mcp.tool()
def search_arxiv(topic: str, max_results: int = MAX_RESULTS) -> list[dict]:
    """Search arXiv for academic papers on a given topic.

    Returns recent papers with titles, authors, summaries, and PDF links.
    Best for finding papers with direct PDF access.

    Args:
        topic: The research topic to search for (e.g. "vision transformers")
        max_results: Number of papers to return (default 5)
    """
    query = "+".join(topic.lower().split())
    for char in '()"\'':
        query = query.replace(char, "")

    url = (
        "http://arxiv.org/api/query"
        f"?search_query=all:{query}"
        f"&max_results={max_results}"
        "&sortBy=relevance"
        "&sortOrder=descending"
    )

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        return [{"error": f"arXiv search failed: {str(e)}"}]

    # Parse Atom XML
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(resp.text)
    papers = []

    for entry in root.findall("atom:entry", ns):
        authors = [
            a.findtext("atom:name", namespaces=ns)
            for a in entry.findall("atom:author", ns)
        ]
        pdf_link = None
        for link in entry.findall("atom:link", ns):
            if link.attrib.get("type") == "application/pdf":
                pdf_link = link.attrib.get("href")
                break

        papers.append({
            "title": entry.findtext("atom:title", namespaces=ns).strip(),
            "summary": entry.findtext("atom:summary", namespaces=ns).strip(),
            "authors": authors,
            "pdf": pdf_link,
        })

    if not papers:
        return [{"error": f"No papers found for: {topic}"}]

    return papers



@mcp.tool()
def search_semantic_scholar(topic: str, max_results: int = MAX_RESULTS) -> list[dict]:
    """Search Semantic Scholar for papers with semantic relevance ranking.

    Returns papers with titles, authors, abstracts, year, and links.
    Best for relevance — uses semantic ranking rather than keyword matching.

    Args:
        topic: The research topic to search for
        max_results: Number of papers to return (default 5)
    """
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": topic,
        "limit": max_results,
        "fields": "title,authors,abstract,year,url,externalIds",
    }

    headers = {}
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key

    for attempt in range(3):
        time.sleep(1)  # respect shared rate limit
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=30)
        except Exception as e:
            return [{"error": f"Semantic Scholar request failed: {str(e)}"}]

        if resp.status_code == 429:
            time.sleep(3 * (attempt + 1))
            continue
        resp.raise_for_status()

        data = resp.json()
        papers = []

        for paper in data.get("data", []):
            if not paper.get("abstract"):
                continue

            arxiv_id = paper.get("externalIds", {}).get("ArXiv")
            pdf_link = (
                f"https://arxiv.org/pdf/{arxiv_id}"
                if arxiv_id else paper.get("url", "N/A")
            )
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



if __name__ == "__main__":
    mcp.run()
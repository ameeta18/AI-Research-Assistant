"""
MCP Server for Academic Research Tools

Exposes arXiv and Semantic Scholar paper search as Model Context Protocol (MCP)
tools, making them usable in Claude Desktop or any MCP-compatible client.

"""
from mcp.server.fastmcp import FastMCP

from src.config import get_settings
from src.reliability import UserFacingError
from src.tools.arxiv_tool import _search_arxiv_papers
from src.tools.semantic_scholar_tool import _search_semantic_scholar_papers


mcp = FastMCP("academic-research")

MAX_RESULTS = get_settings().arxiv_max_results



@mcp.tool()
def search_arxiv(topic: str, max_results: int = MAX_RESULTS) -> list[dict]:
    """Search arXiv for academic papers on a given topic.

    Returns recent papers with titles, authors, summaries, and PDF links.
    Best for finding papers with direct PDF access.

    Args:
        topic: The research topic to search for (e.g. "vision transformers")
        max_results: Number of papers to return (default 5)
    """
    try:
        papers = _search_arxiv_papers(topic, max_results)
        return papers or [{"error": f"No papers found for: {topic.strip()}"}]
    except UserFacingError as exc:
        return [{"error": str(exc)}]



@mcp.tool()
def search_semantic_scholar(topic: str, max_results: int = MAX_RESULTS) -> list[dict]:
    """Search Semantic Scholar for papers with semantic relevance ranking.

    Returns papers with titles, authors, abstracts, year, and verified PDF links.
    Best for relevance — uses semantic ranking rather than keyword matching.

    Args:
        topic: The research topic to search for
        max_results: Number of papers to return (default 5)
    """
    try:
        papers = _search_semantic_scholar_papers(topic, max_results)
        return papers or [{"error": f"No papers found for: {topic.strip()}"}]
    except UserFacingError as exc:
        return [{"error": str(exc)}]



if __name__ == "__main__":
    mcp.run()

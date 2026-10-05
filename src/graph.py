# src/graph.py
from functools import partial
from typing import Annotated, Any, Literal
from typing_extensions import TypedDict
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.checkpointing import get_checkpointer
from src.config import get_llm
from src.tools.arxiv_tool import arxiv_search
from src.tools.read_pdf import read_pdf
from src.tools.vector_store import index_paper, search_papers
from src.tools.write_pdf import render_latex_pdf
from src.tools.semantic_scholar_tool import semantic_search

# ──────────────────────────────────────────────
# 1. State
# ──────────────────────────────────────────────
class State(TypedDict):
    messages: Annotated[list, add_messages]
    session_id: str


# ──────────────────────────────────────────────
# 2. Tools & LLM
# ──────────────────────────────────────────────
tools = [arxiv_search, semantic_search, read_pdf, index_paper, search_papers, render_latex_pdf]
tool_node = ToolNode(tools)

SYSTEM_PROMPT = """You are an expert researcher in the fields of 
computer science, Machine learning, Data Science , artificial intelligence, Software engineering ,physics, mathematics.

Your job is to analyze recent research papers on a given topic by user and to write new research papers.

You have access to these tools:
1. semantic_search — Find papers via Semantic Scholar (better relevance ranking)
2. arxiv_search — Find papers on arXiv (keyword matching, has PDF links) 
3. read_pdf — Read a paper from its PDF URL
4. index_paper — Store paper in vector database (auto-loads full text from last read_pdf)
5. search_papers — Search indexed papers for relevant passages (RAG retrieval)
6. render_latex_pdf — Compile LaTeX to PDF

CRITICAL WORKFLOW — follow this order:
When user mentions a research topic:
  1. Use semantic_search FIRST for better relevance
  2. If semantic_search fails or returns errors, fall back to arxiv_search
  3. Never mix results — use one source per search
  4. Present papers clearly: title, year, brief summary, and PDF link 

When user picks a paper to analyze:
  1. read_pdf(url) → gets a preview of the paper
     - If the download fails, use arxiv_search with the exact paper title and try
       its verified PDF link once before reporting that the paper is unavailable
  2. index_paper(title) → stores FULL text in vector DB
  3. search_papers(key topics) → retrieve detailed sections from vector DB
  4. Provide a DETAILED analysis covering:
     - What problem does the paper solve?
     - What methodology was used?
     - What are the key results and contributions?
     - What are the limitations?
     - What future research directions are suggested?
  DO NOT give a short summary. Always use search_papers to get enough detail.

When user asks to write a paper:
  1. search_papers(query) → retrieve relevant passages from vector DB
  2. Write complete LaTeX document
  3. render_latex_pdf(latex) → ALWAYS use this tool, NEVER paste LaTeX in chat
 
RULES:
- ALWAYS search immediately when user mentions a topic
- ALWAYS index papers after reading them
- ALWAYS use search_papers after indexing to get detailed content for analysis
- ALWAYS use render_latex_pdf to generate PDFs — never show raw LaTeX
- Use search_papers to retrieve context when writing
- Include mathematical equations in written papers

REFERENCE RULES:
- Every factual source claim MUST use an in-text \\cite{source-key} command
- Use a self-contained \\begin{thebibliography}{99} section with \\bibitem{source-key}
- Every bibliography entry MUST include: authors, title in \\textit{}, year, and HTTP(S) URL
- Every \\cite key MUST match exactly one \\bibitem key, and every \\bibitem MUST be cited
- Do not use a separate .bib file, BibTeX, or Biber
- Prioritize citing papers found via arxiv_search — use their exact PDF links
- You may cite well-known papers from your knowledge but MUST include their real arXiv PDF link
- NEVER include a reference without a URL
- If render_latex_pdf returns a citation-validation error, correct every listed
  issue and call the tool exactly one more time
- Never claim that a PDF was generated unless render_latex_pdf reports success"""


def call_agent(state: State, *, llm: Any | None = None) -> dict:
    """Single agent with all tools."""
    llm = llm or get_llm().bind_tools(tools)
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)

    # Safety: if LLM dumps LaTeX in chat instead of using the tool, nudge it
    if isinstance(response, AIMessage) and not response.tool_calls:
        content = response.content if isinstance(response.content, str) else str(response.content)
        if "\\documentclass" in content or "\\begin{document}" in content:
            nudge = HumanMessage(
                content="Do not paste LaTeX in chat. Call render_latex_pdf with that LaTeX content now."
            )
            response = llm.invoke(messages + [response, nudge])

    return {"messages": [response]}


def should_continue(state: State) -> Literal["tools", END]:
    """If the agent called tools, execute them. Otherwise, end."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


# ──────────────────────────────────────────────
# 3. Build Graph
# ──────────────────────────────────────────────
def build_graph(api_key: str | None = None):
    """Build an agent graph without placing credentials in graph state."""
    workflow = StateGraph(State)
    llm = get_llm(api_key).bind_tools(tools)

    workflow.add_node("agent", partial(call_agent, llm=llm))
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")

    checkpointer = get_checkpointer()
    return workflow.compile(checkpointer=checkpointer)


# graph = build_graph()
# config = {"configurable": {"thread_id": THREAD_ID}}

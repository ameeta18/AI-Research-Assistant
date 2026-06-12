# src/graph.py
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, SystemMessage

from src.config import get_llm, THREAD_ID
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


# ──────────────────────────────────────────────
# 2. Tools & LLM
# ──────────────────────────────────────────────
tools = [arxiv_search, semantic_search, read_pdf, index_paper, search_papers, render_latex_pdf]
tool_node = ToolNode(tools)

SYSTEM_PROMPT = """You are an expert researcher in the fields of 
computer science, Machine learning, Data Science , artificial intelligence, Software engineering ,physics, mathematics.

Your job is to analyze recent research papers on a given topic by user and to write new research papers.

You have access to these tools:
1. arxiv_search — Find papers on arXiv (keyword matching, has PDF links)
2. semantic_search — Find papers via Semantic Scholar (better relevance ranking)
3. read_pdf — Read a paper from its PDF URL
4. index_paper — Store paper in vector database (auto-loads full text from last read_pdf)
5. search_papers — Search indexed papers for relevant passages (RAG retrieval)
6. render_latex_pdf — Compile LaTeX to PDF

CRITICAL WORKFLOW — follow this order:
When user mentions a research topic:
  1. Use semantic_search FIRST for better relevance
  2. If semantic_search fails or returns errors, fall back to arxiv_search
  3. Present papers clearly: title, authors, year, brief summary, and PDF link 

When user picks a paper to analyze:
  1. read_pdf(url) → gets a preview of the paper
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
- Every reference MUST include: authors, title, year, and arXiv PDF link
- Format: [1] Author et al. (Year). Title. URL: https://arxiv.org/pdf/XXXX
- Prioritize citing papers found via arxiv_search — use their exact PDF links
- You may cite well-known papers from your knowledge but MUST include their real arXiv PDF link
- NEVER include a reference without a URL"""


def call_agent(state: State) -> dict:
    """Single agent with all tools."""
    llm = get_llm().bind_tools(tools)
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
def build_graph():
    workflow = StateGraph(State)

    workflow.add_node("agent", call_agent)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


# graph = build_graph()
# config = {"configurable": {"thread_id": THREAD_ID}}
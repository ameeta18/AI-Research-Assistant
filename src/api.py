# src/api.py
import os
import uuid
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import AIMessage, HumanMessage

# ──────────────────────────────────────────────
# App Setup
# ──────────────────────────────────────────────
app = FastAPI(
    title="AI Research Assistant API",
    description="RAG-powered research paper discovery, analysis, and generation",
    version="1.0.0",
)

# Allow frontend requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# Request/Response Models
# ──────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    api_key: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    tools_used: list[str]


# ──────────────────────────────────────────────
# Graph Setup (lazy — after API key is set)
# ──────────────────────────────────────────────
_graph = None


def get_graph(api_key: str):
    """Initialize the graph with the provided API key."""
    global _graph
    os.environ["GOOGLE_API_KEY"] = api_key

    from src.tools.vector_store import init_embeddings
    init_embeddings(api_key)

    if _graph is None:
        from src.graph import build_graph
        _graph = build_graph()

    return _graph


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "AI Research Assistant API",
        "status": "running",
        "endpoints": ["/chat", "/health"],
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Send a message to the research assistant.

    The agent will search papers, analyze them, or write papers based on the message.
    """
    # Resolve API key — from request or environment
    api_key = request.api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="API key required")

    # Get or create session
    session_id = request.session_id or str(uuid.uuid4())

    try:
        graph = get_graph(api_key)
        config = {"configurable": {"thread_id": session_id}}

        graph_input = {
            "messages": [HumanMessage(content=request.message)],
        }

        full_response = ""
        tools_used = []

        for event in graph.stream(graph_input, config, stream_mode="values"):
            last_msg = event["messages"][-1]

            # Track tools
            if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                for tc in last_msg.tool_calls:
                    tools_used.append(tc["name"])

            # Capture response
            if isinstance(last_msg, AIMessage) and last_msg.content:
                content = last_msg.content
                if isinstance(content, list):
                    content = " ".join(
                        block["text"] for block in content
                        if isinstance(block, dict) and "text" in block
                    )
                if content and content.strip():
                    full_response = content

        return ChatResponse(
            response=full_response,
            session_id=session_id,
            tools_used=list(set(tools_used)),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
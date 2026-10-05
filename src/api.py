# src/api.py
import uuid
from contextlib import asynccontextmanager
import time
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator
from langchain_core.messages import AIMessage, HumanMessage

from src.config import MAX_CHAT_MESSAGE_CHARS, get_settings
from src.observability import (
    configure_logging,
    get_logger,
    log_context,
    trace_research_run,
)
from src.reliability import ExternalServiceError, UserFacingError
from src.session_store import SessionCapacityError
from src.persistence import close_persistence, get_persistence
from src.checkpointing import close_checkpointer


_graph = None
logger = get_logger("api")


def get_graph():
    """Initialize the graph from validated server configuration."""
    global _graph

    from src.tools.vector_store import init_embeddings
    init_embeddings()

    if _graph is None:
        from src.graph import build_graph
        _graph = build_graph()

    return _graph


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Validate required configuration before accepting traffic."""
    configure_logging()
    get_settings().require_google_api_key()
    persistence = get_persistence()
    if not persistence.healthcheck():
        raise RuntimeError("PostgreSQL health check failed")
    get_graph()
    logger.info("application_started")
    yield
    close_checkpointer()
    close_persistence()
    logger.info("application_stopped")

# ──────────────────────────────────────────────
# App Setup
# ──────────────────────────────────────────────
app = FastAPI(
    title="AI Research Assistant API",
    description="RAG-powered research paper discovery, analysis, and generation",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow frontend requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_observability(request: Request, call_next):
    """Correlate and time requests without logging request bodies or query values."""
    request_id = str(uuid.uuid4())
    started_at = time.perf_counter()

    with log_context(request_id=request_id):
        try:
            response = await call_next(request)
        except Exception as exc:
            logger.error(
                "http_request_failed",
                extra={
                    "event_data": {
                        "method": request.method,
                        "path": request.url.path,
                        "duration_ms": round(
                            (time.perf_counter() - started_at) * 1000,
                            2,
                        ),
                        "exception_type": type(exc).__name__,
                    }
                },
            )
            raise

        response.headers["X-Request-ID"] = request_id
        logger.info(
            "http_request_completed",
            extra={
                "event_data": {
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": round(
                        (time.perf_counter() - started_at) * 1000,
                        2,
                    ),
                }
            },
        )
        return response

# ──────────────────────────────────────────────
# Request/Response Models
# ──────────────────────────────────────────────
class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=MAX_CHAT_MESSAGE_CHARS)
    session_id: Optional[uuid.UUID] = None

    @field_validator("message")
    @classmethod
    def message_must_contain_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("message cannot be blank")
        return normalized


class ChatResponse(BaseModel):
    response: str
    session_id: str
    tools_used: list[str]


# ──────────────────────────────────────────────
# Graph Setup (initialized during application startup)
# ──────────────────────────────────────────────
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
    database_healthy = get_persistence().healthcheck()
    if not database_healthy:
        raise HTTPException(status_code=503, detail="Database is unavailable")
    return {"status": "healthy", "database": "healthy" if get_persistence().enabled else "disabled"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Send a message to the research assistant.

    The agent will search papers, analyze them, or write papers based on the message.
    """
    # Get or create session
    session_id = str(request.session_id or uuid.uuid4())

    started_at = time.perf_counter()
    with log_context(session_id=session_id):
        logger.info("research_request_started")
        try:
            persistence = get_persistence()
            persistence.ensure_session(session_id)
            persistence.save_message(session_id, "user", request.message)
            graph = get_graph()
            config = {"configurable": {"thread_id": session_id}}

            graph_input = {
                "messages": [HumanMessage(content=request.message)],
                "session_id": session_id,
            }

            full_response = ""
            tools_used = []

            with trace_research_run(interface="api", session_id=session_id):
                for event in graph.stream(graph_input, config, stream_mode="values"):
                    last_msg = event["messages"][-1]

                    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                        for tool_call in last_msg.tool_calls:
                            tools_used.append(tool_call["name"])

                    if isinstance(last_msg, AIMessage) and last_msg.content:
                        content = last_msg.content
                        if isinstance(content, list):
                            content = " ".join(
                                block["text"] for block in content
                                if isinstance(block, dict) and "text" in block
                            )
                        if content and content.strip():
                            full_response = content

            unique_tools = sorted(set(tools_used))
            if full_response:
                persistence.save_message(session_id, "assistant", full_response)
            logger.info(
                "research_request_completed",
                extra={
                    "event_data": {
                        "duration_ms": round(
                            (time.perf_counter() - started_at) * 1000,
                            2,
                        ),
                        "tools_used": unique_tools,
                    }
                },
            )
            return ChatResponse(
                response=full_response,
                session_id=session_id,
                tools_used=unique_tools,
            )

        except ExternalServiceError as exc:
            logger.warning("research_request_external_service_failed")
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except SessionCapacityError as exc:
            logger.warning("research_request_capacity_reached")
            raise HTTPException(
                status_code=503,
                detail="The service is at session capacity. Please try again shortly.",
            ) from exc
        except UserFacingError as exc:
            logger.warning("research_request_validation_failed")
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.error(
                "research_request_failed",
                extra={"event_data": {"exception_type": type(exc).__name__}},
            )
            raise HTTPException(
                status_code=500,
                detail="The research request could not be completed.",
            ) from exc

# src/app.py
import hashlib
import streamlit as st
import time
import uuid
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from src.observability import (
    configure_logging,
    get_logger,
    log_context,
    trace_research_run,
)
from src.reliability import UserFacingError
from src.persistence import get_persistence


configure_logging()
logger = get_logger("streamlit")

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 AI Research Assistant")
st.caption("Search → Analyze → Write research papers with proper citations")

# ──────────────────────────────────────────────
# Server Configuration
# ──────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

with st.sidebar:
    st.subheader("🔐 Your Gemini Access")
    ui_api_key = st.text_input(
        "Gemini API key",
        type="password",
        key="gemini_api_key_input",
        help=(
            "Required for Streamlit. The key is used only by your in-memory "
            "session and is never replaced with a server-side key."
        ),
    )

effective_api_key = ui_api_key.strip()
if not effective_api_key:
    st.info("Enter your own Gemini API key in the sidebar to start the app.")
    st.stop()

# ──────────────────────────────────────────────
# Initialize from validated server configuration
# ──────────────────────────────────────────────
from src.graph import build_graph
from src.session_store import get_session_store
from src.tools.vector_store import init_embeddings

persistence = get_persistence()
persistence.ensure_session(st.session_state.thread_id)

credential_fingerprint = hashlib.sha256(
    effective_api_key.encode("utf-8")
).hexdigest()
if st.session_state.get("credential_fingerprint") != credential_fingerprint:
    if "credential_fingerprint" in st.session_state:
        get_session_store().delete(st.session_state.thread_id)
    init_embeddings(
        api_key=effective_api_key,
        session_id=st.session_state.thread_id,
    )
    st.session_state.graph = build_graph(api_key=effective_api_key)
    st.session_state.credential_fingerprint = credential_fingerprint

del effective_api_key

config = {"configurable": {"thread_id": st.session_state.thread_id}}
graph = st.session_state.graph

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("🤖 How to Use")
    st.markdown("""
    1. **Tell me a research topic** → Finds papers via Semantic Scholar and arXiv
    2. **Pick a paper to analyze** → Reads & indexes it in Vector DB
    3. **Ask to write a paper** → Writes using RAG and generates PDF with proper citations
    """)
    st.divider()
    st.subheader("🛠️ Tools")
    st.markdown("""
    - 🔍 `arxiv_search` — Find papers (keyword)
    - 🔎 `semantic_search` — Find papers (semantic ranking)
    - 📄 `read_pdf` — Read paper content
    - 💾 `index_paper` — Store in FAISS
    - 🗂️ `search_papers` — RAG retrieval
    - 📝 `render_latex_pdf` — Generate PDF
    """)
# ──────────────────────────────────────────────
# Chat History
# ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": message.role, "content": message.content}
        for message in persistence.list_messages(st.session_state.thread_id)
        if message.role in {"user", "assistant"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# ──────────────────────────────────────────────
# Tool icon mapping
# ──────────────────────────────────────────────
TOOL_ICONS = {
    "arxiv_search": "🔍",
    "semantic_search": "🔎",
    "read_pdf": "📄",
    "index_paper": "💾",
    "search_papers": "🗂️",
    "render_latex_pdf": "📝",
}

# ──────────────────────────────────────────────
# Chat Input & Processing
# ──────────────────────────────────────────────
user_input = st.chat_input("What research topic would you like to explore?")

if user_input:
    request_id = str(uuid.uuid4())
    started_at = time.perf_counter()
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    persistence.save_message(st.session_state.thread_id, "user", user_input)

    graph_input = {
        "messages": [HumanMessage(content=user_input)],
        "session_id": st.session_state.thread_id,
    }

    def observed_events():
        with (
            log_context(
                request_id=request_id,
                session_id=st.session_state.thread_id,
            ),
            trace_research_run(
                interface="streamlit",
                session_id=st.session_state.thread_id,
            ),
        ):
            logger.info("research_request_started")
            try:
                yield from graph.stream(graph_input, config, stream_mode="values")
            except Exception as exc:
                logger.error(
                    "research_request_failed",
                    extra={"event_data": {"exception_type": type(exc).__name__}},
                )
                raise
            logger.info(
                "research_request_completed",
                extra={
                    "event_data": {
                        "duration_ms": round(
                            (time.perf_counter() - started_at) * 1000,
                            2,
                        )
                    }
                },
            )

    with st.chat_message("assistant"):
        status = st.empty()
        response_area = st.empty()
        full_response = ""

        try:
            for event in observed_events():
                last_msg = event["messages"][-1]

                if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                    for tc in last_msg.tool_calls:
                        icon = TOOL_ICONS.get(tc["name"], "🔧")
                        status.caption(f"{icon} Calling `{tc['name']}`...")

                if isinstance(last_msg, AIMessage) and last_msg.content:
                    content = last_msg.content
                    if isinstance(content, list):
                        content = " ".join(
                            block["text"] for block in content
                            if isinstance(block, dict) and "text" in block
                        )
                    if content and content.strip():
                        full_response = content
                        response_area.markdown(full_response)

            status.empty()

        except UserFacingError as exc:
            status.empty()
            full_response = f"Error: {exc}"
            st.error(full_response)
        except Exception:
            status.empty()
            full_response = "Error: The research request could not be completed."
            st.error(full_response)

    if full_response:
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        persistence.save_message(
            st.session_state.thread_id,
            "assistant",
            full_response,
        )
    st.rerun()

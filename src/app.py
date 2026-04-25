# src/app.py
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from src.graph import graph
import uuid

# Generate unique thread per browser session
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

config = {"configurable": {"thread_id": st.session_state.thread_id}}
# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 AI Research Assistant")
st.caption("Search → Analyze → Write research papers ")

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("🤖 How to Use")
    st.markdown("""
    1. **Tell me a research topic** → Finds papers on arXiv
    2. **Pick a paper to analyze** → Reads & indexes it in Vector DB
    3. **Ask to write a paper** → Writes using RAG and generates PDF
    """)
    st.divider()
    st.subheader("🛠️ Tools")
    st.markdown("""
    - 🔍 `arxiv_search` — Find papers
    - 📄 `read_pdf` — Read paper content
    - 💾 `index_paper` — Store in FAISS
    - 🗂️ `search_papers` — RAG retrieval
    - 📝 `render_latex_pdf` — Generate PDF
    """)

# ──────────────────────────────────────────────
# Chat History
# ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# ──────────────────────────────────────────────
# Tool icon mapping
# ──────────────────────────────────────────────
TOOL_ICONS = {
    "arxiv_search": "🔍",
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
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    graph_input = {
        "messages": [
            HumanMessage(content=m["content"]) if m["role"] == "user"
            else AIMessage(content=m["content"])
            for m in st.session_state.messages
        ],
    }

    with st.chat_message("assistant"):
        status = st.empty()
        response_area = st.empty()
        full_response = ""

        try:
            for event in graph.stream(graph_input, config, stream_mode="values"):
                last_msg = event["messages"][-1]

                # Show tool calls as status
                if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                    for tc in last_msg.tool_calls:
                        icon = TOOL_ICONS.get(tc["name"], "🔧")
                        status.caption(f"{icon} Calling `{tc['name']}`...")

                # Display AI response                
                
                if isinstance(last_msg, AIMessage) and last_msg.content:
                    content = last_msg.content
                    # Gemini returns list of content blocks — extract text only
                    if isinstance(content, list):
                        content = " ".join(
                            block["text"] for block in content
                            if isinstance(block, dict) and "text" in block
                        )
                    if content and content.strip():
                        full_response = content
                        response_area.markdown(full_response)

            status.empty()

        except Exception as e:
            status.empty()
            full_response = f"Error: {str(e)}"
            st.error(full_response)

    if full_response:
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.rerun()
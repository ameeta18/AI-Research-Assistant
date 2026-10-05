# src/tools/vector_store.py
from typing import Annotated

from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langgraph.prebuilt import InjectedState

from src.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    MAX_PAPER_TITLE_CHARS,
    MAX_PDF_TEXT_CHARS,
    MAX_RETRIEVAL_K,
    MAX_SEARCH_QUERY_CHARS,
    get_settings,
)
from src.reliability import (
    InputValidationError,
    UserFacingError,
    validate_int_range,
    validate_text,
)
from src.persistence import get_persistence
from src.session_store import get_session_store
from src.tools.read_pdf import get_last_read_text

# The embeddings client is configuration, not user data, so it is safe to share.
_embeddings = None

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def init_embeddings(
    api_key: str | None = None,
    session_id: str | None = None,
):
    """Initialize server-wide or session-scoped embedding credentials."""
    global _embeddings
    if session_id is None and _embeddings is not None:
        return

    resolved_api_key = (
        api_key.strip() if api_key else get_settings().require_google_api_key()
    )
    if not resolved_api_key:
        raise ValueError("GOOGLE_API_KEY is not configured")
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=resolved_api_key,
    )
    if session_id is None:
        _embeddings = embeddings
        return

    resources = get_session_store().get_or_create(session_id)
    with resources.lock:
        resources.embeddings = embeddings


def get_embeddings():
    """Return the shared embeddings client, if initialized."""
    return _embeddings


def get_vectorstore(session_id: str) -> FAISS | None:
    """Return the FAISS index owned by one session."""
    resources = get_session_store().get(session_id)
    if resources is None:
        return None
    with resources.lock:
        return resources.vectorstore


def index_paper_for_session(title: str, session_id: str, text: str = "") -> str:
    """Index a paper into the FAISS store owned by one session."""
    title = validate_text(title, "Paper title", MAX_PAPER_TITLE_CHARS)
    resources = get_session_store().get_or_create(session_id)
    with resources.lock:
        embeddings = resources.embeddings or _embeddings
    if embeddings is None:
        return "Error: Embeddings are not initialized. Check the server configuration."

    source_url = ""

    if not text or len(text.strip()) < 100:
        text = get_last_read_text(session_id)
        if not text:
            return "Error: No text to index. Read a paper with read_pdf first."
        with resources.lock:
            source_url = resources.last_read_url
    elif len(text) > MAX_PDF_TEXT_CHARS:
        raise InputValidationError(
            f"Paper text exceeds the {MAX_PDF_TEXT_CHARS}-character indexing limit."
        )

    chunks = _splitter.split_text(text)
    metadatas = [
        {
            "title": title,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "session_id": session_id,
            "source_url": source_url,
        }
        for i in range(len(chunks))
    ]

    with resources.lock:
        if resources.vectorstore is None:
            resources.vectorstore = FAISS.from_texts(
                chunks,
                embeddings,
                metadatas=metadatas,
            )
        else:
            resources.vectorstore.add_texts(chunks, metadatas=metadatas)

    get_persistence().save_paper_chunks(
        session_id=session_id,
        title=title,
        source_url=source_url,
        chunks=chunks,
    )

    return f"Successfully indexed '{title}' — {len(chunks)} chunks stored in vector database."


@tool
def index_paper(
    title: str,
    session_id: Annotated[str, InjectedState("session_id")],
    text: str = "",
) -> str:
    """Index a research paper into the vector store for later semantic search.

    Args:
        title: The paper title
        text: The paper text (optional — auto-loaded from last read_pdf if empty)

    Returns:
        Confirmation with number of chunks indexed
    """
    try:
        return index_paper_for_session(title, session_id, text)
    except UserFacingError as exc:
        return f"Error: {exc}"


def search_papers_for_session(query: str, session_id: str, k: int = 5) -> str:
    """Search only the FAISS index owned by one session."""
    query = validate_text(query, "Search query", MAX_SEARCH_QUERY_CHARS)
    k = validate_int_range(k, "k", 1, MAX_RETRIEVAL_K)
    resources = get_session_store().get(session_id)
    if resources is None:
        resources = get_session_store().get_or_create(session_id)

    with resources.lock:
        if resources.vectorstore is None:
            stored_chunks = get_persistence().load_paper_chunks(session_id)
            embeddings = resources.embeddings or _embeddings
            if stored_chunks and embeddings is not None:
                resources.vectorstore = FAISS.from_texts(
                    [chunk.content for chunk in stored_chunks],
                    embeddings,
                    metadatas=[
                        {
                            "title": chunk.title,
                            "chunk_index": chunk.chunk_index,
                            "total_chunks": chunk.total_chunks,
                            "session_id": session_id,
                            "source_url": chunk.source_url,
                        }
                        for chunk in stored_chunks
                    ],
                )
            else:
                return "No papers indexed yet. Use read_pdf then index_paper first."
        results = resources.vectorstore.similarity_search_with_score(query, k=k)

    if not results:
        return f"No relevant passages found for: {query}"

    output = []
    for doc, score in results:
        title = doc.metadata.get("title", "Unknown")
        chunk_idx = doc.metadata.get("chunk_index", "?")
        total = doc.metadata.get("total_chunks", "?")
        output.append(
            f"📄 [{title}] (chunk {chunk_idx}/{total}, relevance: {score:.3f})\n"
            f"{doc.page_content}"
        )

    return "\n\n---\n\n".join(output)


@tool
def search_papers(
    query: str,
    session_id: Annotated[str, InjectedState("session_id")],
    k: int = 5,
) -> str:
    """Search indexed papers for passages relevant to a query.

    Args:
        query: What to search for
        k: Number of relevant passages to return (default: 5)

    Returns:
        Relevant passages with their source paper titles
    """
    try:
        return search_papers_for_session(query, session_id, k)
    except UserFacingError as exc:
        return f"Error: {exc}"

# src/tools/vector_store.py
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.tools.read_pdf import get_last_read_text
# ─── Module-level state ───
_vectorstore: FAISS | None = None

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY. Set it as an environment variable.")
_embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=GOOGLE_API_KEY,
)



def get_vectorstore() -> FAISS | None:
    """Access the current vector store (for debugging/testing)."""
    return _vectorstore


@tool
def index_paper(title: str, text: str = "") -> str:
    """Index a research paper into the vector store for later semantic search.

    Call this AFTER reading a paper with read_pdf to store it for retrieval.
    If text is empty or short, automatically uses the full text from the last read_pdf call.

    Args:
        title: The paper title (used as metadata for source tracking)
        text: The paper text (optional — auto-loaded from last read_pdf if empty)

    Returns:
        Confirmation with number of chunks indexed
    """
    global _vectorstore

    # Auto-load full text if not provided or too short
    if not text or len(text.strip()) < 100:
        text = get_last_read_text()
        if not text:
            return "Error: No text to index. Read a paper with read_pdf first."

    chunks = _splitter.split_text(text)
    metadatas = [
        {"title": title, "chunk_index": i, "total_chunks": len(chunks)}
        for i in range(len(chunks))
    ]

    if _vectorstore is None:
        _vectorstore = FAISS.from_texts(chunks, _embeddings, metadatas=metadatas)
    else:
        _vectorstore.add_texts(chunks, metadatas=metadatas)

    return f"Successfully indexed '{title}' — {len(chunks)} chunks stored in vector database."

@tool
def search_papers(query: str, k: int = 5) -> str:
    """Search indexed papers for passages relevant to a query.

    Use this to find specific information across all indexed papers
    instead of relying on the full paper text in conversation history.

    Args:
        query: What to search for (e.g., 'methodology for anomaly detection')
        k: Number of relevant passages to return (default: 5)

    Returns:
        Relevant passages with their source paper titles
    """
    if _vectorstore is None:
        return "No papers indexed yet. Use read_pdf then index_paper first."

    results = _vectorstore.similarity_search_with_score(query, k=k)

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
# src/tools/vector_store.py
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL
from src.tools.read_pdf import get_last_read_text

# ─── Module-level state ───
_vectorstore: FAISS | None = None
_embeddings = None

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def init_embeddings(api_key: str):
    """Initialize embeddings with user's API key."""
    global _embeddings
    _embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
    )


def get_vectorstore() -> FAISS | None:
    return _vectorstore


@tool
def index_paper(title: str, text: str = "") -> str:
    """Index a research paper into the vector store for later semantic search.

    Args:
        title: The paper title
        text: The paper text (optional — auto-loaded from last read_pdf if empty)

    Returns:
        Confirmation with number of chunks indexed
    """
    global _vectorstore

    if _embeddings is None:
        return "Error: API key not set. Please enter your Gemini API key."

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

    Args:
        query: What to search for
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
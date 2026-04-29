# src/config.py
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# ─── Model Configuration ───
LLM_MODEL = "gemini-2.5-flash"
LLM_TEMPERATURE = 0.3

def get_llm(api_key: str = None) -> ChatGoogleGenerativeAI:
    """Create a fresh LLM instance."""
    key = api_key or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ValueError("Google API key not provided")
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        google_api_key=key,
    )

# ─── Vector Store Settings ───
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "models/gemini-embedding-001"

# ─── ArXiv Settings ───
ARXIV_MAX_RESULTS = 5

# ─── Graph Settings ───
THREAD_ID = "research-session-001"
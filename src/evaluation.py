# src/evaluation.py
"""
Evaluation metrics for the Multi-Agent Research Assistant.
Measures retrieval quality, generation faithfulness, and tool reliability.
"""
import time
import json
from datetime import datetime
from pathlib import Path
from langchain_core.tools import tool
from src.tools.vector_store import get_vectorstore


# ──────────────────────────────────────────────
# 1. Session Metrics Tracker
# ──────────────────────────────────────────────
class SessionMetrics:
    """Track all metrics across a research session."""

    def __init__(self):
        self.tool_calls = []        # Every tool call with timing
        self.retrievals = []        # search_papers results with relevance
        self.papers_indexed = []    # Papers stored in vector DB
        self.generation_checks = [] # Faithfulness checks on generated content
        self.session_start = datetime.now()

    def log_tool_call(self, tool_name: str, success: bool, duration: float, details: str = ""):
        self.tool_calls.append({
            "tool": tool_name,
            "success": success,
            "duration_seconds": round(duration, 2),
            "details": details,
            "timestamp": datetime.now().isoformat(),
        })

    def log_retrieval(self, query: str, results: list[dict]):
        self.retrievals.append({
            "query": query,
            "num_results": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat(),
        })

    def log_paper_indexed(self, title: str, num_chunks: int):
        self.papers_indexed.append({
            "title": title,
            "num_chunks": num_chunks,
            "timestamp": datetime.now().isoformat(),
        })

    def log_generation_check(self, check_type: str, score: float, details: str = ""):
        self.generation_checks.append({
            "check_type": check_type,
            "score": score,
            "details": details,
            "timestamp": datetime.now().isoformat(),
        })

    # ──────────────────────────────────────────
    # Computed Metrics
    # ──────────────────────────────────────────
    def tool_success_rate(self) -> dict:
        """Percentage of successful tool calls per tool."""
        if not self.tool_calls:
            return {}
        tools = {}
        for call in self.tool_calls:
            name = call["tool"]
            if name not in tools:
                tools[name] = {"success": 0, "total": 0}
            tools[name]["total"] += 1
            if call["success"]:
                tools[name]["success"] += 1

        return {
            name: {
                "success_rate": round(data["success"] / data["total"] * 100, 1),
                "total_calls": data["total"],
                "avg_duration": round(
                    sum(c["duration_seconds"] for c in self.tool_calls if c["tool"] == name)
                    / data["total"], 2
                ),
            }
            for name, data in tools.items()
        }

    def avg_retrieval_relevance(self) -> float:
        """Average relevance score across all retrievals."""
        all_scores = []
        for r in self.retrievals:
            for result in r["results"]:
                if "relevance_score" in result:
                    all_scores.append(result["relevance_score"])
        return round(sum(all_scores) / len(all_scores), 3) if all_scores else 0.0

    def retrieval_hit_rate(self) -> float:
        """Percentage of retrievals that returned at least one relevant result (score < 1.5)."""
        if not self.retrievals:
            return 0.0
        hits = 0
        for r in self.retrievals:
            for result in r["results"]:
                if result.get("relevance_score", 999) < 1.5:
                    hits += 1
                    break
        return round(hits / len(self.retrievals) * 100, 1)

    def summary(self) -> dict:
        """Full evaluation summary."""
        duration = (datetime.now() - self.session_start).total_seconds()
        return {
            "session_duration_seconds": round(duration, 1),
            "papers_indexed": len(self.papers_indexed),
            "total_chunks_stored": sum(p["num_chunks"] for p in self.papers_indexed),
            "tool_performance": self.tool_success_rate(),
            "retrieval_metrics": {
                "total_queries": len(self.retrievals),
                "avg_relevance_score": self.avg_retrieval_relevance(),
                "hit_rate_percent": self.retrieval_hit_rate(),
            },
            "generation_checks": self.generation_checks,
        }

    def to_json(self, filepath: str = None) -> str:
        """Export metrics as JSON."""
        data = self.summary()
        data["exported_at"] = datetime.now().isoformat()
        json_str = json.dumps(data, indent=2)

        if filepath:
            Path(filepath).write_text(json_str, encoding="utf-8")

        return json_str


# ──────────────────────────────────────────────
# 2. Retrieval Evaluation
# ──────────────────────────────────────────────
def evaluate_retrieval(query: str, k: int = 5) -> dict:
    """
    Evaluate retrieval quality for a given query.
    Returns relevance scores and ranking metrics.

    Lower FAISS distance = more relevant (L2 distance).
    """
    store = get_vectorstore()
    if store is None:
        return {"error": "No papers indexed yet"}

    results = store.similarity_search_with_score(query, k=k)

    evaluated = []
    for doc, score in results:
        evaluated.append({
            "title": doc.metadata.get("title", "Unknown"),
            "chunk_index": doc.metadata.get("chunk_index", "?"),
            "relevance_score": round(float(score), 3),
            "preview": doc.page_content[:150] + "...",
        })

    # Metrics
    scores = [r["relevance_score"] for r in evaluated]
    return {
        "query": query,
        "num_results": len(evaluated),
        "scores": {
            "best": min(scores) if scores else None,
            "worst": max(scores) if scores else None,
            "mean": round(sum(scores) / len(scores), 3) if scores else None,
        },
        "results": evaluated,
    }


# ──────────────────────────────────────────────
# 3. Faithfulness Check (Citation Grounding)
# ──────────────────────────────────────────────
def check_faithfulness(generated_text: str, query: str, k: int = 5) -> dict:
    """
    Check how well generated text is grounded in retrieved documents.
    Compares key claims in generated text against vector DB content.

    This is a simple keyword overlap approach — production systems
    would use an LLM-as-judge or NLI model.
    """
    store = get_vectorstore()
    if store is None:
        return {"error": "No papers indexed yet"}

    results = store.similarity_search_with_score(query, k=k)
    source_text = " ".join(doc.page_content for doc, _ in results).lower()

    # Extract key terms from generated text (simple approach)
    generated_lower = generated_text.lower()

    # Check for technical terms overlap
    generated_words = set(generated_lower.split())
    source_words = set(source_text.split())

    # Filter to meaningful words (>5 chars, likely technical terms)
    technical_generated = {w for w in generated_words if len(w) > 5}
    technical_source = {w for w in source_words if len(w) > 5}

    if not technical_generated:
        return {"grounding_score": 0.0, "details": "No technical terms found in generated text"}

    overlap = technical_generated & technical_source
    grounding_score = len(overlap) / len(technical_generated)

    return {
        "grounding_score": round(grounding_score, 3),
        "technical_terms_in_generated": len(technical_generated),
        "technical_terms_in_sources": len(technical_source),
        "overlapping_terms": len(overlap),
        "sample_grounded_terms": list(overlap)[:10],
        "sample_ungrounded_terms": list(technical_generated - technical_source)[:10],
    }


# ──────────────────────────────────────────────
# 4. Chunk Coverage Analysis
# ──────────────────────────────────────────────
def analyze_chunk_coverage() -> dict:
    """
    Analyze the vector store contents.
    Shows distribution of chunks across indexed papers.
    """
    store = get_vectorstore()
    if store is None:
        return {"error": "No papers indexed yet"}

    # Get all docs from the store
    all_docs = store.docstore._dict
    papers = {}

    for doc_id, doc in all_docs.items():
        title = doc.metadata.get("title", "Unknown")
        if title not in papers:
            papers[title] = {
                "total_chunks": doc.metadata.get("total_chunks", 0),
                "stored_chunks": 0,
                "avg_chunk_length": 0,
                "total_chars": 0,
            }
        papers[title]["stored_chunks"] += 1
        papers[title]["total_chars"] += len(doc.page_content)

    for title, data in papers.items():
        if data["stored_chunks"] > 0:
            data["avg_chunk_length"] = round(data["total_chars"] / data["stored_chunks"])

    return {
        "total_papers": len(papers),
        "total_chunks": len(all_docs),
        "papers": papers,
    }
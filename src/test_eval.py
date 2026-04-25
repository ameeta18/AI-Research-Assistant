# test_eval.py (in project root)
import json
from src.tools.vector_store import index_paper, search_papers
from src.tools.read_pdf import read_pdf
from src.evaluation import (
    SessionMetrics,
    evaluate_retrieval, 
    check_faithfulness, 
    analyze_chunk_coverage,
)

# ─── Setup: Read and index a paper ───
print("📄 Reading paper...")
text = read_pdf.invoke({"url": "https://arxiv.org/pdf/1706.03762"})
print(f"Read {len(text)} characters\n")

print("💾 Indexing paper")
result = index_paper.invoke({"title": "Attention Is All You Need"})
print(result, "\n")

# ─── Evaluation 1: Chunk Coverage ───
print("=" * 50)
print("📦 CHUNK COVERAGE")
print("=" * 50)
coverage = analyze_chunk_coverage()
print(json.dumps(coverage, indent=2))

# ─── Evaluation 2: Retrieval Quality ───
print("\n" + "=" * 50)
print("🗂️ RETRIEVAL QUALITY")
print("=" * 50)
queries = ["attention mechanism", "transformer architecture", "self-attention"]
for query in queries:
    result = evaluate_retrieval(query)
    print(f"\nQuery: '{query}'")
    print(f"  Results: {result['num_results']}")
    print(f"  Best score: {result['scores']['best']}")
    print(f"  Worst score: {result['scores']['worst']}")
    print(f"  Mean score: {result['scores']['mean']}")

# ─── Evaluation 3: Faithfulness ───
print("\n" + "=" * 50)
print("✅ FAITHFULNESS CHECK")
print("=" * 50)
sample_text = """The Transformer model architecture relies entirely on 
self-attention mechanisms, dispensing with recurrence and convolutions. 
Multi-head attention allows the model to jointly attend to information 
from different representation subspaces. The encoder maps an input 
sequence to continuous representations using stacked self-attention 
and feed-forward layers."""
faith = check_faithfulness(sample_text, "transformer attention")
print(f"  Grounding score: {faith['grounding_score']:.1%}")
print(f"  Terms in generated: {faith['technical_terms_in_generated']}")
print(f"  Terms in source: {faith['technical_terms_in_sources']}")
print(f"  Overlapping terms: {faith['overlapping_terms']}")

# ─── Export Full Metrics ───
print("\n" + "=" * 50)
print("📥 EXPORTING METRICS")
print("=" * 50)
metrics = SessionMetrics()
metrics.log_paper_indexed("Attention Is All You Need", coverage.get("total_chunks", 0))
for query in queries:
    r = evaluate_retrieval(query)
    metrics.log_retrieval(query, r.get("results", []))

metrics.to_json("output/evaluation_results.json")
print("Saved to: output/evaluation_results.json")
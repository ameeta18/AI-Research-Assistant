# test_eval.py

import json
import os
from dotenv import load_dotenv
load_dotenv()

# Initialize embeddings before anything else
from src.tools.vector_store import init_embeddings
init_embeddings(os.getenv("GOOGLE_API_KEY"))

from src.tools.vector_store import index_paper, search_papers
from src.tools.read_pdf import read_pdf
from src.evaluation import (
    SessionMetrics,
    evaluate_retrieval,
    check_faithfulness,
    analyze_chunk_coverage,
    check_semantic_faithfulness
)


# ─── Read and index paper ───
print(" Reading paper")
text = read_pdf.invoke({"url": "https://arxiv.org/pdf/1706.03762"})
print(f"Read {len(text)} characters\n")

print(" Indexing paper")
result = index_paper.invoke({"title": "Attention Is All You Need"})
print(result, "\n")

# ─── Chunk Coverage ───

print(" CHUNK COVERAGE")

coverage = analyze_chunk_coverage()
print(json.dumps(coverage, indent=2))

# ─── Retrieval Quality ───

print(" RETRIEVAL QUALITY")
queries = [
    "multi-head attention mechanism",
    "encoder decoder architecture",
    "self-attention dot product",
    "positional encoding sinusoidal",
    "layer normalization residual",
]
for query in queries:
    result = evaluate_retrieval(query)
    print(f"\nQuery: '{query}'")
    print(f"  Results: {result['num_results']}")
    print(f"  Best score: {result['scores']['best']}")
    print(f"  Worst score: {result['scores']['worst']}")
    print(f"  Mean score: {result['scores']['mean']}")

# ─── Faithfulness ───
print(" FAITHFULNESS CHECK")
sample_text = """The Transformer model relies entirely on self-attention 
mechanisms to compute representations of its input and output without using 
sequence-aligned recurrence or convolution. Multi-head attention allows the 
model to jointly attend to information from different representation 
subspaces at different positions. The encoder is composed of a stack of 
identical layers, each having two sub-layers: a multi-head self-attention 
mechanism and a position-wise fully connected feed-forward network. Layer 
normalization and residual connections are employed around each sub-layer. 
The decoder inserts a third sub-layer which performs multi-head attention 
over the output of the encoder stack. Positional encoding using sinusoidal 
functions is added to the input embeddings to inject sequence order information."""

faith = check_faithfulness(sample_text, "transformer attention mechanism")
print(f"  Grounding score: {faith['grounding_score']:.1%}")
print(f"  Terms in generated: {faith['technical_terms_in_generated']}")
print(f"  Terms in source: {faith['technical_terms_in_sources']}")
print(f"  Overlapping terms: {faith['overlapping_terms']}")
print(f"  Sample grounded: {faith['sample_grounded_terms']}")
semantic_test_text = """Transformers process all tokens simultaneously rather 
than sequentially. The attention mechanism lets each word attend to every other 
word in the sequence. Multiple attention heads capture different types of 
relationships. The model was trained exclusively on vintage cat photographs 
from the 1980s. Quantum entanglement is the core principle behind the 
optimizer. Positional encoding injects information about word order."""
print(" SEMANTIC FAITHFULNESS CHECK")
semantic_faith = check_semantic_faithfulness(semantic_test_text)
print(f"  Grounding score: {semantic_faith['grounding_score']:.1%}")
print(f"  Method: {semantic_faith['method']}")
print(f"  Sentences grounded: {semantic_faith['sentences_grounded']}/{semantic_faith['sentences_evaluated']}")
print(f"  Avg similarity: {semantic_faith['avg_sentence_similarity']}")

# ─── Export ───

print(" EXPORTING METRICS")
metrics = SessionMetrics()
metrics.log_paper_indexed("Attention Is All You Need", coverage.get("total_chunks", 0))
for query in queries:
    r = evaluate_retrieval(query)
    metrics.log_retrieval(query, r.get("results", []))
metrics.log_generation_check("faithfulness", faith["grounding_score"], 
    f"{faith['overlapping_terms']}/{faith['technical_terms_in_generated']} terms grounded")

metrics.log_generation_check("semantic_faithfulness", semantic_faith["grounding_score"],
    f"{semantic_faith['sentences_grounded']}/{semantic_faith['sentences_evaluated']} sentences grounded")

metrics.to_json("output/evaluation_results.json")
print("Saved to: output/evaluation_results.json")
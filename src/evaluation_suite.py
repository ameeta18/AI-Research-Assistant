"""Deterministic offline evaluation and regression reporting."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, ConfigDict, Field, model_validator


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
REFERENCE_PATTERN = re.compile(
    r"^\[\d+\]\s+(.+?)\s+\((\d{4})\)\.\s+(.+?)\.\s+URL:\s+(https?://\S+)",
    re.MULTILINE,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceDocument(StrictModel):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=20)


class RetrievalCase(StrictModel):
    id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    expected_document_ids: list[str] = Field(min_length=1)
    expected_concepts: list[str] = Field(min_length=1)
    k: int = Field(default=3, ge=1, le=10)


class ToolSelectionCase(StrictModel):
    id: str = Field(min_length=1)
    user_request: str = Field(min_length=1)
    # Empty lists are valid when the correct response requires no tool call.
    expected_tools: list[str]
    observed_tools: list[str]


class GroundedClaim(StrictModel):
    statement: str = Field(min_length=1)
    evidence_terms: list[str] = Field(min_length=1)


class GenerationCase(StrictModel):
    id: str = Field(min_length=1)
    source_context: str = Field(min_length=20)
    generated_text: str = Field(min_length=20)
    claims: list[GroundedClaim] = Field(min_length=1)
    forbidden_claims: list[str] = Field(default_factory=list)
    expected_citation_urls: list[str] = Field(min_length=1)


class ToolEvent(StrictModel):
    tool: str = Field(min_length=1)
    success: bool
    duration_ms: float = Field(ge=0)


class RegressionThreshold(StrictModel):
    minimum: float = Field(ge=0, le=1)
    max_drop: float = Field(default=0.05, ge=0, le=1)


class EvaluationDataset(StrictModel):
    schema_version: str = Field(min_length=1)
    name: str = Field(min_length=1)
    evaluation_mode: str = "offline_challenge_fixtures"
    corpus: list[SourceDocument] = Field(min_length=1)
    retrieval_cases: list[RetrievalCase] = Field(min_length=1)
    tool_selection_cases: list[ToolSelectionCase] = Field(min_length=1)
    generation_cases: list[GenerationCase] = Field(min_length=1)
    tool_events: list[ToolEvent] = Field(min_length=1)
    thresholds: dict[str, RegressionThreshold]

    @model_validator(mode="after")
    def validate_references_and_ids(self) -> "EvaluationDataset":
        document_ids = [document.id for document in self.corpus]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("Corpus document IDs must be unique")

        case_ids = [
            case.id
            for group in (
                self.retrieval_cases,
                self.tool_selection_cases,
                self.generation_cases,
            )
            for case in group
        ]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Evaluation case IDs must be unique")

        known_documents = set(document_ids)
        for case in self.retrieval_cases:
            unknown = set(case.expected_document_ids) - known_documents
            if unknown:
                raise ValueError(
                    f"Retrieval case {case.id} references unknown documents: "
                    f"{sorted(unknown)}"
                )
        return self


class DeterministicHashEmbeddings(Embeddings):
    """Stable local embeddings for regression tests, not production ranking."""

    def __init__(self, dimensions: int = 512) -> None:
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        tokens = TOKEN_PATTERN.findall(text.casefold())
        features = tokens + [
            f"{left}_{right}"
            for left, right in zip(tokens, tokens[1:])
        ]
        vector = [0.0] * self.dimensions
        for feature in features:
            digest = hashlib.sha256(feature.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1.0

        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def _normalize_text(text: str) -> str:
    return " ".join(TOKEN_PATTERN.findall(text.casefold()))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_evaluation_dataset(path: str | Path) -> EvaluationDataset:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return EvaluationDataset.model_validate(data)


def evaluate_retrieval_cases(dataset: EvaluationDataset) -> dict[str, Any]:
    documents = [
        Document(
            page_content=document.text,
            metadata={"document_id": document.id, "title": document.title},
        )
        for document in dataset.corpus
    ]
    store = FAISS.from_documents(documents, DeterministicHashEmbeddings())

    case_results = []
    hit_scores = []
    reciprocal_ranks = []
    concept_coverages = []

    for case in dataset.retrieval_cases:
        results = store.similarity_search_with_score(case.query, k=case.k)
        retrieved_ids = [
            document.metadata.get("document_id", "")
            for document, _score in results
        ]
        expected_ids = set(case.expected_document_ids)
        relevant_ranks = [
            index + 1
            for index, document_id in enumerate(retrieved_ids)
            if document_id in expected_ids
        ]
        hit = bool(relevant_ranks)
        reciprocal_rank = 1 / min(relevant_ranks) if relevant_ranks else 0.0

        retrieved_text = _normalize_text(
            " ".join(document.page_content for document, _score in results)
        )
        matched_concepts = [
            concept
            for concept in case.expected_concepts
            if _normalize_text(concept) in retrieved_text
        ]
        concept_coverage = len(matched_concepts) / len(case.expected_concepts)

        hit_scores.append(float(hit))
        reciprocal_ranks.append(reciprocal_rank)
        concept_coverages.append(concept_coverage)
        case_results.append(
            {
                "id": case.id,
                "passed": hit and concept_coverage == 1.0,
                "retrieved_document_ids": retrieved_ids,
                "first_relevant_rank": min(relevant_ranks) if relevant_ranks else None,
                "concept_coverage": round(concept_coverage, 4),
            }
        )

    return {
        "metrics": {
            "retrieval_hit_rate": round(_mean(hit_scores), 4),
            "retrieval_mrr": round(_mean(reciprocal_ranks), 4),
            "retrieval_concept_coverage": round(_mean(concept_coverages), 4),
        },
        "cases": case_results,
    }


def evaluate_tool_selection_cases(dataset: EvaluationDataset) -> dict[str, Any]:
    case_results = []
    scores = []
    for case in dataset.tool_selection_cases:
        passed = case.observed_tools == case.expected_tools
        scores.append(float(passed))
        case_results.append(
            {
                "id": case.id,
                "passed": passed,
                "expected_tools": case.expected_tools,
                "observed_tools": case.observed_tools,
            }
        )
    return {
        "metrics": {"tool_selection_accuracy": round(_mean(scores), 4)},
        "cases": case_results,
    }


def _evaluate_grounding_case(case: GenerationCase) -> tuple[float, dict[str, Any]]:
    source = _normalize_text(case.source_context)
    generated = _normalize_text(case.generated_text)
    supported_claims = []
    unsupported_claims = []

    for claim in case.claims:
        statement_present = _normalize_text(claim.statement) in generated
        evidence_present = all(
            _normalize_text(term) in source
            for term in claim.evidence_terms
        )
        if statement_present and evidence_present:
            supported_claims.append(claim.statement)
        else:
            unsupported_claims.append(claim.statement)

    forbidden_found = [
        claim
        for claim in case.forbidden_claims
        if _normalize_text(claim) in generated
    ]
    passed_expectations = len(supported_claims) + (
        len(case.forbidden_claims) - len(forbidden_found)
    )
    total_expectations = len(case.claims) + len(case.forbidden_claims)
    score = passed_expectations / total_expectations
    return score, {
        "supported_claims": len(supported_claims),
        "expected_claims": len(case.claims),
        "unsupported_claims": unsupported_claims,
        "forbidden_claims_found": forbidden_found,
    }


def _evaluate_citations(case: GenerationCase) -> tuple[float, dict[str, Any]]:
    parsed_references = [
        {
            "author": author.strip(),
            "year": year,
            "title": title.strip(),
            "url": url.rstrip(".,"),
        }
        for author, year, title, url in REFERENCE_PATTERN.findall(case.generated_text)
    ]
    complete_urls = {
        reference["url"]
        for reference in parsed_references
        if reference["author"] and reference["year"] and reference["title"]
    }
    matched_urls = [
        url for url in case.expected_citation_urls if url in complete_urls
    ]
    score = len(matched_urls) / len(case.expected_citation_urls)
    return score, {
        "expected_citations": len(case.expected_citation_urls),
        "complete_citations": len(matched_urls),
        "missing_urls": sorted(set(case.expected_citation_urls) - complete_urls),
    }


def evaluate_generation_cases(dataset: EvaluationDataset) -> dict[str, Any]:
    case_results = []
    grounding_scores = []
    citation_scores = []

    for case in dataset.generation_cases:
        grounding_score, grounding_details = _evaluate_grounding_case(case)
        citation_score, citation_details = _evaluate_citations(case)
        grounding_scores.append(grounding_score)
        citation_scores.append(citation_score)
        case_results.append(
            {
                "id": case.id,
                "passed": grounding_score == 1.0 and citation_score == 1.0,
                "groundedness": round(grounding_score, 4),
                "citation_completeness": round(citation_score, 4),
                **grounding_details,
                **citation_details,
            }
        )

    return {
        "metrics": {
            "groundedness": round(_mean(grounding_scores), 4),
            "citation_completeness": round(_mean(citation_scores), 4),
        },
        "cases": case_results,
    }


def evaluate_tool_reliability(dataset: EvaluationDataset) -> dict[str, Any]:
    successful = sum(event.success for event in dataset.tool_events)
    durations = [event.duration_ms for event in dataset.tool_events]
    return {
        "metrics": {
            "tool_success_rate": round(successful / len(dataset.tool_events), 4),
        },
        "diagnostics": {
            "total_tool_calls": len(dataset.tool_events),
            "successful_tool_calls": successful,
            "mean_tool_duration_ms": round(_mean(durations), 2),
        },
    }


def run_offline_evaluation(dataset: EvaluationDataset) -> dict[str, Any]:
    retrieval = evaluate_retrieval_cases(dataset)
    tool_selection = evaluate_tool_selection_cases(dataset)
    generation = evaluate_generation_cases(dataset)
    reliability = evaluate_tool_reliability(dataset)

    return {
        "dataset": dataset.name,
        "dataset_version": dataset.schema_version,
        "evaluation_mode": dataset.evaluation_mode,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case_counts": {
            "retrieval": len(dataset.retrieval_cases),
            "tool_selection": len(dataset.tool_selection_cases),
            "generation": len(dataset.generation_cases),
            "tool_events": len(dataset.tool_events),
        },
        "metrics": {
            **retrieval["metrics"],
            **tool_selection["metrics"],
            **generation["metrics"],
            **reliability["metrics"],
        },
        "diagnostics": reliability["diagnostics"],
        "case_results": {
            "retrieval": retrieval["cases"],
            "tool_selection": tool_selection["cases"],
            "generation": generation["cases"],
        },
    }


def compare_with_baseline(
    report: dict[str, Any],
    baseline: dict[str, Any],
    thresholds: dict[str, RegressionThreshold],
) -> dict[str, Any]:
    current_metrics = report["metrics"]
    baseline_metrics = baseline.get("metrics", {})
    comparisons = {}
    all_passed = True

    for metric_name, threshold in thresholds.items():
        current = current_metrics.get(metric_name)
        previous = baseline_metrics.get(metric_name)
        if current is None or previous is None:
            passed = False
            delta = None
        else:
            delta = round(current - previous, 4)
            passed = (
                current >= threshold.minimum
                and delta >= -threshold.max_drop
            )
        comparisons[metric_name] = {
            "current": current,
            "baseline": previous,
            "delta": delta,
            "minimum": threshold.minimum,
            "max_allowed_drop": threshold.max_drop,
            "passed": passed,
        }
        all_passed = all_passed and passed

    return {"passed": all_passed, "metrics": comparisons}


def baseline_from_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_version": report["dataset_version"],
        "evaluation_mode": report["evaluation_mode"],
        "metrics": report["metrics"],
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    regression = report.get("regression", {})
    status = "PASS" if regression.get("passed", False) else "FAIL"
    rows = [
        "# AI Research Assistant Evaluation Report",
        "",
        f"- Dataset: `{report['dataset']}`",
        f"- Dataset version: `{report['dataset_version']}`",
        f"- Mode: `{report['evaluation_mode']}`",
        f"- Regression status: **{status}**",
        "",
        "| Metric | Current | Baseline | Delta | Minimum | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, comparison in regression.get("metrics", {}).items():
        delta = comparison["delta"]
        delta_text = "N/A" if delta is None else f"{delta:+.4f}"
        rows.append(
            "| "
            f"{name} | {comparison['current']} | {comparison['baseline']} | "
            f"{delta_text} | {comparison['minimum']} | "
            f"{'PASS' if comparison['passed'] else 'FAIL'} |"
        )

    rows.extend(
        [
            "",
            "## Coverage",
            "",
            f"- Retrieval cases: {report['case_counts']['retrieval']}",
            f"- Tool-selection cases: {report['case_counts']['tool_selection']}",
            f"- Generation cases: {report['case_counts']['generation']}",
            f"- Recorded tool events: {report['case_counts']['tool_events']}",
            "",
            "The default suite is deterministic and performs no external API calls.",
        ]
    )

    failed_cases = [
        (category, case)
        for category, cases in report.get("case_results", {}).items()
        for case in cases
        if not case.get("passed", False)
    ]
    rows.extend(["", "## Failed cases", ""])
    if failed_cases:
        rows.extend(
            f"- `{category}/{case['id']}`"
            for category, case in failed_cases
        )
    else:
        rows.append("- None")

    rows.extend(
        [
            "",
            "These failures are retained in the accepted baseline to represent "
            "known benchmark retrieval and agent-output limitations. New changes "
            "must not exceed the configured regression tolerance.",
        ]
    )
    return "\n".join(rows) + "\n"


def write_evaluation_report(
    report: dict[str, Any],
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    json_path = Path(json_path)
    markdown_path = Path(markdown_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")

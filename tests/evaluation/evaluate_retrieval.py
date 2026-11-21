"""Retrieval evaluation: `python -m tests.evaluation.evaluate_retrieval [--label NAME]`.

Scores each retriever on the answerable questions of the golden set and writes
results/<label>/retrieval.json and retrieval.md. Local retrievers cost nothing to run.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from regintel.config import get_settings
from regintel.evaluation.provenance import run_metadata
from regintel.evaluation.retrieval_metrics import load_golden_set, score_question, summarise
from regintel.ingestion.pipeline import read_chunks
from regintel.retrieval.bm25 import BM25Retriever
from regintel.retrieval.dense import DEFAULT_MODEL, DenseRetriever, FastEmbedEmbedder
from regintel.retrieval.hybrid import HybridRetriever

if TYPE_CHECKING:
    from regintel.retrieval.base import Retriever

try:  # use the OS certificate store for the model download (corporate proxies, Windows)
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

HERE = Path(__file__).parent
TOP_K = 5


def local_retrievers(model: str) -> list[Retriever]:
    settings = get_settings()
    chunks = read_chunks(settings.data_dir / "processed" / "chunks.jsonl")
    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever(chunks, FastEmbedEmbedder(model), cache_dir=settings.index_dir, model=model)
    return [bm25, dense, HybridRetriever([bm25, dense])]


def azure_retrievers(variants: list[str], efforts: list[str | None]) -> list[Retriever]:
    """Knowledge base variants x reasoning efforts (needs `az login` and the objects from scripts/azure/)."""
    from azure.identity import AzureCliCredential
    from azure.search.documents.knowledgebases import KnowledgeBaseRetrievalClient

    from regintel.azure_search.kb_retriever import KnowledgeBaseRetriever
    from regintel.azure_search.knowledge import AzureNames

    settings = get_settings()
    names = AzureNames(settings.azure_name_prefix)
    chunks = read_chunks(settings.data_dir / "processed" / "chunks.jsonl")
    credential = AzureCliCredential()
    retrievers: list[Retriever] = []
    for variant in variants:
        client = KnowledgeBaseRetrievalClient(
            settings.azure_search_endpoint or "",
            credential,
            knowledge_base_name=names.knowledge_base(variant),
            api_version=settings.azure_search_api_version,
        )
        for effort in efforts:
            name = f"kb-{variant}" + (f"-{effort}" if effort else "")
            retrievers.append(KnowledgeBaseRetriever(name, client, chunks, effort))  # type: ignore[arg-type]
    return retrievers


def evaluate(retrievers: list[Retriever]) -> dict[str, object]:
    golden = load_golden_set(HERE / "datasets" / "golden_set.json")
    questions = golden.answerable()
    report: dict[str, object] = {"golden_set_version": golden.version, "top_k": TOP_K, "retrievers": {}}
    for retriever in retrievers:
        started = time.perf_counter()
        results = [score_question(q, retriever.search(q.question, top_k=TOP_K)) for q in questions]
        entry: dict[str, object] = {
            "summary": summarise(results, TOP_K),
            "by_type": {
                t: summarise([r for r in results if r.type == t], TOP_K) for t in ("single", "comparison", "multihop")
            },
            "seconds_per_query": round((time.perf_counter() - started) / len(questions), 2),
            "questions": [asdict(r) for r in results],
        }
        cost = getattr(retriever, "cost", None)
        if cost is not None:
            entry["cost"] = asdict(cost)
        report["retrievers"][retriever.name] = entry  # type: ignore[index]
    return report


def to_markdown(report: dict[str, object], label: str) -> str:
    retrievers: dict[str, dict[str, object]] = report["retrievers"]  # type: ignore[assignment]
    k = report["top_k"]
    metrics = list(next(iter(retrievers.values()))["summary"].keys())[1:]  # type: ignore[attr-defined]
    lines = [
        f"# Retrieval evaluation — {label}",
        "",
        f"Golden set {report['golden_set_version']}, answerable questions only, top {k}.",
        "",
        "| Retriever | " + " | ".join(metrics) + " |",
        "|---|" + "---|" * len(metrics),
    ]
    for name, data in retrievers.items():
        summary: dict[str, float] = data["summary"]  # type: ignore[assignment]
        lines.append(f"| {name} | " + " | ".join(f"{summary[m]:.3f}" for m in metrics) + " |")
    lines += ["", "## Latency and cost", "", "| Retriever | s/query | planner tokens in/out per query | "
              "retrieval reasoning tokens per query | sub-queries per query |", "|---|---|---|---|---|"]  # fmt: skip
    for name, data in retrievers.items():
        cost: dict[str, object] | None = data.get("cost")  # type: ignore[assignment]
        if cost and cost["calls"]:
            n = int(cost["calls"])  # type: ignore[call-overload]
            lines.append(
                f"| {name} | {data['seconds_per_query']} | {int(cost['planning_input_tokens']) // n}"  # type: ignore[call-overload]
                f" / {int(cost['planning_output_tokens']) // n} | {int(cost['reasoning_tokens']) // n}"  # type: ignore[call-overload]
                f" | {int(cost['subqueries']) / n:.1f} |"  # type: ignore[call-overload]
            )
        else:
            lines.append(f"| {name} | {data['seconds_per_query']} | — | — | — |")
    lines += ["", "## Misses (no expected section in the top k)", ""]
    for name, data in retrievers.items():
        misses = [q for q in data["questions"] if q["section_rank"] is None]  # type: ignore[attr-defined, index]
        lines.append(f"**{name}** — {len(misses)} misses: " + ", ".join(q["id"] for q in misses))
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default=f"{date.today().isoformat()}-local")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--azure", action="store_true", help="evaluate the Azure knowledge bases instead")
    parser.add_argument("--variants", default="auto,sections", help="knowledge base variants (--azure)")
    parser.add_argument("--efforts", default="", help="comma-separated minimal,low,medium; empty = the KB's own")
    args = parser.parse_args()

    if args.azure:
        efforts: list[str | None] = [e for e in args.efforts.split(",") if e] or [None]
        report = evaluate(azure_retrievers(args.variants.split(","), efforts))
        report["setup"] = (
            "Azure AI Search knowledge bases; gpt-5-mini planning; extractive; "
            f"reasoning effort {args.efforts or 'low (knowledge base default)'}"
        )
    else:
        report = evaluate(local_retrievers(args.model))
        report["embedding_model"] = args.model
    out = HERE / "results" / args.label
    out.mkdir(parents=True, exist_ok=True)
    report["run"] = run_metadata()
    (out / "retrieval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "retrieval.md").write_text(to_markdown(report, args.label), encoding="utf-8")
    print(to_markdown(report, args.label))


if __name__ == "__main__":
    main()

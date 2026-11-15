"""Answer evaluation: `python -m tests.evaluation.evaluate_answers --label NAME [--effort medium]`.

Runs every golden question (answerable and unanswerable) through retrieval, the answer model and
the grounding verifier, and writes results/<label>/answers.json and answers.md.

Retrieval is cached per question in data/index/ (keyed by retriever and effort), so experiments
on the answering step re-use identical passages and cost only the answer model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from regintel.answering.models import answer_model_from_settings
from regintel.answering.pipeline import Answerer, verify_draft
from regintel.answering.schema import DraftAnswer, Usage
from regintel.config import get_settings
from regintel.corpus.manifest import load_manifest
from regintel.evaluation.answer_metrics import AnswerScore, score_answer, summarise_answers
from regintel.evaluation.retrieval_metrics import load_golden_set
from regintel.ingestion.pipeline import read_chunks
from regintel.retrieval.base import Hit

if TYPE_CHECKING:
    from regintel.ingestion.chunking import Chunk
    from regintel.retrieval.base import Retriever

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

HERE = Path(__file__).parent
TOP_K = 8
# List prices (USD per million tokens) used for the cost estimate; check before quoting.
PRICES = {"gpt-5-mini": {"input": 0.25, "cached": 0.025, "output": 2.00}}


def _save(path: Path, data: object) -> None:
    """Write via a temp file and replace, retrying: on Windows a scanner can briefly lock the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except OSError:
            time.sleep(0.5 * (attempt + 1))
    os.replace(tmp, path)


def cached_hits(retriever: Retriever, key: str, questions: list[str], chunks: dict[str, Chunk]) -> list[list[Hit]]:
    settings = get_settings()
    # Chunk ids are renumbered whenever the chunker changes, so the cache is tied to the chunk file.
    corpus = hashlib.sha256((settings.data_dir / "processed" / "chunks.jsonl").read_bytes()).hexdigest()[:10]
    path = settings.index_dir / f"eval-retrieval-{key}-{corpus}.json"
    cache: dict[str, list[list[object]]] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for question in questions:
        if question not in cache:
            cache[question] = [[h.chunk.chunk_id, h.score] for h in retriever.search(question, top_k=TOP_K)]
            _save(path, cache)
    return [[Hit(chunks[str(cid)], float(score)) for cid, score in cache[q] if str(cid) in chunks] for q in questions]  # type: ignore[arg-type]


def azure_kb(effort: str) -> Retriever:
    from azure.identity import AzureCliCredential
    from azure.search.documents.knowledgebases import KnowledgeBaseRetrievalClient

    from regintel.azure_search.kb_retriever import KnowledgeBaseRetriever
    from regintel.azure_search.knowledge import AzureNames

    settings = get_settings()
    client = KnowledgeBaseRetrievalClient(
        settings.azure_search_endpoint or "",
        AzureCliCredential(),
        knowledge_base_name=AzureNames(settings.azure_name_prefix).knowledge_base("sections"),
        api_version=settings.azure_search_api_version,
    )
    chunks = read_chunks(settings.data_dir / "processed" / "chunks.jsonl")
    return KnowledgeBaseRetriever(f"kb-sections-{effort}", client, chunks, effort)  # type: ignore[arg-type]


def to_markdown(report: dict[str, object], label: str) -> str:
    summary: dict[str, object] = report["summary"]  # type: ignore[assignment]
    lines = [
        f"# Answer evaluation — {label}",
        "",
        f"{report['setup']}",
        "",
        "| Metric | Value |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in summary.items()],
        "",
        "## By question type",
        "",
        "| Type | answer_rate | abstention_rate | fact_recall | all_facts | cited_expected_source |",
        "|---|---|---|---|---|---|",
    ]
    by_type: dict[str, dict[str, object]] = report["by_type"]  # type: ignore[assignment]
    for kind, s in by_type.items():
        lines.append(
            f"| {kind} | {s['answer_rate']} | {s['abstention_rate']} | {s['fact_recall']} | {s['all_facts']} "
            f"| {s['cited_expected_source']} |"
        )
    lines += ["", "## Questions with missing facts, wrong abstention or dropped statements", ""]
    for q in report["questions"]:  # type: ignore[attr-defined]
        wrong_abstention = (q["type"] == "unanswerable") == q["answered"]
        if q["facts_missing"] or wrong_abstention or q["problems"]:
            lines.append(f"- **{q['id']}** [{q['label']}] missing: {q['facts_missing']}; dropped: {q['problems']}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default=f"{date.today().isoformat()}-answers")
    parser.add_argument("--effort", choices=["minimal", "low", "medium"], default="medium", help="KB retrieval effort")
    parser.add_argument("--limit", type=int, default=0, help="only the first N questions (smoke test)")
    parser.add_argument("--reverify", action="store_true", help="re-verify and re-score the saved run --label")
    parser.add_argument("--strict-spaces", action="store_true", help="with --reverify: whitespace-sensitive quotes")
    parser.add_argument("--review", action="store_true", help="with --reverify: add the answer reviewer")
    args = parser.parse_args()
    if args.reverify:
        reverify(args.label, args.strict_spaces, args.review)
        return

    from azure.identity import AzureCliCredential, get_bearer_token_provider

    settings = get_settings()
    golden = load_golden_set(HERE / "datasets" / "golden_set.json")
    questions = golden.questions[: args.limit] if args.limit else golden.questions
    chunks = {c.chunk_id: c for c in read_chunks(settings.data_dir / "processed" / "chunks.jsonl")}
    retriever = azure_kb(args.effort)
    hits = cached_hits(retriever, f"kb-sections-{args.effort}", [q.question for q in questions], chunks)

    tokens = get_bearer_token_provider(AzureCliCredential(), "https://cognitiveservices.azure.com/.default")
    model = answer_model_from_settings(settings, tokens)
    answerer = Answerer(retriever, model, load_manifest(settings.data_dir / "manifest.json"), top_k=TOP_K)
    scores = []
    for question, question_hits in zip(questions, hits, strict=True):
        result = answerer.answer(question.question, hits=question_hits)
        scores.append(score_answer(question, result))
        print(f"{question.id:7} {result.label:18} facts={scores[-1].fact_recall}")

    setup = (
        f"Retrieval: Azure AI Search knowledge base over section chunks, {args.effort} reasoning effort, "
        f"top {TOP_K}. Answers: {model.name} (structured output), verified by regintel.answering.verify."
    )
    write_report(build_report(scores, golden.version, setup, settings.azure_openai_chat_deployment), args.label)


def build_report(scores: list[AnswerScore], version: str, setup: str, model: str) -> dict[str, object]:
    summary = summarise_answers(scores)
    usage: dict[str, int] = summary["tokens_per_question"]  # type: ignore[assignment]
    price = PRICES.get(model)
    if price:
        per_question = (
            (usage["input"] - usage["cached"]) * price["input"]
            + usage["cached"] * price["cached"]
            + usage["output"] * price["output"]
        ) / 1e6
        summary["est_usd_per_answer"] = round(per_question, 5)
    return {
        "golden_set_version": version,
        "setup": setup,
        "model": model,
        "summary": summary,
        "by_type": {
            t: summarise_answers([s for s in scores if s.type == t])
            for t in ("single", "comparison", "multihop", "unanswerable")
        },
        "questions": [asdict(s) for s in scores],
    }


def write_report(report: dict[str, object], label: str) -> None:
    out = HERE / "results" / label
    out.mkdir(parents=True, exist_ok=True)
    (out / "answers.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "answers.md").write_text(to_markdown(report, label), encoding="utf-8")
    print(to_markdown(report, label))


def reverify(label: str, strict_spaces: bool = False, review: bool = False) -> None:
    """Re-run verification and scoring on a saved run's drafts with the current code (no model calls).
    With strict_spaces, whitespace must match too (the first verifier); written to <label>-strict-spaces."""
    from regintel.answering import verify

    verify.IGNORE_SPACES = not strict_spaces
    settings = get_settings()
    path = HERE / "results" / label / "answers.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    golden = {q.id: q for q in load_golden_set(HERE / "datasets" / "golden_set.json").questions}
    chunks = {c.chunk_id: c for c in read_chunks(settings.data_dir / "processed" / "chunks.jsonl")}
    manifest = load_manifest(settings.data_dir / "manifest.json")
    reviewer = None
    if review:
        from azure.identity import AzureCliCredential, get_bearer_token_provider

        from regintel.answering.review import reviewer_from_settings

        tokens = get_bearer_token_provider(AzureCliCredential(), "https://cognitiveservices.azure.com/.default")
        reviewer = reviewer_from_settings(settings, tokens)
        if reviewer is None:
            raise SystemExit("AZURE_JUDGE_ENDPOINT is not set")
    scores = []
    for saved in report["questions"]:
        question = golden[saved["id"]]
        if saved.get("draft") is None:  # nothing was retrieved, so the model was never called
            scores.append(AnswerScore(**saved))
            continue
        usage = Usage(
            saved["input_tokens"], saved["cached_tokens"], saved["output_tokens"], saved["reasoning_tokens"], 1
        )
        draft = DraftAnswer.model_validate(saved["draft"])
        passages = [chunks[c] for c in saved["retrieved"]]
        result = verify_draft(question.question, draft, passages, manifest, usage)
        result.generation_seconds = saved["generation_seconds"]
        if reviewer is not None:
            result = reviewer.review(result, [c.text for c in passages])
        scores.append(score_answer(question, result))
    # Never overwrite the saved run: a variant goes to its own folder.
    out_label = f"{label}-strict-spaces" if strict_spaces else f"{label}-reviewed" if review else label
    setup = report["setup"]
    if strict_spaces:
        setup += " Re-verified with whitespace-sensitive quote matching."
    if reviewer is not None:
        setup += f" Then reviewed by {reviewer.model} (statement entailment and question-level subject check)."
    write_report(
        build_report(scores, report["golden_set_version"], setup, report.get("model", "gpt-5-mini")), out_label
    )


if __name__ == "__main__":
    main()

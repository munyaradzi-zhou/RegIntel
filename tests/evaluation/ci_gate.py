"""Offline evaluation gate: `python -m tests.evaluation.ci_gate [--update-baseline] [--out FILE]`.

Free and deterministic, so it runs on every pull request (no model call, no Azure, no download):
- retrieval: BM25 over the committed chunk fixture, scored on the golden set at the configured top k;
- verifier: recorded and mutated drafts must be kept and dropped exactly as recorded;
- router: the trained router (ROUTER_MODEL_VERSION) on the held-out real questions of router_dataset.json;
- prompts: every prompt's version and text hash, compared with the baseline.

Exit code 1 when the gate fails. `--update-baseline` records the current numbers as the new bar:
do that deliberately, in the same pull request as the improvement, and say why.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from regintel.answering.schema import DraftAnswer
from regintel.config import get_settings
from regintel.evaluation.gate import (
    DEFAULT_TOLERANCE,
    VerifierCase,
    compare_metrics,
    compare_prompts,
    passed,
    render_markdown,
    run_verifier_cases,
)
from regintel.evaluation.retrieval_metrics import load_golden_set, score_question, summarise
from regintel.ingestion.chunking import Chunk
from regintel.ml_router import load_model
from regintel.prompts import prompt_versions
from regintel.retrieval.bm25 import BM25Retriever

HERE = Path(__file__).parent
FIXTURES = HERE / "fixtures"
BASELINE = HERE / "baselines" / "ci_baseline.json"


def load_fixture_chunks(path: Path = FIXTURES / "chunks.jsonl.gz") -> list[Chunk]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [Chunk(**json.loads(line)) for line in fh if line.strip()]


def retrieval_metrics(chunks: list[Chunk], top_k: int) -> dict[str, float]:
    golden = load_golden_set(HERE / "datasets" / "golden_set.json")
    retriever = BM25Retriever(chunks)
    results = [score_question(q, retriever.search(q.question, top_k=top_k)) for q in golden.answerable()]
    summary = summarise(results, top_k)
    return {
        f"retrieval.bm25.{name.replace(f'@{top_k}', '@k')}": float(value)
        for name, value in summary.items()
        if name != "questions"
    }


def load_verifier_cases(path: Path = FIXTURES / "verifier_cases.json") -> list[VerifierCase]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        VerifierCase(
            id=c["id"],
            kind=c["kind"],
            question=c["question"],
            passages=c["passages"],
            draft=DraftAnswer.model_validate(c["draft"]),
            expected_verified=c["expected_verified"],
            expected_label=c["expected_label"],
        )
        for c in data["cases"]
    ]


def router_metrics(version: int, path: Path = HERE / "datasets" / "router_dataset.json") -> dict[str, float]:
    """Accuracy and macro-F1 of the trained router on the real questions held out from training.
    Unanswerable questions count as correct when not sent to the agent (as for the LLM router)."""
    model = load_model(version)
    test = [q for q in json.loads(path.read_text(encoding="utf-8"))["questions"] if q["split"] == "test"]
    predicted = [model.predict(q["question"])[0] for q in test]
    correct = [
        (p != "agent") if q["kind"] == "unanswerable" else p == q["route"] for q, p in zip(test, predicted, strict=True)
    ]
    pairs = [(q["route"], p) for q, p in zip(test, predicted, strict=True) if q["kind"] != "unanswerable"]
    f1s = []
    for route in model.classes:
        tp = sum(t == route and p == route for t, p in pairs)
        fp = sum(t != route and p == route for t, p in pairs)
        fn = sum(t == route and p != route for t, p in pairs)
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return {"router.ml.test_accuracy": round(sum(correct) / len(correct), 3),
            "router.ml.test_macro_f1": round(sum(f1s) / len(f1s), 3)}  # fmt: skip


def _sha256(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix == ".json":  # line endings differ between a Windows checkout and the CI runner
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()[:12]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--out", type=Path, help="also write the markdown report here (for the PR comment)")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # the report has status icons; Windows consoles default to cp1252

    settings = get_settings()
    top_k = settings.retrieval_top_k
    chunks = load_fixture_chunks()
    outcome = run_verifier_cases(load_verifier_cases(), {c.chunk_id: c for c in chunks})
    metrics = retrieval_metrics(chunks, top_k) | outcome.metrics() | router_metrics(settings.router_model_version)
    prompts = prompt_versions()
    fixtures = {p.name: _sha256(p) for p in sorted(FIXTURES.glob("*.*")) if p.suffix in (".gz", ".json")}

    if args.update_baseline:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        baseline = {"updated": datetime.now(UTC).date().isoformat(), "top_k": top_k, "fixtures": fixtures,
                    "router_model_version": settings.router_model_version,
                    "metrics": metrics, "prompts": prompts}  # fmt: skip
        BASELINE.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
        print(f"baseline written: {BASELINE.relative_to(HERE.parent.parent)}")

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    metric_rows = compare_metrics(baseline["metrics"], metrics, args.tolerance)
    prompt_rows = compare_prompts(baseline["prompts"], prompts)
    notes = [
        f"BM25 over {len(chunks):,} chunks at top_k = {top_k}; {outcome.cases} verifier cases "
        f"({outcome.kept} statements kept, {outcome.dropped} dropped)."
    ]
    if fixtures != baseline.get("fixtures"):
        notes.append("The fixtures changed since the baseline was recorded: update the baseline in this change.")
    if settings.router_model_version != baseline.get("router_model_version"):
        notes.append(f"Router model v{baseline.get('router_model_version')} -> v{settings.router_model_version}.")
    if top_k != baseline.get("top_k"):
        notes.append(f"top_k changed from {baseline.get('top_k')} to {top_k}.")
    report = render_markdown(metric_rows, prompt_rows, args.tolerance, notes, outcome.mismatches)

    print(report)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(report)
    if args.out:
        args.out.write_text(report, encoding="utf-8")
    return 0 if passed(metric_rows, prompt_rows) else 1


if __name__ == "__main__":
    sys.exit(main())

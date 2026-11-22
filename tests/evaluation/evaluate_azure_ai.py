"""Azure AI Evaluation beside RegIntel's own metrics, on a saved answer run.

    python -m tests.evaluation.evaluate_azure_ai --run 2026-11-21-writer-gpt-5-mini-reviewed
    python -m tests.evaluation.evaluate_azure_ai --run <label> --limit 5        # smoke test

For each golden question of the run (the passages the writer saw are rebuilt from the chunk store):
- GroundednessEvaluator on the answer as shown (after the verifier and reviewer) and on the raw
  draft (every statement the writer wrote), so the effect of verification is visible on the SDK's
  own scale;
- RelevanceEvaluator on the shown answer;
- RetrievalEvaluator on the passages.
Scores are 1-5 (pass at 3 or more), judged by an LLM (JUDGE_DEPLOYMENT, default gpt-5-mini on the
Foundry project's account). Results go to results/<run>-azure-ai/: per-question rows, a summary,
and the cases where the SDK and RegIntel disagree.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from regintel.config import get_settings
from regintel.evaluation.provenance import run_metadata
from regintel.ingestion.pipeline import read_chunks

HERE = Path(__file__).parent
ANSWERED = ("grounded", "partially_grounded")
DEFAULT_ENDPOINT = "https://ais-regintel-dev-5bbwe.cognitiveservices.azure.com"


def _score(result: dict[str, Any], name: str) -> float | None:
    value = result.get(name)
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else float(value)


def _tokens(*results: dict[str, Any]) -> int:
    return sum(int(v.get("total_tokens", 0)) for r in results for k, v in r.items() if k.endswith("_properties"))


class CachedToken:
    """AzureCliCredential runs `az` for every token; parallel judges then time the CLI out. One
    token, refreshed five minutes before it expires, shared by every thread."""

    def __init__(self, inner: Any) -> None:
        import threading

        self.inner = inner
        self.lock = threading.Lock()
        self.tokens: dict[tuple[str, ...], Any] = {}

    def get_token(self, *scopes: str, **kwargs: Any) -> Any:
        import time

        with self.lock:
            token = self.tokens.get(scopes)
            if token is None or token.expires_on - time.time() < 300:
                token = self.tokens[scopes] = self.inner.get_token(*scopes, **kwargs)
            return token


def with_retries(fn: Any) -> Any:
    """The judge shares a rate-limited deployment: back off on 429 instead of failing the run."""
    from openai import RateLimitError
    from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

    return retry(retry=retry_if_exception_type(RateLimitError), stop=stop_after_attempt(8),
                 wait=wait_exponential_jitter(initial=5, max=60), reraise=True)(fn)  # fmt: skip


def evaluate_one(
    q: dict[str, Any], golden: dict[str, Any], chunks: dict[str, Any], evaluators: dict[str, Any]
) -> dict[str, Any]:
    question = golden["question"]
    context = "\n\n".join(chunks[c].text for c in q["retrieved"] if c in chunks)
    draft = " ".join(s["text"] for s in (q.get("draft") or {}).get("statements", []))
    row: dict[str, Any] = {"id": q["id"], "type": q["type"], "label": q["label"], "fact_recall": q.get("fact_recall"),
                           "statements": q.get("statements"), "verified": q.get("verified")}  # fmt: skip
    results = []
    if context:
        shown = evaluators["groundedness"](query=question, response=q["answer"], context=context)
        results.append(shown)
        row["groundedness_shown"] = _score(shown, "groundedness")
        row["groundedness_reason"] = shown.get("groundedness_reason", "")
        if draft:
            raw = evaluators["groundedness"](query=question, response=draft, context=context)
            results.append(raw)
            row["groundedness_draft"] = _score(raw, "groundedness")
        retrieval = evaluators["retrieval"](query=question, context=context)
        results.append(retrieval)
        row["retrieval"] = _score(retrieval, "retrieval")
    relevance = evaluators["relevance"](query=question, response=q["answer"])
    results.append(relevance)
    row["relevance"] = _score(relevance, "relevance")
    row["relevance_reason"] = relevance.get("relevance_reason", "")
    row["judge_tokens"] = _tokens(*results)
    return row


def _mean(values: list[float | None]) -> float | None:
    kept = [v for v in values if v is not None]
    return round(sum(kept) / len(kept), 2) if kept else None


def _share(values: list[float | None]) -> str:
    kept = [v for v in values if v is not None]
    return f"{sum(v >= 3 for v in kept)}/{len(kept)}" if kept else "-"


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [r for r in rows if r["type"] != "unanswerable"]
    unanswerable = [r for r in rows if r["type"] == "unanswerable"]
    answered = [r for r in answerable if r["label"] in ANSWERED]
    declined = [r for r in answerable if r["label"] not in ANSWERED]
    dropped = [r for r in answered if r.get("statements") and r["verified"] < r["statements"]]
    return {
        "questions": len(rows),
        "answered": {
            "n": len(answered),
            "groundedness_shown_mean": _mean([r.get("groundedness_shown") for r in answered]),
            "groundedness_shown_pass": _share([r.get("groundedness_shown") for r in answered]),
            "relevance_mean": _mean([r.get("relevance") for r in answered]),
            "relevance_pass": _share([r.get("relevance") for r in answered]),
            "retrieval_mean": _mean([r.get("retrieval") for r in answered]),
        },
        "answers_with_dropped_statements": {
            "n": len(dropped),
            "groundedness_draft_mean": _mean([r.get("groundedness_draft") for r in dropped]),
            "groundedness_shown_mean": _mean([r.get("groundedness_shown") for r in dropped]),
        },
        "answerable_but_declined": {
            "n": len(declined),
            "ids": [r["id"] for r in declined],
            "retrieval_mean": _mean([r.get("retrieval") for r in declined]),
            "relevance_pass": _share([r.get("relevance") for r in declined]),
        },
        "unanswerable_declined": {
            "n": len(unanswerable),
            "relevance_mean": _mean([r.get("relevance") for r in unanswerable]),
            "relevance_pass": _share([r.get("relevance") for r in unanswerable]),
        },
        "disagreements": {
            "grounded_by_regintel_but_groundedness_below_3": [
                r["id"] for r in answered if (r.get("groundedness_shown") or 5) < 3
            ],
            "declined_by_regintel_but_retrieval_5": [r["id"] for r in declined if r.get("retrieval") == 5],
        },
        "judge_tokens": sum(r["judge_tokens"] for r in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="a results/<label> folder with answers.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--judge", default="gpt-5-mini", help="judge deployment")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    import truststore

    truststore.inject_into_ssl()
    from azure.ai.evaluation import GroundednessEvaluator, RelevanceEvaluator, RetrievalEvaluator
    from azure.identity import AzureCliCredential

    config = {"azure_endpoint": args.endpoint, "azure_deployment": args.judge, "api_version": "2025-04-01-preview"}
    credential = CachedToken(AzureCliCredential(process_timeout=60))
    reasoning = args.judge.startswith(("gpt-5", "o"))
    evaluators = {
        "groundedness": with_retries(
            GroundednessEvaluator(config, credential=credential, is_reasoning_model=reasoning)
        ),
        "relevance": with_retries(RelevanceEvaluator(config, credential=credential, is_reasoning_model=reasoning)),
        "retrieval": with_retries(RetrievalEvaluator(config, credential=credential, is_reasoning_model=reasoning)),
    }
    run = json.loads((HERE / "results" / args.run / "answers.json").read_text(encoding="utf-8"))
    golden = {
        q["id"]: q for q in json.loads((HERE / "datasets" / "golden_set.json").read_text(encoding="utf-8"))["questions"]
    }
    chunks = {c.chunk_id: c for c in read_chunks(get_settings().data_dir / "processed" / "chunks.jsonl")}
    questions = run["questions"][: args.limit] if args.limit else run["questions"]

    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(lambda q: evaluate_one(q, golden[q["id"]], chunks, evaluators), questions))
    out = HERE / "results" / f"{args.run}-azure-ai"
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    summary = {"run": run_metadata(), "source_run": args.run, "judge": args.judge} | summarise(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "run"}, indent=2))


if __name__ == "__main__":
    main()

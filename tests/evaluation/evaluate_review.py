"""Answer-reviewer calibration: `python -m tests.evaluation.evaluate_review --label NAME`.

Runs the reviewer on hand-built answers (datasets/review_cases.json) with known verdicts and
reports how many overreaching or unsupported statements it catches, how many supported
statements it wrongly rejects, and its question-level accuracy (answer is about the subject asked).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from openai import OpenAI

from regintel.answering.review import AnswerReviewer
from regintel.answering.schema import AnswerResult, CheckedStatement, Usage, VerifiedCitation
from regintel.config import get_settings
from regintel.evaluation.provenance import run_metadata

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

HERE = Path(__file__).parent


def as_result(case: dict[str, object]) -> AnswerResult:
    statements = [
        CheckedStatement(
            s["text"],
            [VerifiedCitation(f"{case['id']}#{i}", "doc", "", "", s["source"], q) for q in s["quotes"]],
            True,
        )
        for i, s in enumerate(case["statements"])  # type: ignore[attr-defined]
    ]
    return AnswerResult(str(case["question"]), "grounded", statements, None, [], Usage(), 0.0, 0.0)


def judge_client() -> tuple[OpenAI, str]:
    from azure.identity import AzureCliCredential, get_bearer_token_provider

    settings = get_settings()
    if not settings.azure_judge_endpoint:
        raise SystemExit("AZURE_JUDGE_ENDPOINT is not set")
    tokens = get_bearer_token_provider(AzureCliCredential(), "https://cognitiveservices.azure.com/.default")
    client = OpenAI(base_url=settings.azure_judge_endpoint.rstrip("/") + "/openai/v1/", api_key=tokens)
    return client, settings.azure_judge_deployment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    client, model = judge_client()
    reviewer = AnswerReviewer(client, model)
    cases = json.loads((HERE / "datasets" / "review_cases.json").read_text(encoding="utf-8"))["cases"]
    rows, question_rows, seconds = [], [], []
    for case in cases:
        started = time.perf_counter()
        review, _ = reviewer.judge(as_result(case))
        seconds.append(time.perf_counter() - started)
        got = {v.statement: v for v in review.verdicts}
        for i, s in enumerate(case["statements"], start=1):
            verdict = got.get(i)
            rows.append({
                "case": case["id"], "statement": s["text"], "expected": s["expected"],
                "got": verdict.verdict if verdict else "missing", "reason": verdict.reason if verdict else "",
            })  # fmt: skip
        question_rows.append({"case": case["id"], "expected": case["addresses_question"],
                              "got": review.addresses_question, "missing_subject": review.missing_subject})  # fmt: skip
        print(
            case["id"], [(r["expected"], r["got"]) for r in rows if r["case"] == case["id"]], review.addresses_question
        )

    bad = [r for r in rows if r["expected"] != "supported"]
    good = [r for r in rows if r["expected"] == "supported"]
    summary = {
        "statements": len(rows),
        "problems_caught": f"{sum(r['got'] != 'supported' for r in bad)}/{len(bad)}",
        "exact_verdict": f"{sum(r['got'] == r['expected'] for r in rows)}/{len(rows)}",
        "supported_wrongly_rejected": f"{sum(r['got'] != 'supported' for r in good)}/{len(good)}",
        "question_level_correct": f"{sum(q['expected'] == q['got'] for q in question_rows)}/{len(question_rows)}",
        "mean_seconds_per_answer": round(sum(seconds) / len(seconds), 1),
    }
    out = HERE / "results" / args.label
    out.mkdir(parents=True, exist_ok=True)
    report = {"run": run_metadata(), "model": model, "summary": summary, "statements": rows, "questions": question_rows}
    (out / "review.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [f"# Answer reviewer calibration — {args.label}", "", f"Reviewer: {model}.", ""]
    lines += [f"- {k}: {v}" for k, v in summary.items()]
    lines += ["", "Disagreements:"]
    lines += [f"- {r['case']}: expected {r['expected']}, got {r['got']} — {r['statement']} ({r['reason']})"
              for r in rows if r["got"] != r["expected"]]  # fmt: skip
    lines += [f"- {q['case']}: addresses_question expected {q['expected']}, got {q['got']}"
              for q in question_rows if q["expected"] != q["got"]]  # fmt: skip
    (out / "review.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

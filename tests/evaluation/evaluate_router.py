"""Routing accuracy: `python -m tests.evaluation.evaluate_router --label NAME`.

Expected routes come from the datasets, not from judgement: tool questions -> agent; golden
comparison and multi-hop questions -> guidance_multi; golden single questions -> guidance_single.
Unanswerable golden questions are scored only on not being sent to the agent (either guidance
path is acceptable). Misrouting is reported by direction, since the costs differ: a single
question sent to the multi path costs latency; a data question sent to guidance cannot be answered.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from regintel.answering.models import azure_openai_client
from regintel.config import get_settings
from regintel.evaluation.retrieval_metrics import load_golden_set
from regintel.router import Router

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

HERE = Path(__file__).parent


def expected_route(qtype: str) -> str | None:
    return {"single": "guidance_single", "comparison": "guidance_multi", "multihop": "guidance_multi"}.get(qtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    from azure.identity import AzureCliCredential, get_bearer_token_provider

    settings = get_settings()
    tokens = get_bearer_token_provider(AzureCliCredential(), "https://cognitiveservices.azure.com/.default")
    router = Router(azure_openai_client(settings, tokens), settings.azure_openai_chat_deployment)

    questions = [
        (q.id, q.question, q.type, expected_route(q.type))
        for q in load_golden_set(HERE / "datasets" / "golden_set.json").questions
    ]
    tools = json.loads((HERE / "datasets" / "tool_questions.json").read_text(encoding="utf-8"))["questions"]
    questions += [(q["id"], q["question"], "tool", "agent") for q in tools]

    rows = []
    for qid, question, qtype, expected in questions:
        started = time.perf_counter()
        decision, usage = router.route(question)
        ok = (decision.path != "agent") if expected is None else decision.path == expected
        rows.append({"id": qid, "type": qtype, "expected": expected or "guidance_*", "got": decision.path, "ok": ok,
                     "reason": decision.reason, "seconds": round(time.perf_counter() - started, 2),
                     "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens})  # fmt: skip
        print(f"{qid:7} {qtype:12} expected={rows[-1]['expected']:16} got={decision.path:16} {'ok' if ok else 'MISS'}")

    n = len(rows)
    errors = Counter(f"{r['expected']} -> {r['got']}" for r in rows if not r["ok"])
    data_rows = [r for r in rows if r["type"] == "tool"]
    summary = {
        "accuracy": round(sum(r["ok"] for r in rows) / n, 3),
        "data_questions_to_agent": f"{sum(r['ok'] for r in data_rows)}/{len(data_rows)}",
        "guidance_questions_to_agent": sum(r["got"] == "agent" for r in rows if r["type"] != "tool"),
        "errors": dict(errors),
        "mean_seconds": round(sum(r["seconds"] for r in rows) / n, 2),
        "mean_tokens_in_out": [sum(r["input_tokens"] for r in rows) // n, sum(r["output_tokens"] for r in rows) // n],
    }
    out = HERE / "results" / args.label
    out.mkdir(parents=True, exist_ok=True)
    (out / "router.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    lines = [f"# Routing evaluation — {args.label}", "", *[f"- {k}: {v}" for k, v in summary.items()], "", "Misses:"]
    lines += [
        f"- {r['id']} ({r['type']}): expected {r['expected']}, got {r['got']} — {r['reason']}"
        for r in rows
        if not r["ok"]
    ]
    (out / "router.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

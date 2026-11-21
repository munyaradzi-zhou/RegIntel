"""Live evaluation of a deployed API: `python -m tests.evaluation.evaluate_live --url https://... --label NAME`.

Asks the golden set and the tool questions over HTTP, one at a time, exactly as a client would,
and records what the deployed system answered and how long it took. Scores use the same rules as
the local evaluations, so the numbers compare: expected facts found in the answer, the expected
document cited, unanswerable questions declined, data questions sent to the agent.

Latency is measured at the client (network, TLS, the container and every model call) next to the
server's own split (route, research, writing, web). The API key comes from REGINTEL_API_KEY or,
with --vault, from Key Vault (owner-api-key) through the Azure CLI; it is never printed or saved.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess  # nosec B404: fixed az command, no shell
import time
from collections import Counter
from pathlib import Path

import httpx

from regintel.evaluation.answer_metrics import fact_found
from regintel.evaluation.provenance import run_metadata

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

HERE = Path(__file__).parent
ANSWERED = ("grounded", "partially_grounded")


def api_key(vault: str | None) -> str:
    if key := os.environ.get("REGINTEL_API_KEY"):
        return key
    if not vault:
        raise SystemExit("set REGINTEL_API_KEY or pass --vault")
    az = "az.cmd" if os.name == "nt" else "az"
    command = [az, "keyvault", "secret", "show", "--vault-name", vault, "-n", "owner-api-key"]
    return subprocess.run(  # nosec B603
        [*command, "--query", "value", "-o", "tsv"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def load_questions(limit: int | None) -> list[dict[str, object]]:
    golden = json.loads((HERE / "datasets" / "golden_set.json").read_text(encoding="utf-8"))["questions"]
    tools = json.loads((HERE / "datasets" / "tool_questions.json").read_text(encoding="utf-8"))["questions"]
    for q in tools:
        q["type"] = "data"
    questions = golden + tools
    return questions[:limit] if limit else questions


def score(q: dict[str, object], body: dict[str, object]) -> dict[str, object]:
    qtype = str(q["type"])
    label = str(body.get("label", "error"))
    answer = str(body.get("answer", ""))
    facts = [str(f) for f in q.get("facts", [])]  # type: ignore[attr-defined]
    cited = {s["doc_id"] for s in body.get("sources", [])}  # type: ignore[attr-defined]
    expected = {e["doc_id"] for e in q.get("expected", [])}  # type: ignore[attr-defined]
    row: dict[str, object] = {"label": label, "path": body.get("path")}
    if qtype == "unanswerable":
        row["declined"] = label not in ANSWERED
    else:
        row["answered"] = label in ANSWERED
        row["facts"] = [fact_found(f, answer) for f in facts]
        if expected:
            row["cited_expected"] = bool(cited & expected)
    if qtype == "data":
        row["to_agent"] = body.get("path") == "agent"
    return row


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(p * (len(ordered) - 1)))]


def summarise(rows: list[dict[str, object]]) -> dict[str, object]:
    def share(key: str, subset: list[dict[str, object]]) -> str:
        have = [r[key] for r in subset if key in r]
        return f"{sum(map(bool, have))}/{len(have)}" if have else "-"

    ok = [r for r in rows if r["status"] == 200]
    answerable = [r for r in ok if r["type"] != "unanswerable"]
    facts = [f for r in answerable for f in r.get("facts", [])]  # type: ignore[attr-defined]
    latency: dict[str, object] = {}
    groups = [("all", ok)] + [(p, [r for r in ok if str(r["path"]) == p]) for p in sorted({str(r["path"]) for r in ok})]
    for name, subset in groups:
        seconds = [float(r["seconds"]) for r in subset]  # type: ignore[arg-type]
        if seconds:
            latency[name] = {
                "n": len(seconds),
                "p50": round(statistics.median(seconds), 1),
                "p95": round(percentile(seconds, 0.95), 1),
                "max": round(max(seconds), 1),
            }
    stages = ("route", "research", "writing", "web")
    server = {k: round(statistics.mean(float(r["timings"].get(k, 0)) for r in ok), 2) for k in stages} if ok else {}  # type: ignore[attr-defined]
    return {
        "questions": len(rows),
        "http_errors": len(rows) - len(ok),
        "labels": dict(Counter(str(r["label"]) for r in ok)),
        "answer_rate": share("answered", answerable),
        "abstention": share("declined", ok),
        "fact_recall": f"{sum(facts)}/{len(facts)}" if facts else "-",
        "expected_doc_cited": share("cited_expected", answerable),
        "data_to_agent": share("to_agent", ok),
        "latency_seconds": latency,
        "server_mean_seconds": server,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="the API base URL")
    parser.add_argument("--label", required=True)
    parser.add_argument("--vault", help="Key Vault holding owner-api-key (else REGINTEL_API_KEY)")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    out = HERE / "results" / args.label
    out.mkdir(parents=True, exist_ok=True)
    headers = {"X-API-Key": api_key(args.vault)}
    base = args.url.rstrip("/")
    rows: list[dict[str, object]] = []
    with httpx.Client(base_url=base, headers=headers, timeout=300) as client:
        start = time.perf_counter()
        ready = client.get("/readyz")
        first_contact = round(time.perf_counter() - start, 1)
        print(f"readyz {ready.status_code} after {first_contact}s")
        for q in load_questions(args.limit):
            start = time.perf_counter()
            response = client.post("/v1/ask", json={"question": q["question"]})
            seconds = round(time.perf_counter() - start, 2)
            body = response.json() if response.status_code == 200 else {}
            row = {
                "id": q["id"],
                "type": q["type"],
                "status": response.status_code,
                "seconds": seconds,
                "timings": body.get("timings", {}),
                "prompts": body.get("prompts"),
                **score(q, body),
            }
            rows.append(row)
            print(f"{q['id']:<8} {response.status_code} {row['label']:<20} {row['path']!s:<16} {seconds:>6.1f}s")

    # the deployed API reports its own prompt versions; record those, not the local checkout's
    deployed = next((r["prompts"] for r in rows if r.get("prompts")), None)
    summary = {"run": run_metadata(deployed), "url": base, "first_contact_seconds": first_contact,
               **summarise(rows)}  # fmt: skip
    (out / "live.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

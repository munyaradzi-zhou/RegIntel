"""Agent evaluation: `python -m tests.evaluation.evaluate_agent --label NAME [--set golden|tools|all] [--limit N]`.

Runs questions through the research agent (Agent Framework, guidance/CDISC/FAERS tools), the
writer and the verifier. Each finished question is appended to results/<label>/agent.jsonl, so a
run can be done in batches (--limit) or resumed after an interruption; the report is rebuilt from
that file every time.

- golden set (60): the same answer metrics as Phase 3, plus tool calls, latency and tokens;
- tool questions (10): FAERS numbers checked against values computed from the Parquet tables at
  run time, CDISC codes, whether the right tools were called, and wasted guidance searches.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from regintel.agent.factory import build_azure_agent, build_azure_service
from regintel.config import get_settings
from regintel.evaluation.answer_metrics import fact_found, score_answer, summarise_answers
from regintel.evaluation.retrieval_metrics import load_golden_set, section_matches
from regintel.faers.query import FaersStore

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

HERE = Path(__file__).parent
# List prices (USD per million tokens) for the estimate; check before quoting.
PRICE = {"input": 0.25, "cached": 0.025, "output": 2.00}


def faers_facts(spec: dict[str, Any], faers: FaersStore) -> list[str]:
    """Expected strings, computed from the data now (formatted as the tool formats them)."""
    if "summary" in spec:
        summary = faers.drug_summary(spec["summary"])
        values = {
            "reports": f"{summary.reports:,}",
            "serious": f"{summary.serious:,}",
            "deaths": f"{summary.deaths:,}",
        }
        facts = [values[f] for f in spec["fields"] if f in values]
        if "top3" in spec["fields"]:
            facts += [name for name, _ in summary.top_reactions[:3]]
        return facts
    drug, reaction = spec["ror"]
    result = faers.disproportionality(drug, reaction)
    if result.ror is None or result.ci95 is None:
        return ["cannot be computed"]
    return [f"{result.ror:.2f}", f"{result.ci95[0]:.2f}", f"{result.ci95[1]:.2f}"]


def _cost(record: dict[str, Any]) -> float:
    research = record["research_usage"]
    writer = record["writer_usage"]
    inputs = research.get("input_token_count", 0) + writer["input_tokens"]
    cached = research.get("openai.cached_input_tokens", 0) + writer["cached_tokens"]
    outputs = research.get("output_token_count", 0) + writer["output_tokens"]
    return ((inputs - cached) * PRICE["input"] + cached * PRICE["cached"] + outputs * PRICE["output"]) / 1e6


async def run_question(system: Any, qid: str, question: str) -> dict[str, Any]:
    """`system` is the research agent (answer) or the routed service (ask); both return the same fields."""
    answer = await (system.ask(question) if hasattr(system, "ask") else system.answer(question))
    result = answer.result
    return {
        "id": qid,
        "label": result.label,
        "answer": result.text,
        "calls": [asdict(c) for c in answer.calls],
        "research_seconds": round(answer.research_seconds, 2),
        "generation_seconds": round(result.generation_seconds, 2),
        "research_usage": answer.research_usage,
        "writer_usage": asdict(result.usage),
        "research_note": answer.research_note,
        "path": getattr(answer, "path", "agent"),
        "route_seconds": round(getattr(answer, "route_seconds", 0.0), 2),
        "cited": [[c.doc_id, c.section] for s in result.shown for c in s.citations],
        "statements": len(result.statements),
        "verified": sum(s.verified for s in result.statements),
        "problems": [p for s in result.statements for p in s.problems],
        "result": result,  # removed before writing; used for golden scoring
    }


def score_tool_question(q: dict[str, Any], record: dict[str, Any], faers: FaersStore) -> dict[str, Any]:
    facts = list(q["facts"]) + (faers_facts(q["faers"], faers) if "faers" in q else [])
    answered = record["label"] in ("grounded", "partially_grounded")
    missing = [f for f in facts if not (answered and fact_found(f, record["answer"]))]
    called = {c["name"] for c in record["calls"]}
    guidance_ok = all(
        any(doc == g["doc_id"] and any(section_matches(sec, s) for s in g["sections"]) for doc, sec in record["cited"])
        for g in q.get("guidance", [])
    )
    return {
        "facts_expected": facts,
        "facts_missing": missing,
        "all_facts": not missing,
        "tools_ok": set(q["tools"]) <= called,
        "wasted_guidance_search": q["no_guidance_search"] and "search_guidance" in called,
        "guidance_cited": guidance_ok,
    }


def load_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {r["id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if r}


def report(label: str) -> str:
    out = HERE / "results" / label
    records = load_records(out / "agent.jsonl")
    golden = [r for r in records.values() if "score" in r]
    tools = [r for r in records.values() if "tool_score" in r]
    n = max(1, len(records))
    lines = [
        f"# Agent evaluation — {label}",
        "",
        "Research agent (Microsoft Agent Framework, gpt-5-mini, low reasoning effort, parallel tool calls, "
        "max 8 tool calls; guidance search = knowledge base at minimal effort) -> writer (gpt-5-mini, prompt v3) "
        "-> verifier.",
        "",
        f"Questions run: {len(records)} ({len(golden)} golden, {len(tools)} tool).",
        "",
        "## Cost and latency (all questions)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| mean tool calls | {sum(len(r['calls']) for r in records.values()) / n:.2f} |",
        f"| mean research seconds | {sum(r['research_seconds'] for r in records.values()) / n:.1f} |",
        f"| mean writing seconds | {sum(r['generation_seconds'] for r in records.values()) / n:.1f} |",
        f"| est. USD per answer | {sum(_cost(r) for r in records.values()) / n:.4f} |",
    ]
    paths = sorted({str(r.get("path")) for r in records.values()})
    if len(paths) > 1:
        lines += [
            "",
            "## By path",
            "",
            "| Path | questions | mean seconds (route + research/retrieval + writing) |",
            "|---|---|---|",
        ]
        for name in paths:
            rows = [r for r in records.values() if str(r.get("path")) == name]
            seconds = sum(r.get("route_seconds", 0) + r["research_seconds"] + r["generation_seconds"] for r in rows)
            lines.append(f"| {name} | {len(rows)} | {seconds / len(rows):.1f} |")
    if golden:
        from regintel.evaluation.answer_metrics import AnswerScore

        scores = [AnswerScore(**r["score"]) for r in golden]
        lines += ["", "## Golden set (same metrics as Phase 3)", "", "| Metric | Value |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in summarise_answers(scores).items() if k != "tokens_per_question"]
    if tools:
        t = len(tools)
        mixed = [r for r in tools if r["id"].startswith("mx-")]
        lines += [
            "",
            "## Tool questions (FAERS numbers computed from the data at run time)",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| all expected facts stated | {sum(r['tool_score']['all_facts'] for r in tools) / t:.3f} |",
            f"| expected tools called | {sum(r['tool_score']['tools_ok'] for r in tools) / t:.3f} |",
            f"| wasted guidance searches | {sum(r['tool_score']['wasted_guidance_search'] for r in tools)} |",
            f"| guidance cited (mixed) | {sum(r['tool_score']['guidance_cited'] for r in mixed)}/{len(mixed)} |",
            "",
        ]
        for r in sorted(tools, key=lambda r: r["id"]):
            s = r["tool_score"]
            calls = ", ".join(c["name"] for c in r["calls"])
            lines.append(f"- **{r['id']}** [{r['label']}] calls: {calls}; missing: {s['facts_missing']}")
    text = "\n".join(lines)
    (out / "agent.md").write_text(text, encoding="utf-8")
    return text


async def run_all(
    todo: list[tuple[str, str]],
    golden: dict[str, Any],
    tools_by_id: dict[str, dict[str, Any]],
    path: Path,
    routed: bool,
) -> None:
    """One event loop for the whole run: the agent's async HTTP client is bound to the loop it started on."""
    settings = get_settings()
    agent = build_azure_service(settings) if routed else build_azure_agent(settings)
    faers = FaersStore(settings.data_dir / "processed" / "faers")
    for qid, question in todo:
        record = await run_question(agent, qid, question)
        result = record.pop("result")
        if qid in golden:
            record["score"] = asdict(score_answer(golden[qid], result))
        else:
            record["tool_score"] = score_tool_question(tools_by_id[qid], record, faers)
        with path.open("a", encoding="utf-8") as fh:
            print(json.dumps(record, ensure_ascii=False, default=str), file=fh)
        print(f"{qid:7} {record['label']:18} calls={len(record['calls'])} research={record['research_seconds']}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--set", choices=["golden", "tools", "all"], default="all")
    parser.add_argument("--limit", type=int, default=0, help="run at most N new questions, then report")
    parser.add_argument("--system", choices=["agent", "routed"], default="agent", help="agent only, or router + paths")
    args = parser.parse_args()

    out = HERE / "results" / args.label
    out.mkdir(parents=True, exist_ok=True)
    path = out / "agent.jsonl"
    done = load_records(path)

    golden = {q.id: q for q in load_golden_set(HERE / "datasets" / "golden_set.json").questions}
    tool_set = json.loads((HERE / "datasets" / "tool_questions.json").read_text(encoding="utf-8"))["questions"]
    todo: list[tuple[str, str]] = []
    if args.set in ("tools", "all"):
        todo += [(q["id"], q["question"]) for q in tool_set]
    if args.set in ("golden", "all"):
        todo += [(q.id, q.question) for q in golden.values()]
    todo = [t for t in todo if t[0] not in done]
    if args.limit:
        todo = todo[: args.limit]

    asyncio.run(run_all(todo, golden, {q["id"]: q for q in tool_set}, path, args.system == "routed"))
    print(report(args.label))


if __name__ == "__main__":
    main()

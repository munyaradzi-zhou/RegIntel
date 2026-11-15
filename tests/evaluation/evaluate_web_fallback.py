"""Web fallback: `python -m tests.evaluation.evaluate_web_fallback --label NAME`.

Runs the full service on questions the indexed guidance cannot answer (the golden set's
unanswerable questions, plus topics covered only on regulator websites) and checks that the
answer stays a decline, that pointers are offered, and that every pointer is on a regulator domain.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from regintel.agent.factory import build_azure_service
from regintel.config import get_settings
from regintel.evaluation.retrieval_metrics import load_golden_set
from regintel.service import on_regulator_domain

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

HERE = Path(__file__).parent
OUT_OF_CORPUS = [
    ("oc-01", "What has the FDA published on conducting clinical trials with decentralized elements?"),
    (
        "oc-02",
        "Has the EMA published a reflection paper on the use of artificial intelligence in the medicinal "
        "product lifecycle?",
    ),
    ("oc-03", "What does FDA guidance say about using real-world evidence to support regulatory decisions for drugs?"),
    ("oc-04", "What does Annex 2 of ICH E6(R3) cover?"),
]


async def run(label: str) -> None:
    service = build_azure_service(get_settings())
    questions = [(q.id, q.question) for q in load_golden_set(HERE / "datasets" / "golden_set.json").questions
                 if q.type == "unanswerable"] + OUT_OF_CORPUS  # fmt: skip
    rows = []
    for qid, question in questions:
        answer = await service.ask(question)
        pointers = answer.result.web_pointers
        rows.append({
            "id": qid, "label": answer.result.label, "path": answer.path,
            "pointers": [{"title": p.title, "url": p.url} for p in pointers],
            "all_on_regulator_domains": all(on_regulator_domain(p.url) for p in pointers),
            "web_seconds": round(answer.web_seconds, 1), "answer": answer.result.text,
        })  # fmt: skip
        print(f"{qid:6} {answer.result.label:18} pointers={len(pointers)} web={answer.web_seconds:.1f}s")

    declined = [r for r in rows if r["label"] in ("declined", "not_grounded")]
    summary = {
        "questions": len(rows),
        "declined": f"{len(declined)}/{len(rows)}",
        "declines_with_pointers": f"{sum(bool(r['pointers']) for r in declined)}/{len(declined)}",
        "pointers_on_regulator_domains": f"{sum(r['all_on_regulator_domains'] for r in rows)}/{len(rows)}",
        "mean_web_seconds": round(sum(r["web_seconds"] for r in declined) / max(1, len(declined)), 1),
    }
    out = HERE / "results" / label
    out.mkdir(parents=True, exist_ok=True)
    (out / "web_fallback.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [f"# Web fallback — {label}", "", *[f"- {k}: {v}" for k, v in summary.items()], ""]
    for r in rows:
        lines.append(f"- **{r['id']}** [{r['label']}]")
        lines += [f"  - {p['title']}: {p['url']}" for p in r["pointers"]]
    (out / "web_fallback.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    asyncio.run(run(parser.parse_args().label))


if __name__ == "__main__":
    main()

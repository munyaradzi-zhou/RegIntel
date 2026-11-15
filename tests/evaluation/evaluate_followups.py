"""Follow-up questions: `python -m tests.evaluation.evaluate_followups --label NAME`.

Two-turn conversations through the full chat service (rewriter, guard, router, verifier, reviewer).
Scores whether the second message was rewritten into a standalone question that names what it
refers to, whether the follow-up flag is right (one case is a topic switch that must not inherit
the first topic), and whether the second answer states the expected facts.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from regintel.api.app import build_chat
from regintel.config import get_settings
from regintel.evaluation.answer_metrics import fact_found
from regintel.faers.query import FaersStore
from tests.evaluation.evaluate_agent import faers_facts

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

HERE = Path(__file__).parent


async def run(label: str) -> None:
    chat = build_chat()
    faers = FaersStore(get_settings().data_dir / "processed" / "faers")
    conversations = json.loads((HERE / "datasets" / "followups.json").read_text(encoding="utf-8"))["conversations"]
    rows = []
    for conv in conversations:
        first = await chat.ask(conv["first"])
        second = await chat.ask(conv["second"], first.session_id)
        standalone = second.standalone.lower()
        facts = list(conv["facts"]) + (faers_facts(conv["faers"], faers) if "faers" in conv else [])
        answered = second.answer.result.label in ("grounded", "partially_grounded")
        missing = [f for f in facts if not (answered and fact_found(f, second.answer.result.text))]
        rows.append({
            "id": conv["id"], "standalone": second.standalone,
            "resolved": all(w.lower() in standalone for w in conv["resolves_to"]),
            "flag_ok": second.is_followup == conv["is_followup"],
            "label": second.answer.result.label, "path": second.answer.path,
            "facts_missing": missing, "all_facts": not missing,
            # a fact the user did not type must not be in the rewritten question (answer leakage)
            "leaked": [f for f in facts if f.lower() in standalone and f.lower() not in conv["second"].lower()],
        })  # fmt: skip
        r = rows[-1]
        print(
            f"{r['id']} resolved={r['resolved']} flag={r['flag_ok']} {r['label']} missing={missing} leak={r['leaked']}"
        )

    n = len(rows)
    summary = {
        "conversations": n,
        "rewrite_resolved": f"{sum(r['resolved'] for r in rows)}/{n}",
        "followup_flag_correct": f"{sum(r['flag_ok'] for r in rows)}/{n}",
        "second_answer_all_facts": f"{sum(r['all_facts'] for r in rows)}/{n}",
        "answer_facts_leaked_into_rewrite": sum(bool(r["leaked"]) for r in rows),
    }
    out = HERE / "results" / label
    out.mkdir(parents=True, exist_ok=True)
    (out / "followups.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [f"# Follow-up questions — {label}", "", *[f"- {k}: {v}" for k, v in summary.items()], ""]
    for r in rows:
        note = f" — missing {r['facts_missing']}" if r["facts_missing"] else ""
        lines.append(f"- {r['id']} [{r['label']}, {r['path']}] {r['standalone']}{note}")
    (out / "followups.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    asyncio.run(run(parser.parse_args().label))


if __name__ == "__main__":
    main()

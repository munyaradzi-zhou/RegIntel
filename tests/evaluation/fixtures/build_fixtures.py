"""Rebuild the CI gate's fixtures: `python -m tests.evaluation.fixtures.build_fixtures [--answers LABEL]`.

Writes, next to this file:
- chunks.jsonl.gz: the chunk store the gate retrieves from and verifies against;
- verifier_cases.json: real drafts from a recorded answer evaluation, plus mutated copies of their
  verified statements. Each mutation is either harmless (the verifier must keep the statement) or
  harmful (it must drop it); the builder refuses to write a case the current verifier disagrees with.

Needs the local data (`make ingest`) and a recorded answers run under tests/evaluation/results/.
Run it deliberately, then `make eval-ci ARGS=--update-baseline`, and commit all three files.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path

from regintel.answering.pipeline import verify_draft
from regintel.answering.schema import DraftAnswer, DraftCitation, DraftStatement
from regintel.config import get_settings
from regintel.ingestion.pipeline import read_chunks

HERE = Path(__file__).parent
RESULTS = HERE.parent / "results"
PER_KIND = 6  # mutated cases per kind
_INT = re.compile(r"(?<![\d.])\d+(?![\d.,])")


def recorded_cases(answers: dict[str, object], chunk_ids: set[str]) -> list[dict[str, object]]:
    by_id = {q.id: q.question for q in _golden()}
    cases = []
    for q in answers["questions"]:  # type: ignore[union-attr]
        if not q.get("draft") or not set(q["retrieved"]) <= chunk_ids:
            continue
        cases.append({"id": q["id"], "kind": "recorded", "question": by_id[q["id"]],
                      "passages": q["retrieved"], "draft": q["draft"]})  # fmt: skip
    return cases


def _golden():  # type: ignore[no-untyped-def]
    from regintel.evaluation.retrieval_metrics import load_golden_set

    return load_golden_set(HERE.parent / "datasets" / "golden_set.json").questions


def _one(statement: DraftStatement) -> DraftAnswer:
    return DraftAnswer(answerable=True, statements=[statement], decline_reason=None)


def mutations(statement: DraftStatement, n_passages: int) -> dict[str, tuple[DraftStatement, bool]]:
    """kind -> (mutated statement, should it still be verified)."""
    cite = statement.citations[0]
    words = cite.quote.split()
    out: dict[str, tuple[DraftStatement, bool]] = {}

    def with_quote(quote: str) -> DraftStatement:
        return DraftStatement(text=statement.text, citations=[DraftCitation(passage=cite.passage, quote=quote)])

    # Harmful: the verifier must drop these.
    if len(words) >= 6:
        swapped = [*words]
        swapped[len(words) // 2] = "never"
        out["misquote"] = (with_quote(" ".join(swapped)), False)
    match = _INT.search(statement.text)
    invented = (
        statement.text[: match.start()] + str(int(match.group()) + 7) + statement.text[match.end() :]
        if match
        else statement.text + " This applies within 999 days."
    )
    out["invented_number"] = (DraftStatement(text=invented, citations=[cite]), False)
    out["passage_not_retrieved"] = (
        DraftStatement(text=statement.text, citations=[DraftCitation(passage=n_passages + 1, quote=cite.quote)]),
        False,
    )
    out["no_citation"] = (DraftStatement(text=statement.text, citations=[]), False)
    if len(words) >= 4:
        out["too_short_quote"] = (with_quote(" ".join(words[:2])), False)
    # Harmless: extraction artefacts and typography the verifier is designed to tolerate.
    if len(words) >= 3:
        split = [*words]
        longest = max(range(len(words)), key=lambda i: len(words[i]))
        w = words[longest]
        if len(w) >= 6:
            split[longest] = w[:3] + " " + w[3:]
            out["split_word"] = (with_quote("  ".join(split)), True)
    out["case_and_typography"] = (with_quote(cite.quote.upper().replace("'", "\u2019").replace("-", "\u2013")), True)
    if len(words) >= 10:
        out["ellipsis"] = (with_quote(" ".join(words[:4]) + " ... " + " ".join(words[-4:])), True)
    if match and statement.text.count(match.group()) == 1:
        out["number_format"] = (
            DraftStatement(
                text=statement.text[: match.start()] + match.group() + ".0" + statement.text[match.end() :],
                citations=[cite],
            ),
            True,
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", default="2026-11-14-answers-v5", help="recorded run under results/")
    args = parser.parse_args()

    settings = get_settings()
    source = settings.data_dir / "processed" / "chunks.jsonl"
    chunks = {c.chunk_id: c for c in read_chunks(source)}
    # mtime=0 keeps the archive byte-identical when the chunks are, so its hash only moves with the content
    with gzip.GzipFile(HERE / "chunks.jsonl.gz", "wb", compresslevel=9, mtime=0) as out:
        out.write(source.read_bytes().replace(b"\r\n", b"\n"))

    answers = json.loads((RESULTS / args.answers / "answers.json").read_text(encoding="utf-8"))
    cases: list[dict[str, object]] = []
    per_kind: dict[str, int] = {}
    for rec in recorded_cases(answers, set(chunks)):
        draft = DraftAnswer.model_validate(rec["draft"])
        passages = [chunks[c] for c in rec["passages"]]  # type: ignore[union-attr]
        result = verify_draft(str(rec["question"]), draft, passages, None)
        cases.append({**rec, "expected_verified": [s.verified for s in result.statements],
                      "expected_label": result.label})  # fmt: skip
        for n, (checked, statement) in enumerate(zip(result.statements, draft.statements, strict=True), start=1):
            if not checked.verified or len(statement.citations) != 1:
                continue
            for kind, (mutated, keep) in mutations(statement, len(passages)).items():
                if per_kind.get(kind, 0) >= PER_KIND:
                    continue
                got = verify_draft(str(rec["question"]), _one(mutated), passages, None)
                if got.statements[0].verified != keep:
                    print(f"skipped {rec['id']} {kind}: verifier says {got.statements[0].verified}, intended {keep}")
                    continue
                per_kind[kind] = per_kind.get(kind, 0) + 1
                cases.append({"id": f"{rec['id']}-s{n}-{kind}", "kind": kind, "question": rec["question"],
                              "passages": rec["passages"], "draft": _one(mutated).model_dump(),
                              "expected_verified": [keep],
                              "expected_label": "grounded" if keep else "not_grounded"})  # fmt: skip
    fixture = {
        "description": "Recorded drafts and deliberate mutations; the verifier must keep and drop exactly these.",
        "source_run": args.answers,
        "cases": cases,
    }
    (HERE / "verifier_cases.json").write_text(json.dumps(fixture, indent=1, ensure_ascii=False) + "\n", "utf-8")
    kinds = {k: sum(1 for c in cases if c["kind"] == k) for k in sorted({str(c["kind"]) for c in cases})}
    print(f"{len(chunks)} chunks; {len(cases)} verifier cases: {kinds}")


if __name__ == "__main__":
    main()

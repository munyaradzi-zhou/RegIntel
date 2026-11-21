"""Replay an audited answer.

1. Sources: every passage the writer saw is rebuilt, corpus chunks from the chunk store (their
   sha256 must match the record, otherwise the corpus changed since) and tool records from the
   text kept in the record.
2. Prompts: the record's prompt versions are compared with the current ones.
3. Re-verification (deterministic, no model call): the recorded draft goes through the verifier
   again. Every statement must be kept or dropped exactly as recorded. Statements the reviewer
   dropped count as kept by the verifier, since the reviewer is a model and is not replayed here.
4. Optionally, regeneration: the writer drafts a new answer from the same passages. Models are not
   deterministic, so this shows differences rather than a pass or fail.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from regintel.answering.pipeline import is_tool_record, verify_draft
from regintel.answering.schema import DraftAnswer
from regintel.ingestion.chunking import Chunk
from regintel.prompts import prompt_tags

if TYPE_CHECKING:
    from regintel.answering.schema import AnswerResult
    from regintel.corpus.manifest import Manifest


@dataclass
class Replay:
    record_id: str
    sources_ok: bool
    source_problems: list[str]
    prompt_changes: dict[str, tuple[str, str]]  # name -> (recorded, current)
    decisions_match: bool
    decision_differences: list[str]
    regenerated: AnswerResult | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def reproduced(self) -> bool:
        return self.sources_ok and self.decisions_match


def rebuild_passages(record: dict[str, Any], chunks: dict[str, Chunk]) -> tuple[list[Chunk], list[str]]:
    passages, problems = [], []
    tool_records: dict[str, str] = record.get("tool_records", {})
    for source in record.get("sources", []):
        chunk_id, expected = source["chunk_id"], source.get("sha256", "")
        if is_tool_record(chunk_id):
            text = tool_records.get(chunk_id)
            if text is None:
                problems.append(f"{chunk_id}: tool record text missing from the audit record")
                continue
            chunk = Chunk(chunk_id, chunk_id.split(":")[0], "", "", chunk_id, 0, 0, text, f"[{chunk_id}]")
        elif (found := chunks.get(chunk_id)) is not None:
            chunk = found
        else:
            problems.append(f"{chunk_id}: no longer in the chunk store")
            continue
        actual = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
        if expected and actual != expected:
            problems.append(f"{chunk_id}: text changed since the answer (sha256 {expected[:12]} -> {actual[:12]})")
        passages.append(chunk)
    return passages, problems


def _kept_by_verifier(statement: dict[str, Any]) -> bool:
    return bool(statement["verified"]) or (
        bool(statement["problems"]) and all(p.startswith("reviewer:") for p in statement["problems"])
    )


def replay(record: dict[str, Any], chunks: dict[str, Chunk], manifest: Manifest | None = None) -> Replay:
    passages, problems = rebuild_passages(record, chunks)
    current = prompt_tags()
    changes = {k: (v, current.get(k, "missing")) for k, v in record.get("prompts", {}).items() if current.get(k) != v}
    rid = str(record.get("id", ""))
    if record.get("draft") is None:  # blocked or declined before a model wrote anything
        same = not record.get("statements")
        return Replay(rid, not problems, problems, changes, same, [] if same else ["statements without a draft"])
    draft = DraftAnswer.model_validate(record["draft"])
    result = verify_draft(record["standalone_question"], draft, passages, manifest)
    recorded = record.get("statements", [])
    differences = []
    if len(recorded) != len(result.statements):
        differences.append(f"{len(recorded)} statements recorded, {len(result.statements)} re-verified")
    for n, (was, now) in enumerate(zip(recorded, result.statements, strict=False), start=1):
        if _kept_by_verifier(was) != now.verified:
            state = "kept" if now.verified else f"dropped ({'; '.join(now.problems)})"
            differences.append(f"S{n}: recorded {'kept' if _kept_by_verifier(was) else 'dropped'}, now {state}")
    return Replay(rid, not problems, problems, changes, not differences, differences)

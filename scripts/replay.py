"""Replay an audited answer: rebuild its sources, re-run the verifier on the recorded draft, and
optionally regenerate the answer from the same passages.

    python scripts/replay.py <audit_id>                 # from Cosmos DB (COSMOS_ENDPOINT, az login)
    python scripts/replay.py --file record.json         # a record exported earlier
    python scripts/replay.py <audit_id> --regenerate    # also draft a new answer (model calls)

Exit code 0 when the answer is reproduced (sources unchanged, every verifier decision the same),
1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from regintel.audit.replay import replay
from regintel.config import get_settings
from regintel.corpus.manifest import load_manifest
from regintel.ingestion.pipeline import read_chunks

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass


def load_record(audit_id: str | None, file: Path | None) -> dict[str, Any]:
    if file is not None:
        record: dict[str, Any] = json.loads(file.read_text(encoding="utf-8"))
        return record
    from azure.identity import AzureCliCredential

    from regintel.audit.store import CosmosAuditStore

    settings = get_settings()
    if not settings.cosmos_endpoint or not audit_id:
        raise SystemExit("pass an audit id with COSMOS_ENDPOINT set, or --file")
    found = CosmosAuditStore.connect(settings.cosmos_endpoint, AzureCliCredential()).get(audit_id)
    if found is None:
        raise SystemExit(f"no audit record {audit_id}")
    return found


def regenerate(record: dict[str, Any], passages: list[Any]) -> None:
    from azure.identity import AzureCliCredential, get_bearer_token_provider

    from regintel.answering.models import answer_model_from_settings
    from regintel.answering.pipeline import verify_draft
    from regintel.answering.review import reviewer_from_settings
    from regintel.retrieval.base import Hit

    settings = get_settings()
    tokens = get_bearer_token_provider(AzureCliCredential(), "https://cognitiveservices.azure.com/.default")
    question = record["standalone_question"]
    draft, usage = answer_model_from_settings(settings, tokens).draft(question, [Hit(c, 1.0) for c in passages])
    result = verify_draft(question, draft, passages, None, usage)
    if (reviewer := reviewer_from_settings(settings, tokens)) is not None:
        result = reviewer.review(result, [c.text for c in passages])
    print(f"\nRegenerated: {result.label} (recorded: {record['label']})")
    before = {s["text"] for s in record["statements"] if s["verified"]}
    after = {s.text for s in result.shown}
    for text in sorted(before - after):
        print(f"  - only in the record: {text}")
    for text in sorted(after - before):
        print(f"  + only in the replay: {text}")
    if before == after:
        print("  same statements shown")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_id", nargs="?")
    parser.add_argument("--file", type=Path)
    parser.add_argument("--regenerate", action="store_true", help="draft a new answer from the same passages")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    settings = get_settings()
    record = load_record(args.audit_id, args.file)
    chunks = {c.chunk_id: c for c in read_chunks(settings.data_dir / "processed" / "chunks.jsonl")}
    result = replay(record, chunks, load_manifest(settings.data_dir / "manifest.json"))

    print(f"Audit record {result.record_id} · {record['at']} · {record['label']} · route {record['route']['path']}")
    print(f"Question: {record['standalone_question']}")
    print(f"Sources: {len(record['sources'])}, {'unchanged' if result.sources_ok else 'CHANGED'}")
    for problem in result.source_problems:
        print(f"  ! {problem}")
    for name, (was, now) in result.prompt_changes.items():
        print(f"  ! prompt {name}: recorded {was}, current {now}")
    print(f"Verifier decisions: {'identical' if result.decisions_match else 'DIFFERENT'}")
    for difference in result.decision_differences:
        print(f"  ! {difference}")
    if args.regenerate:
        from regintel.audit.replay import rebuild_passages

        regenerate(record, rebuild_passages(record, chunks)[0])
    print("\nReproduced." if result.reproduced else "\nNot reproduced.")
    sys.exit(0 if result.reproduced else 1)


if __name__ == "__main__":
    main()

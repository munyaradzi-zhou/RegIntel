"""Turn the downloaded guidance PDFs into chunks: `python -m regintel.ingestion.pipeline`.

Output is one JSON object per line in data/processed/chunks.jsonl, which both the local
index and Azure AI Search are built from, so the two backends always search the same text.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING

import structlog

from regintel.config import get_settings
from regintel.corpus.manifest import load_manifest
from regintel.ingestion.chunking import Chunk, chunk_document
from regintel.ingestion.pdf_text import read_pdf

if TYPE_CHECKING:
    from pathlib import Path

log = structlog.get_logger()


def build_chunks(data_dir: Path) -> list[Chunk]:
    manifest = load_manifest(data_dir / "manifest.json")
    chunks: list[Chunk] = []
    for doc in manifest.guidance:
        path = data_dir / "raw" / "guidance" / f"{doc.id}.pdf"
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing: run `make corpus` first")
        doc_chunks = chunk_document(doc, read_pdf(path))
        log.info("chunked", doc=doc.id, chunks=len(doc_chunks))
        chunks.extend(doc_chunks)
    return chunks


def write_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def read_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as fh:
        return [Chunk(**json.loads(line)) for line in fh if line.strip()]


def main() -> None:
    data_dir = get_settings().data_dir
    chunks = build_chunks(data_dir)
    out = data_dir / "processed" / "chunks.jsonl"
    write_chunks(chunks, out)
    log.info("done", chunks=len(chunks), documents=len({c.doc_id for c in chunks}), path=str(out))


if __name__ == "__main__":
    main()

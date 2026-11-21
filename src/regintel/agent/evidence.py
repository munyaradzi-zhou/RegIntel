"""Evidence gathered during one question: every tool result, numbered in the order it arrived.

Guidance passages keep their real chunk (document, section, pages) so citations and the
evaluation work exactly as in Phase 3. Terminology entries, FAERS computations and openFDA labels and recalls become
chunk-shaped records with their own source label, so the same writer and verifier handle them.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Literal

from regintel.ingestion.chunking import Chunk

Kind = Literal["guidance", "terminology", "faers", "label", "recall"]


@dataclass(frozen=True)
class Evidence:
    number: int  # [E1], [E2], ...
    kind: Kind
    chunk: Chunk

    def render(self) -> str:
        return f"[E{self.number}] {self.chunk.header}\n{self.chunk.text}"


class EvidencePool:
    """Thread-safe: the agent may run tools concurrently."""

    def __init__(self) -> None:
        self._items: list[Evidence] = []
        self._by_id: dict[str, Evidence] = {}
        self._lock = threading.Lock()

    def add(self, kind: Kind, chunk: Chunk) -> Evidence:
        with self._lock:
            if chunk.chunk_id in self._by_id:  # the same passage found by two searches keeps its number
                return self._by_id[chunk.chunk_id]
            item = Evidence(len(self._items) + 1, kind, chunk)
            self._items.append(item)
            self._by_id[chunk.chunk_id] = item
            return item

    def add_record(self, kind: Kind, record_id: str, label: str, text: str) -> Evidence:
        """A non-guidance result (terminology entry, FAERS computation) as a citable record."""
        header = f"[{label}]"
        return self.add(kind, Chunk(record_id, kind, "", "", label, 0, 0, text, header))

    @property
    def items(self) -> list[Evidence]:
        with self._lock:
            return list(self._items)

    def chunks(self) -> list[Chunk]:
        return [item.chunk for item in self.items]

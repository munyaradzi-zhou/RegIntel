"""The retriever interface every backend implements, so all of them are evaluated the same way.

Local baselines (BM25, dense, hybrid) and the Azure AI Search knowledge base return the same
`Hit` shape: a chunk and a score. The evaluation harness only sees this interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from regintel.ingestion.chunking import Chunk


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class Retriever(Protocol):
    name: str

    def search(self, query: str, top_k: int = 10) -> list[Hit]: ...

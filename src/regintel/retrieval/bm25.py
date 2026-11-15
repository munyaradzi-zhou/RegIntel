"""Keyword retrieval with BM25 (Okapi), over each chunk's contextual header and text."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from rank_bm25 import BM25Okapi

from regintel.retrieval.base import Hit

if TYPE_CHECKING:
    from regintel.ingestion.chunking import Chunk

# Keep regulatory tokens whole: "80.00", "E9(R1)", "312.32", "III.A.1", "non-serious".
_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-()][a-z0-9]+\)?)*")
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "has", "have", "how",
    "if", "in", "is", "it", "its", "may", "must", "of", "on", "or", "should", "that", "the", "their", "there",
    "these", "this", "to", "under", "was", "were", "what", "when", "which", "who", "will", "with", "would",
})  # fmt: skip


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self._index = BM25Okapi([tokenize(c.search_text) for c in chunks], k1=k1, b=b)

    def search(self, query: str, top_k: int = 10) -> list[Hit]:
        scores = self._index.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [Hit(self.chunks[i], float(scores[i])) for i in ranked if scores[i] > 0]

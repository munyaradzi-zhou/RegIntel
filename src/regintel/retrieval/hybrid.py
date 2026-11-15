"""Hybrid retrieval: Reciprocal Rank Fusion of several retrievers.

RRF uses ranks, not scores, so BM25 scores and cosine similarities need no calibration:
score(d) = sum over retrievers of 1 / (k + rank(d)). k=60 is the value from the original
paper (Cormack et al., 2009) and what Azure AI Search uses for its hybrid queries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from regintel.retrieval.base import Hit

if TYPE_CHECKING:
    from regintel.ingestion.chunking import Chunk
    from regintel.retrieval.base import Retriever


def reciprocal_rank_fusion(rankings: list[list[Hit]], k: int = 60) -> list[Hit]:
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            key = hit.chunk.chunk_id
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            chunks[key] = hit.chunk
    ordered = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [Hit(chunks[key], scores[key]) for key in ordered]


class HybridRetriever:
    name = "hybrid"

    def __init__(self, retrievers: list[Retriever], candidates: int = 50, k: int = 60) -> None:
        self.retrievers = retrievers
        self.candidates = candidates
        self.k = k

    def search(self, query: str, top_k: int = 10) -> list[Hit]:
        rankings = [r.search(query, top_k=self.candidates) for r in self.retrievers]
        return reciprocal_rank_fusion(rankings, k=self.k)[:top_k]

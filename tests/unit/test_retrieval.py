from pathlib import Path

import numpy as np

from regintel.ingestion.chunking import Chunk
from regintel.retrieval.base import Hit
from regintel.retrieval.bm25 import BM25Retriever, tokenize
from regintel.retrieval.dense import DenseRetriever
from regintel.retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion


def chunk(cid: str, text: str, doc: str = "doc", section: str = "1") -> Chunk:
    return Chunk(cid, doc, "EU", section, "", 1, 1, text, f"[{doc}]")


CHUNKS = [
    chunk("a", "The acceptance interval for bioequivalence is 80.00-125.00%."),
    chunk("b", "Serious adverse reactions are reported within 15 calendar days."),
    chunk("c", "Audit trails record GMP-relevant changes and deletions."),
]


def test_tokenizer_keeps_regulatory_tokens_whole() -> None:
    assert tokenize("Is the 80.00 limit in E9(R1) and 21 CFR 312.32?") == [
        "80.00",
        "limit",
        "e9(r1)",
        "21",
        "cfr",
        "312.32",
    ]


def test_bm25_ranks_matching_chunk_first_and_drops_zero_scores() -> None:
    hits = BM25Retriever(CHUNKS).search("audit trails for deletions", top_k=3)
    assert hits[0].chunk.chunk_id == "c"
    assert all(h.score > 0 for h in hits)


class FakeEmbedder:
    """Maps a text to a one-hot vector by keyword, so similarity is predictable."""

    KEYWORDS = ("bioequivalence", "adverse", "audit")

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        return np.array([[float(k in t.lower()) for k in self.KEYWORDS] for t in texts], dtype=np.float32)


def test_dense_retrieval_and_cache(tmp_path: Path) -> None:
    embedder = FakeEmbedder()
    first = DenseRetriever(CHUNKS, embedder, cache_dir=tmp_path, model="fake")
    assert first.search("adverse event timelines", top_k=1)[0].chunk.chunk_id == "b"
    calls = embedder.calls
    DenseRetriever(CHUNKS, embedder, cache_dir=tmp_path, model="fake")  # corpus vectors come from the cache
    assert embedder.calls == calls
    assert len(list(tmp_path.glob("embeddings-*.npy"))) == 1


def test_rrf_rewards_agreement_across_rankings() -> None:
    a, b, c = (Hit(ch, 0.0) for ch in CHUNKS)
    fused = reciprocal_rank_fusion([[a, b, c], [b, a, c], [b, c, a]], k=60)
    assert [h.chunk.chunk_id for h in fused] == ["b", "a", "c"]
    assert fused[0].score == 1 / 62 + 1 / 61 + 1 / 61


def test_hybrid_combines_retrievers(tmp_path: Path) -> None:
    hybrid = HybridRetriever([BM25Retriever(CHUNKS), DenseRetriever(CHUNKS, FakeEmbedder())])
    assert hybrid.search("bioequivalence acceptance interval", top_k=1)[0].chunk.chunk_id == "a"

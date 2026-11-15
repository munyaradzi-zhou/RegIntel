"""Dense retrieval with a local embedding model (fastembed, CPU).

Embeddings are cached next to the chunks and keyed by a hash of the chunk texts and model
name, so re-running an evaluation does not re-embed the corpus.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Protocol

import numpy as np

from regintel.retrieval.base import Hit

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from regintel.ingestion.chunking import Chunk

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
# BGE models retrieve better when the query (not the passage) carries this instruction.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class FastEmbedEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        from fastembed import TextEmbedding  # optional dependency: the `local` extra

        self.model_name = model_name
        self._model = TextEmbedding(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors: Iterable[np.ndarray] = self._model.embed(texts, batch_size=64)
        return np.asarray(list(vectors), dtype=np.float32)


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms)


def corpus_key(chunks: list[Chunk], model_name: str) -> str:
    digest = hashlib.sha256(model_name.encode())
    for chunk in chunks:
        digest.update(chunk.search_text.encode())
    return digest.hexdigest()[:16]


class DenseRetriever:
    name = "dense"

    def __init__(self, chunks: list[Chunk], embedder: Embedder, cache_dir: Path | None = None, model: str = "") -> None:
        self.chunks = chunks
        self.embedder = embedder
        self._matrix = _normalise(self._load_or_embed(cache_dir, model))

    def _load_or_embed(self, cache_dir: Path | None, model: str) -> np.ndarray:
        cache = cache_dir / f"embeddings-{corpus_key(self.chunks, model)}.npy" if cache_dir else None
        if cache is not None and cache.exists():
            return np.asarray(np.load(cache), dtype=np.float32)
        matrix = self.embedder.embed([c.search_text for c in self.chunks])
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, matrix)
        return matrix

    def search(self, query: str, top_k: int = 10) -> list[Hit]:
        query_vector = _normalise(self.embedder.embed([QUERY_PREFIX + query]))[0]
        scores = self._matrix @ query_vector
        ranked = np.argsort(-scores)[:top_k]
        return [Hit(self.chunks[int(i)], float(scores[i])) for i in ranked]

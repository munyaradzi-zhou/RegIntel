"""A knowledge base as a `Retriever`, so it is scored by the same harness as the local baselines.

The retrieve response has two parts: `references` (one per passage, with a reranker score and a
pointer to its source) and `response` (the passage texts, keyed by reference id). Each reference
becomes a `Hit`:

- section-index references carry RegIntel's own chunk id (`docKey`), so the chunk is exact;
- Blob references only carry the PDF's URL. The document is exact, but Azure's chunks have no
  section or page, so the section is *inferred*: the snippet is matched to the RegIntel chunk of
  the same document that shares the most words with it. Section scores for the Blob knowledge
  base are therefore approximate. That the auto pipeline loses citation locators is a finding.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from azure.core.exceptions import HttpResponseError
from azure.search.documents.knowledgebases.models import (
    KnowledgeBaseMessage,
    KnowledgeBaseMessageTextContent,
    KnowledgeBaseRetrievalRequest,
    KnowledgeRetrievalLowReasoningEffort,
    KnowledgeRetrievalMediumReasoningEffort,
    KnowledgeRetrievalMinimalReasoningEffort,
    KnowledgeRetrievalReasoningEffort,
    KnowledgeRetrievalSemanticIntent,
)
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from regintel.ingestion.chunking import Chunk
from regintel.retrieval.base import Hit

if TYPE_CHECKING:
    from collections.abc import Callable

    from azure.search.documents.knowledgebases import KnowledgeBaseRetrievalClient

_WORD = re.compile(r"[a-z0-9]+")
WEB_SECTION = "web"  # section marker for Web knowledge source hits (doc_id is "web:<url>")

Effort = Literal["minimal", "low", "medium"]
_EFFORTS: dict[str, Callable[[], KnowledgeRetrievalReasoningEffort]] = {
    "minimal": KnowledgeRetrievalMinimalReasoningEffort,
    "low": KnowledgeRetrievalLowReasoningEffort,
    "medium": KnowledgeRetrievalMediumReasoningEffort,
}


def is_rate_limited(error: BaseException) -> bool:
    """The planner's 429 reaches us wrapped in the knowledge base's own error, so check the message too."""
    return isinstance(error, HttpResponseError) and (error.status_code == 429 or "429" in str(error))


def retrieval_request(query: str, effort: Effort | None = None) -> KnowledgeBaseRetrievalRequest:
    """`minimal` skips LLM query planning, so the query goes in as a semantic intent, not a message.
    `None` uses the effort configured on the knowledge base."""
    if effort == "minimal":
        return KnowledgeBaseRetrievalRequest(
            intents=[KnowledgeRetrievalSemanticIntent(search=query)],
            retrieval_reasoning_effort=KnowledgeRetrievalMinimalReasoningEffort(),
            include_activity=True,
        )
    return KnowledgeBaseRetrievalRequest(
        messages=[KnowledgeBaseMessage(role="user", content=[KnowledgeBaseMessageTextContent(text=query)])],
        retrieval_reasoning_effort=_EFFORTS[effort]() if effort else None,
        include_activity=True,
    )


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def doc_id_from_blob_url(url: str) -> str:
    return url.rsplit("/", 1)[-1].removesuffix(".pdf")


def chunk_id_from_doc_key(key: str) -> str:
    """Index keys replaced '#' with '_' ('fda-be-anda_0063' -> 'fda-be-anda#0063')."""
    head, _, tail = key.rpartition("_")
    return f"{head}#{tail}"


def best_matching_chunk(snippet: str, candidates: list[Chunk]) -> Chunk | None:
    """The candidate containing the largest share of the snippet's words (containment, not Jaccard,
    because an auto chunk can be much shorter or longer than ours)."""
    words = _words(snippet)
    if not words or not candidates:
        return None
    best = max(candidates, key=lambda c: len(words & _words(c.text)))
    return best if words & _words(best.text) else None


@dataclass
class RetrievalCost:
    """Token use reported by the knowledge base's activity log, summed over calls."""

    calls: int = 0
    planning_input_tokens: int = 0
    planning_output_tokens: int = 0
    reasoning_tokens: int = 0
    subqueries: int = 0
    web_calls: int = 0
    web_summary_tokens: int = 0
    planning_ms: list[int] = field(default_factory=list)

    def add(self, activity: list[dict[str, Any]]) -> None:
        self.calls += 1
        for record in activity:
            kind = record.get("type")
            if kind == "modelQueryPlanning":
                self.planning_input_tokens += int(record.get("inputTokens") or 0)
                self.planning_output_tokens += int(record.get("outputTokens") or 0)
                self.planning_ms.append(int(record.get("elapsedMs") or 0))
            elif kind == "agenticReasoning":
                self.reasoning_tokens += int(record.get("reasoningTokens") or 0)
            elif kind in ("searchIndex", "azureBlob", "web"):
                self.subqueries += 1
            if kind == "web":
                self.web_calls += 1
            elif kind == "modelWebSummarization":
                self.web_summary_tokens += int(record.get("inputTokens") or 0) + int(record.get("outputTokens") or 0)


class KnowledgeBaseRetriever:
    def __init__(
        self, name: str, client: KnowledgeBaseRetrievalClient, chunks: list[Chunk], effort: Effort | None = None
    ) -> None:
        self.name = name
        self.client = client
        self.effort = effort
        self.by_id = {c.chunk_id: c for c in chunks}
        self.by_doc: dict[str, list[Chunk]] = {}
        for chunk in chunks:
            self.by_doc.setdefault(chunk.doc_id, []).append(chunk)
        self.cost = RetrievalCost()

    @retry(
        retry=retry_if_exception(is_rate_limited),
        wait=wait_exponential(multiplier=5, max=60),
        stop=stop_after_attempt(8),
        reraise=True,
    )
    def retrieve_raw(self, query: str) -> dict[str, Any]:
        result: dict[str, Any] = self.client.retrieve(retrieval_request(query, self.effort)).as_dict()
        return result

    def search(self, query: str, top_k: int = 10) -> list[Hit]:
        raw = self.retrieve_raw(query)
        self.cost.add(raw.get("activity", []))
        return self.to_hits(raw)[:top_k]

    def to_hits(self, raw: dict[str, Any]) -> list[Hit]:
        texts = passage_texts(raw)
        references = sorted(raw.get("references", []), key=lambda r: r.get("rerankerScore") or 0.0, reverse=True)
        hits = []
        for ref in references:
            chunk = self._chunk_for(ref, texts.get(str(ref.get("id")), ""))
            if chunk is not None:
                hits.append(Hit(chunk, float(ref.get("rerankerScore") or 0.0)))
        return hits

    def _chunk_for(self, ref: dict[str, Any], text: str) -> Chunk | None:
        if ref.get("type") == "searchIndex" and ref.get("docKey"):
            return self.by_id.get(chunk_id_from_doc_key(str(ref["docKey"])))
        if ref.get("type") == "azureBlob" and ref.get("blobUrl"):
            doc_id = doc_id_from_blob_url(str(ref["blobUrl"]))
            match = best_matching_chunk(text, self.by_doc.get(doc_id, []))
            section = match.section if match else "unknown"
            return Chunk(f"{doc_id}@blob{ref.get('id')}", doc_id, "", section, "", 0, 0, text, "")
        if ref.get("type") == "web" and ref.get("url"):
            # An LLM summary of a web page: a pointer for the answer, never quotable evidence.
            url = str(ref["url"])
            return Chunk(
                f"web@{ref.get('id')}", f"web:{url}", "", WEB_SECTION, str(ref.get("title") or ""), 0, 0, text, ""
            )
        return None


def passage_texts(raw: dict[str, Any]) -> dict[str, str]:
    """Passage text by reference id, from the JSON list the extractive response carries."""
    texts: dict[str, str] = {}
    for message in raw.get("response", []):
        for content in message.get("content", []):
            if content.get("type") != "text":
                continue
            try:
                passages = json.loads(content.get("text") or "[]")
            except json.JSONDecodeError:
                continue
            for passage in passages if isinstance(passages, list) else []:
                texts[str(passage.get("ref_id"))] = str(passage.get("content") or "")
    return texts

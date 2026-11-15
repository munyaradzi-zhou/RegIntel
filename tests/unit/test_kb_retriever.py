import json
from typing import Any

from regintel.azure_search.kb_retriever import (
    KnowledgeBaseRetriever,
    RetrievalCost,
    best_matching_chunk,
    chunk_id_from_doc_key,
    doc_id_from_blob_url,
    passage_texts,
    retrieval_request,
)
from regintel.ingestion.chunking import Chunk


def chunk(cid: str, doc: str, section: str, text: str) -> Chunk:
    return Chunk(cid, doc, "EU", section, "", 1, 1, text, "header")


CHUNKS = [
    chunk("ema-be#0001", "ema-be", "4.1.1", "a randomised two-period two-sequence single dose crossover design"),
    chunk("ema-be#0002", "ema-be", "4.1.8", "the 90% confidence interval should be contained within 80.00-125.00%"),
    chunk("fda-be#0001", "fda-be", "III.A", "fasting and fed studies"),
]


def response(references: list[dict[str, Any]], passages: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "response": [{"content": [{"type": "text", "text": json.dumps(passages)}]}],
        "references": references,
        "activity": [],
    }


def test_key_and_url_parsing() -> None:
    assert chunk_id_from_doc_key("fda-be-anda_0063") == "fda-be-anda#0063"
    assert doc_id_from_blob_url("https://x.blob.core.windows.net/regintel-guidance/ema-bioequivalence.pdf") == (
        "ema-bioequivalence"
    )


def test_snippet_aligned_to_most_overlapping_section() -> None:
    snippet = "For these parameters the 90% confidence interval must be within the acceptance interval"
    match = best_matching_chunk(snippet, [c for c in CHUNKS if c.doc_id == "ema-be"])
    assert match is not None and match.section == "4.1.8"
    assert best_matching_chunk("", CHUNKS) is None


def test_hits_from_both_reference_types_ordered_by_reranker_score() -> None:
    raw = response(
        references=[
            {"type": "azureBlob", "id": "1", "rerankerScore": 2.0, "blobUrl": "https://x/c/ema-be.pdf"},
            {"type": "searchIndex", "id": "0", "rerankerScore": 3.0, "docKey": "fda-be_0001"},
            {"type": "searchIndex", "id": "2", "rerankerScore": 1.0, "docKey": "missing_0009"},
        ],
        passages=[{"ref_id": 1, "content": "a two-period crossover design is recommended"}],
    )
    retriever = KnowledgeBaseRetriever("kb", client=None, chunks=CHUNKS)  # type: ignore[arg-type]
    hits = retriever.to_hits(raw)
    assert [h.chunk.doc_id for h in hits] == ["fda-be", "ema-be"]  # unknown key dropped
    assert hits[1].chunk.section == "4.1.1"  # inferred from the snippet
    assert hits[1].chunk.text == "a two-period crossover design is recommended"


def test_passage_texts_tolerates_non_json() -> None:
    assert passage_texts({"response": [{"content": [{"type": "text", "text": "not json"}]}]}) == {}


def test_cost_accumulates_activity() -> None:
    cost = RetrievalCost()
    cost.add([
        {"type": "modelQueryPlanning", "inputTokens": 900, "outputTokens": 100, "elapsedMs": 1500},
        {"type": "searchIndex"}, {"type": "searchIndex"},
        {"type": "agenticReasoning", "reasoningTokens": 60000},
    ])  # fmt: skip
    assert (cost.calls, cost.planning_input_tokens, cost.subqueries, cost.reasoning_tokens) == (1, 900, 2, 60000)


def test_minimal_effort_sends_an_intent_not_a_message() -> None:
    minimal = retrieval_request("EU BE acceptance range", "minimal").as_dict()
    assert minimal["intents"] == [{"type": "semantic", "search": "EU BE acceptance range"}]
    assert "messages" not in minimal
    medium = retrieval_request("EU BE acceptance range", "medium").as_dict()
    assert medium["retrievalReasoningEffort"] == {"kind": "medium"}
    assert medium["messages"][0]["content"][0]["text"] == "EU BE acceptance range"
    assert "retrievalReasoningEffort" not in retrieval_request("q").as_dict()  # the KB's own setting


def test_web_reference_becomes_a_non_citable_pointer() -> None:
    raw = response(
        references=[{"type": "web", "id": "0", "url": "https://www.fda.gov/x", "title": "New guidance"}],
        passages=[{"ref_id": 0, "content": "summary of the page"}],
    )
    hits = KnowledgeBaseRetriever("kb", client=None, chunks=CHUNKS).to_hits(raw)  # type: ignore[arg-type]
    assert hits[0].chunk.doc_id == "web:https://www.fda.gov/x" and hits[0].chunk.section == "web"
    cost = RetrievalCost()
    cost.add([{"type": "web"}, {"type": "modelWebSummarization", "inputTokens": 100, "outputTokens": 20}])
    assert (cost.web_calls, cost.web_summary_tokens, cost.subqueries) == (1, 120, 1)

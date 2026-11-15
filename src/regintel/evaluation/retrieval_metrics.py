"""Retrieval metrics against the golden set.

A hit is judged at two granularities:

- document: the chunk comes from an expected document;
- section: it also sits in an expected section (a locator prefix: "4.1" matches "4.1.8").
  Questions whose expected sections are empty count at document level only.

For comparison questions every expected document must be found, so "all documents found
in the top k" is reported separately from "first relevant rank". Multi-hop questions need
several specific passages (two or three documents, or two sections of one document), so they
are scored on "every expected source found at section level in the top k".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    from pathlib import Path

    from regintel.retrieval.base import Hit


class ExpectedSource(BaseModel):
    doc_id: str
    sections: list[str]


class GoldenQuestion(BaseModel):
    id: str
    type: Literal["single", "comparison", "multihop", "unanswerable"]
    question: str
    expected: list[ExpectedSource]
    facts: list[str]


class GoldenSet(BaseModel):
    description: str
    version: str
    questions: list[GoldenQuestion]

    def answerable(self) -> list[GoldenQuestion]:
        return [q for q in self.questions if q.type != "unanswerable"]


def load_golden_set(path: Path) -> GoldenSet:
    return GoldenSet.model_validate(json.loads(path.read_text(encoding="utf-8")))


def section_matches(locator: str, expected: str) -> bool:
    """'4.1' matches '4.1' and '4.1.8' but not '4.10'; 'III.A' matches 'III.A.10'."""
    return locator == expected or locator.startswith(expected + ".")


def doc_hit(hit: Hit, source: ExpectedSource) -> bool:
    return hit.chunk.doc_id == source.doc_id


def section_hit(hit: Hit, source: ExpectedSource) -> bool:
    if not doc_hit(hit, source):
        return False
    return not source.sections or any(section_matches(hit.chunk.section, s) for s in source.sections)


@dataclass
class QuestionResult:
    id: str
    type: str
    doc_rank: int | None  # 1-based rank of the first chunk from any expected document
    section_rank: int | None  # 1-based rank of the first chunk in an expected section
    all_docs_found: bool  # every expected document appears in the top k
    all_sources_found: bool  # every expected source is matched at section level in the top k
    retrieved: list[str] = field(default_factory=list)  # "doc_id §section" of the top k


def score_question(question: GoldenQuestion, hits: list[Hit]) -> QuestionResult:
    def first_rank(test: str) -> int | None:
        for rank, hit in enumerate(hits, start=1):
            check = doc_hit if test == "doc" else section_hit
            if any(check(hit, source) for source in question.expected):
                return rank
        return None

    found_docs = {hit.chunk.doc_id for hit in hits}
    return QuestionResult(
        id=question.id,
        type=question.type,
        doc_rank=first_rank("doc"),
        section_rank=first_rank("section"),
        all_docs_found=all(source.doc_id in found_docs for source in question.expected),
        all_sources_found=all(any(section_hit(h, source) for h in hits) for source in question.expected),
        retrieved=[f"{h.chunk.doc_id} §{h.chunk.section}" for h in hits],
    )


def _hit_at(ranks: list[int | None], k: int) -> float:
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks) if ranks else 0.0


def _mrr(ranks: list[int | None]) -> float:
    return sum(1 / r for r in ranks if r is not None) / len(ranks) if ranks else 0.0


def summarise(results: list[QuestionResult], k: int) -> dict[str, float | int]:
    doc_ranks = [r.doc_rank for r in results]
    section_ranks = [r.section_rank for r in results]
    comparisons = [r for r in results if r.type == "comparison"]
    multihop = [r for r in results if r.type == "multihop"]
    return {
        "questions": len(results),
        "doc_hit@1": round(_hit_at(doc_ranks, 1), 3),
        f"doc_hit@{k}": round(_hit_at(doc_ranks, k), 3),
        "section_hit@1": round(_hit_at(section_ranks, 1), 3),
        f"section_hit@{k}": round(_hit_at(section_ranks, k), 3),
        "doc_mrr": round(_mrr(doc_ranks), 3),
        "section_mrr": round(_mrr(section_ranks), 3),
        f"comparison_all_docs@{k}": round(sum(r.all_docs_found for r in comparisons) / len(comparisons), 3)
        if comparisons
        else 0.0,
        f"multihop_all_sources@{k}": round(sum(r.all_sources_found for r in multihop) / len(multihop), 3)
        if multihop
        else 0.0,
    }

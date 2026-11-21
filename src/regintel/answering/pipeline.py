"""Retrieve, draft, verify, label.

Labels:
- grounded            every statement verified
- partially_grounded  some statements dropped by verification; the rest are shown, flagged
- not_grounded        the model answered but nothing survived verification; withheld (fail closed)
- declined            the model said the passages do not answer the question, or nothing was retrieved
- blocked             the input guard stopped the question before any model call
"""

from __future__ import annotations

import hashlib
import time
from typing import TYPE_CHECKING

from regintel.answering.schema import AnswerResult, Label, Usage
from regintel.answering.verify import check_statement
from regintel.guard.screens import screen_passages, screen_question

if TYPE_CHECKING:
    from regintel.answering.models import AnswerModel
    from regintel.answering.review import AnswerReviewer
    from regintel.answering.schema import CheckedStatement, DraftAnswer
    from regintel.corpus.manifest import Manifest
    from regintel.ingestion.chunking import Chunk
    from regintel.retrieval.base import Hit, Retriever


def is_tool_record(chunk_id: str) -> bool:
    """Tool results (faers:..., cdisc:..., openfda:...) versus corpus chunks (doc#0001)."""
    return ":" in chunk_id and "#" not in chunk_id


NOT_GROUNDED_MESSAGE = (
    "An answer was drafted but its citations could not be verified against the guidance text, so it is withheld."
)


def label_for(answerable: bool, checked: list[CheckedStatement]) -> Label:
    if not answerable and not checked:
        return "declined"
    verified = sum(s.verified for s in checked)
    if verified == 0:
        return "not_grounded" if checked else "declined"
    return "grounded" if verified == len(checked) else "partially_grounded"


class Answerer:
    def __init__(
        self,
        retriever: Retriever,
        model: AnswerModel,
        manifest: Manifest | None = None,
        top_k: int = 8,
        guard: bool = True,
        reviewer: AnswerReviewer | None = None,
    ) -> None:
        self.retriever = retriever
        self.model = model
        self.manifest = manifest
        self.top_k = top_k
        self.guard = guard  # False only to measure what the guards contribute
        self.reviewer = reviewer  # second model that drops statements stretching their quotes

    def answer(self, question: str, hits: list[Hit] | None = None) -> AnswerResult:
        """`hits` lets an evaluation reuse cached retrieval and vary only the answering step."""
        if self.guard and not (screen := screen_question(question)).allowed:
            return blocked(question, screen.reason)
        started = time.perf_counter()
        if hits is None:
            hits = self.retriever.search(question, top_k=self.top_k)
        hits = [h for h in hits if not h.chunk.doc_id.startswith("web:")][: self.top_k]  # web is never quotable
        retrieval_seconds = time.perf_counter() - started
        if not hits:
            return AnswerResult(
                question,
                "declined",
                [],
                "Nothing relevant was found in the indexed guidance.",
                [],
                Usage(),
                retrieval_seconds,
                0.0,
            )
        started = time.perf_counter()
        screened_out: list[str] = []
        if self.guard:
            _, dropped = screen_passages([h.chunk for h in hits])
            screened_out = [c.chunk_id for c in dropped]
            hits = [h for h in hits if h.chunk.chunk_id not in screened_out]
        draft, usage = self.model.draft(question, hits)
        result = verify_draft(question, draft, [h.chunk for h in hits], self.manifest, usage)
        if self.reviewer is not None:
            result = self.reviewer.review(result, [h.chunk.text for h in hits])
        result.screened_out = screened_out
        result.retrieval_seconds = retrieval_seconds
        result.generation_seconds = time.perf_counter() - started
        return result


def blocked(question: str, reason: str) -> AnswerResult:
    return AnswerResult(question, "blocked", [], f"This request was not processed. {reason}", [], Usage(), 0.0, 0.0)


def verify_draft(
    question: str, draft: DraftAnswer, passages: list[Chunk], manifest: Manifest | None, usage: Usage | None = None
) -> AnswerResult:
    """Check every statement of a draft against the passages it was written from, and label the answer."""
    checked = [check_statement(s, passages, manifest) for s in draft.statements]
    label = label_for(draft.answerable, checked)
    return AnswerResult(
        question=question,
        label=label,
        statements=checked,
        decline_reason=NOT_GROUNDED_MESSAGE if label == "not_grounded" else draft.decline_reason,
        retrieved=[c.chunk_id for c in passages],
        usage=usage or Usage(),
        retrieval_seconds=0.0,
        generation_seconds=0.0,
        draft=draft,
        passage_sha256={c.chunk_id: hashlib.sha256(c.text.encode("utf-8")).hexdigest() for c in passages},
        record_texts={c.chunk_id: c.text for c in passages if is_tool_record(c.chunk_id)},
    )

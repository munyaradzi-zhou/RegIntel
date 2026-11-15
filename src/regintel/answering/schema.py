"""What the model must return (structured output) and what the pipeline returns after verification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field


class DraftCitation(BaseModel):
    passage: int = Field(description="Number of the passage, as in [P3].")
    quote: str = Field(description="Exact words copied from that passage that support the statement.")


class DraftStatement(BaseModel):
    text: str = Field(description="One self-contained statement of the answer.")
    citations: list[DraftCitation]


class DraftAnswer(BaseModel):
    """Strict-mode friendly: every field required, no defaults."""

    answerable: bool = Field(description="False when the passages do not answer the question.")
    statements: list[DraftStatement]
    decline_reason: str | None = Field(description="Why the question cannot be answered, when answerable is false.")


Label = Literal["grounded", "partially_grounded", "not_grounded", "declined", "blocked"]


@dataclass(frozen=True)
class VerifiedCitation:
    chunk_id: str
    doc_id: str
    section: str
    pages: str
    source_label: str  # e.g. "EMA · CPMP/EWP/QWP/1401/98 Rev. 1 · §4.1.8, p.15"
    quote: str


@dataclass(frozen=True)
class CheckedStatement:
    text: str
    citations: list[VerifiedCitation]
    verified: bool
    problems: list[str] = field(default_factory=list)  # why it failed verification


@dataclass(frozen=True)
class WebPointer:
    """A regulator web page suggested when the indexed guidance cannot answer. Its summary is written
    by a model from the page, so it is never quoted or presented as verified."""

    title: str
    url: str
    summary: str


@dataclass
class Usage:
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    calls: int = 0

    def add(self, other: Usage) -> None:
        self.input_tokens += other.input_tokens
        self.cached_tokens += other.cached_tokens
        self.output_tokens += other.output_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.calls += other.calls


@dataclass
class AnswerResult:
    question: str
    label: Label
    statements: list[CheckedStatement]  # all of them; only verified ones are shown
    decline_reason: str | None
    retrieved: list[str]  # chunk ids passed to the model
    usage: Usage
    retrieval_seconds: float
    generation_seconds: float
    draft: DraftAnswer | None = None  # the model's raw output, kept so verification can be re-run offline
    screened_out: list[str] = field(default_factory=list)  # passages dropped by the injection screen
    review_usage: Usage | None = None  # tokens spent by the answer reviewer, when it ran
    web_pointers: list[WebPointer] = field(default_factory=list)  # only on declined answers

    @property
    def shown(self) -> list[CheckedStatement]:
        return [s for s in self.statements if s.verified]

    @property
    def text(self) -> str:
        """The answer as shown to the user, citations as [n] into `sources`."""
        if self.label in ("declined", "not_grounded", "blocked"):
            text = self.decline_reason or "The indexed guidance does not answer this question."
            if self.web_pointers:
                pages = "\n".join(f"- {p.title}: {p.url}" for p in self.web_pointers)
                text += f"\n\nPages on regulator websites that may help (web search, not verified):\n{pages}"
            return text
        numbers = {c.chunk_id: i for i, c in enumerate(self.sources, start=1)}
        lines = []
        for statement in self.shown:
            marks = "".join(f"[{numbers[c.chunk_id]}]" for c in statement.citations)
            lines.append(f"{statement.text} {marks}")
        return "\n".join(lines)

    @property
    def sources(self) -> list[VerifiedCitation]:
        seen: dict[str, VerifiedCitation] = {}
        for statement in self.shown:
            for citation in statement.citations:
                seen.setdefault(citation.chunk_id, citation)
        return list(seen.values())

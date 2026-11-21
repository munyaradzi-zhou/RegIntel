"""Request and response bodies. Every answer carries its label and sources, so a client can show
what was verified, what was dropped and what is only a web pointer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from regintel.guard.screens import MAX_QUESTION_CHARS
from regintel.prompts import prompt_tags

if TYPE_CHECKING:
    from regintel.conversation import ChatAnswer, Session

DISCLAIMER = "Information about regulatory guidance, not legal, medical or regulatory advice."


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    session_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")


class Source(BaseModel):
    number: int
    label: str
    quote: str
    doc_id: str
    section: str
    pages: str


class Statement(BaseModel):
    text: str
    sources: list[int]  # numbers into `sources`


class WebPointerOut(BaseModel):
    title: str
    url: str


class ToolCallOut(BaseModel):
    name: str
    arguments: dict[str, str]
    seconds: float


class AskResponse(BaseModel):
    session_id: str
    question: str
    standalone_question: str
    is_followup: bool
    label: str
    answer: str
    statements: list[Statement]
    sources: list[Source]
    dropped: int  # statements removed by verification or review
    web_pointers: list[WebPointerOut]
    path: str | None
    tool_calls: list[ToolCallOut]
    timings: dict[str, float]
    prompts: dict[str, str] = Field(default_factory=dict)  # "writer": "v5+<hash>": the prompts behind this answer
    disclaimer: str = DISCLAIMER

    @classmethod
    def from_chat(cls, chat: ChatAnswer) -> AskResponse:
        result = chat.answer.result
        # A declined, withheld or blocked answer shows only its reason: statements that survived
        # checking are not presented as an answer the label says does not exist.
        answered = result.label in ("grounded", "partially_grounded")
        shown = result.shown if answered else []
        cited = result.sources if answered else []
        numbers = {c.chunk_id: i for i, c in enumerate(cited, start=1)}
        return cls(
            session_id=chat.session_id,
            question=chat.question,
            standalone_question=chat.standalone,
            is_followup=chat.is_followup,
            label=result.label,
            answer=result.text,
            statements=[Statement(text=s.text, sources=[numbers[c.chunk_id] for c in s.citations]) for s in shown],
            sources=[
                Source(number=i, label=c.source_label, quote=c.quote, doc_id=c.doc_id, section=c.section, pages=c.pages)
                for i, c in enumerate(cited, start=1)
            ],
            dropped=sum(not s.verified for s in result.statements),
            web_pointers=[WebPointerOut(title=p.title, url=p.url) for p in result.web_pointers],
            path=chat.answer.path,
            tool_calls=[ToolCallOut(name=c.name, arguments=c.arguments, seconds=c.seconds) for c in chat.answer.calls],
            timings={
                "route": round(chat.answer.route_seconds, 2),
                "research": round(chat.answer.research_seconds, 2),
                "writing": round(result.generation_seconds, 2),
                "web": round(chat.answer.web_seconds, 2),
            },
            prompts=prompt_tags(),
        )


class TurnOut(BaseModel):
    question: str
    standalone_question: str
    label: str
    answer: str


class SessionOut(BaseModel):
    session_id: str
    turns: list[TurnOut]

    @classmethod
    def from_session(cls, session: Session) -> SessionOut:
        turns = [TurnOut(question=t.question, standalone_question=t.standalone, label=t.label, answer=t.answer)
                 for t in session.turns]  # fmt: skip
        return cls(session_id=session.id, turns=turns)

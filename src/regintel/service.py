"""RegIntel end to end: guard -> router -> fixed pipeline or research agent -> verified answer.

This is what the API (phase 5) calls. Every path ends in the same verifier and reviewer, and every
answer records which path produced it, how long each step took and what it cost in tokens.

When the indexed guidance cannot answer (declined or withheld), a web search restricted to
regulator domains suggests a few pages to check. They are pointers, never part of the answer: the
summaries are model-written, so nothing from them is quoted or verified.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from regintel.answering.pipeline import blocked
from regintel.answering.schema import WebPointer
from regintel.azure_search.knowledge import REGULATOR_DOMAINS
from regintel.guard.screens import screen_question

if TYPE_CHECKING:
    from regintel.agent.research import RegIntelAgent
    from regintel.agent.tools import ToolCall
    from regintel.answering.pipeline import Answerer
    from regintel.answering.schema import AnswerResult, Usage
    from regintel.retrieval.base import Retriever
    from regintel.router import Path, Routes

WEB_POINTERS = 3


@dataclass
class RoutedAnswer:
    result: AnswerResult
    path: Path | None  # None when the guard blocked the question
    route_reason: str = ""
    calls: list[ToolCall] = field(default_factory=list)
    research_note: str = ""
    research_seconds: float = 0.0
    research_usage: dict[str, Any] = field(default_factory=dict)
    route_seconds: float = 0.0
    route_usage: Usage | None = None
    web_seconds: float = 0.0


def on_regulator_domain(url: str, domains: tuple[str, ...] = REGULATOR_DOMAINS) -> bool:
    """The knowledge source is already restricted to these domains; checked again in code."""
    host = (urlparse(url).hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in domains)


def web_pointers(retriever: Retriever, question: str, limit: int = WEB_POINTERS) -> list[WebPointer]:
    pointers = []
    for hit in retriever.search(question, top_k=limit * 2):
        url = hit.chunk.doc_id.removeprefix("web:")
        if hit.chunk.doc_id.startswith("web:") and on_regulator_domain(url):
            pointers.append(WebPointer(hit.chunk.section_title or url, url, hit.chunk.text))
    return pointers[:limit]


class RegIntel:
    def __init__(
        self,
        router: Routes,
        single: Answerer,
        multi: Answerer,
        agent: RegIntelAgent,
        web: Retriever | None = None,
    ) -> None:
        self.router = router
        self.paths = {"guidance_single": single, "guidance_multi": multi}
        self.agent = agent
        self.web = web

    async def ask(self, question: str) -> RoutedAnswer:
        if not (screen := screen_question(question)).allowed:
            return RoutedAnswer(blocked(question, screen.reason), None, screen.category)
        answer = await self._answer(question)
        if self.web is not None and answer.result.label in ("declined", "not_grounded"):
            started = time.perf_counter()
            try:
                answer.result.web_pointers = await asyncio.to_thread(web_pointers, self.web, question)
            except Exception:
                answer.result.web_pointers = []
            answer.web_seconds = time.perf_counter() - started
        return answer

    async def _answer(self, question: str) -> RoutedAnswer:
        started = time.perf_counter()
        # Model and search calls below are blocking: they run in worker threads so the event loop
        # keeps serving other requests and the platform's health probes while a question is answered.
        decision, route_usage = await asyncio.to_thread(self.router.route, question)
        route_seconds = time.perf_counter() - started
        if decision.path == "agent":
            answer = await self.agent.answer(question)
            return RoutedAnswer(
                answer.result,
                decision.path,
                decision.reason,
                answer.calls,
                answer.research_note,
                answer.research_seconds,
                answer.research_usage,
                route_seconds,
                route_usage,
            )
        result = await asyncio.to_thread(self.paths[decision.path].answer, question)
        return RoutedAnswer(
            result,
            decision.path,
            decision.reason,
            research_seconds=result.retrieval_seconds,  # the fixed pipeline's "research" is one retrieval
            route_seconds=route_seconds,
            route_usage=route_usage,
        )

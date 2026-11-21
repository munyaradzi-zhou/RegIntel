"""Conversations: sessions, and follow-up questions rewritten into standalone ones.

A follow-up such as "and for narrow therapeutic index drugs?" means nothing to retrieval on its
own. The rewriter (one structured-output call at minimal reasoning) turns it into a standalone
question using the last few turns, and that standalone question goes through the whole pipeline:
guard, router, verifier, reviewer. Answers are never carried over from earlier turns, so every
answer is grounded in evidence retrieved for it.

Both the question as typed and the rewritten question are screened by the input guard, since the
rewrite is model output built from user text.
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel, Field

from regintel.answering.pipeline import blocked
from regintel.guard.screens import screen_question
from regintel.service import RoutedAnswer

if TYPE_CHECKING:
    from openai import OpenAI

    from regintel.service import RegIntel

HISTORY_TURNS = 4  # turns the rewriter sees; older context rarely changes what a follow-up means

REWRITE_PROMPT_VERSION = "v1"  # bump when REWRITE_PROMPT changes (checked in CI by regintel.prompts)

REWRITE_PROMPT = """\
Rewrite the user's latest message into one standalone question about regulatory guidance for \
medicines, using the user's earlier questions only to resolve what it refers to (pronouns, \
"and for ...", "what about the EMA?", "the same as ..."). Keep the user's wording and meaning; do \
not add facts, numbers, answers or assumptions. If the latest message is already standalone, \
return it unchanged.\
"""


class Rewrite(BaseModel):
    standalone_question: str
    is_followup: bool = Field(description="True if the message depended on earlier turns.")


@dataclass
class Turn:
    question: str  # as typed
    standalone: str  # what the pipeline answered
    answer: str
    label: str
    at: float = field(default_factory=time.time)


@dataclass
class Session:
    id: str
    turns: list[Turn] = field(default_factory=list)


class SessionStore(Protocol):
    def get(self, session_id: str) -> Session | None: ...
    def save(self, session: Session) -> None: ...


class InMemorySessionStore:
    """Process-local sessions for development and tests; Azure uses Cosmos DB (phase 8)."""

    def __init__(self, max_sessions: int = 1_000) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self.max_sessions = max_sessions

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            return self._sessions.get(session_id)

    def save(self, session: Session) -> None:
        with self._lock:
            self._sessions[session.id] = session
            while len(self._sessions) > self.max_sessions:  # evict the oldest
                self._sessions.pop(next(iter(self._sessions)))


SESSION_TTL_SECONDS = 7 * 24 * 3600


class CosmosSessionStore:
    """Sessions in Cosmos DB (serverless), so they survive restarts and are shared by replicas.
    One document per session, partitioned by its id, expiring 7 days after the last turn.
    Keyless: the container app's managed identity holds the Cosmos data-contributor role."""

    def __init__(self, container: Any) -> None:
        self.container = container  # azure.cosmos ContainerProxy

    @classmethod
    def connect(
        cls, endpoint: str, credential: Any, database: str = "regintel", container: str = "sessions"
    ) -> CosmosSessionStore:
        from azure.cosmos import CosmosClient

        return cls(
            CosmosClient(endpoint, credential=credential).get_database_client(database).get_container_client(container)
        )

    def get(self, session_id: str) -> Session | None:
        from azure.cosmos.exceptions import CosmosResourceNotFoundError

        try:
            item = self.container.read_item(item=session_id, partition_key=session_id)
        except CosmosResourceNotFoundError:
            return None
        return Session(item["id"], [Turn(**t) for t in item.get("turns", [])])

    def save(self, session: Session) -> None:
        self.container.upsert_item({
            "id": session.id,
            "turns": [vars(t) for t in session.turns[-50:]],  # bounded document size
            "ttl": SESSION_TTL_SECONDS,
        })  # fmt: skip


class QuestionRewriter:
    def __init__(self, client: OpenAI, model: str) -> None:
        self.client = client
        self.model = model

    def rewrite(self, turns: list[Turn], question: str) -> Rewrite:
        if not turns:
            return Rewrite(standalone_question=question, is_followup=False)
        # Earlier questions only, never answers: a rewrite built from answers smuggles their content
        # (and any error in them) into the next retrieval, and inflates follow-up scores.
        history = "\n".join(f"User: {t.standalone}" for t in turns[-HISTORY_TURNS:])
        response = self.client.responses.parse(
            model=self.model,
            instructions=REWRITE_PROMPT,
            input=f"Earlier turns:\n{history}\n\nLatest message: {question}",
            text_format=Rewrite,
            reasoning={"effort": "minimal"},
            store=False,
        )
        return response.output_parsed or Rewrite(standalone_question=question, is_followup=False)


@dataclass
class ChatAnswer:
    session_id: str
    question: str
    standalone: str
    is_followup: bool
    answer: RoutedAnswer


class RegIntelChat:
    def __init__(self, service: RegIntel, rewriter: QuestionRewriter, store: SessionStore) -> None:
        self.service = service
        self.rewriter = rewriter
        self.store = store

    def session(self, session_id: str | None) -> Session:
        existing = self.store.get(session_id) if session_id else None
        return existing or Session(session_id or uuid.uuid4().hex)

    async def ask(self, question: str, session_id: str | None = None) -> ChatAnswer:
        session = self.session(session_id)
        if not (screen := screen_question(question)).allowed:  # before any model sees the text
            answer = RoutedAnswer(blocked(question, screen.reason), None, screen.category)
            return ChatAnswer(session.id, question, question, False, answer)
        rewrite = await asyncio.to_thread(self.rewriter.rewrite, session.turns, question)
        answer = await self.service.ask(rewrite.standalone_question)  # screened again inside
        session.turns.append(Turn(question, rewrite.standalone_question, answer.result.text, answer.result.label))
        await asyncio.to_thread(self.store.save, session)
        return ChatAnswer(session.id, question, rewrite.standalone_question, rewrite.is_followup, answer)

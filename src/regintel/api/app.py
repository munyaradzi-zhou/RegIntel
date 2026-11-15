"""The RegIntel HTTP API.

    POST /v1/ask              one answer as JSON
    POST /v1/ask/stream       the same, as server-sent events: session, progress ticks, answer, done
    GET  /v1/sessions/{id}    the turns of a conversation
    GET  /v1/faers/profile    FAERS counts and per-reaction reporting odds ratios for a drug
    GET  /v1/cdisc            CDISC Controlled Terminology lookup
    GET  /healthz             process is up
    GET  /readyz              the answering service is built and can take questions

Clients send `X-API-Key` when keys are configured (API_KEYS); requests are rate limited per
client. Every response carries an `X-Request-ID`, also written to the structured log.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Annotated, Any

import structlog
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from regintel import data_tools
from regintel.api.schemas import AskRequest, AskResponse, SessionOut
from regintel.config import get_settings

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from starlette.responses import Response

    from regintel.conversation import RegIntelChat

log = structlog.get_logger("regintel.api")
PROGRESS_INTERVAL_SECONDS = 1.5


class RateLimiter:
    """Sliding one-minute window per client (API key, else address). In-process: one replica."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self.calls: dict[str, deque[float]] = defaultdict(deque)

    def retry_after(self, client: str, now: float | None = None) -> int:
        """0 if the call is allowed (and records it), else seconds until it would be."""
        now = time.monotonic() if now is None else now
        window = self.calls[client]
        while window and now - window[0] >= 60:
            window.popleft()
        if len(window) >= self.per_minute:
            return int(60 - (now - window[0])) + 1
        window.append(now)
        return 0


def build_chat() -> RegIntelChat:
    from regintel.agent.factory import AzureComponents
    from regintel.answering.models import azure_openai_client
    from regintel.conversation import InMemorySessionStore, QuestionRewriter, RegIntelChat, SessionStore

    settings = get_settings()
    components = AzureComponents.connect(settings)
    rewriter = QuestionRewriter(azure_openai_client(settings, components.tokens), settings.azure_openai_chat_deployment)
    store: SessionStore = InMemorySessionStore()
    if settings.cosmos_endpoint:
        from regintel.conversation import CosmosSessionStore
        from regintel.runtime import azure_credential

        store = CosmosSessionStore.connect(settings.cosmos_endpoint, azure_credential())
    return RegIntelChat(components.service(), rewriter, store)


# Dependencies live at module level: with postponed annotations FastAPI resolves `Depends(...)`
# from module globals, and keys and the limiter come from app.state.


def guarded(request: Request, x_api_key: Annotated[str | None, Header()] = None) -> str:
    """Authenticate (when keys are configured) and rate-limit; returns the client id."""
    keys: list[str] = request.app.state.keys
    if keys:
        if not x_api_key or not any(hmac.compare_digest(x_api_key, k) for k in keys):
            raise HTTPException(401, "A valid X-API-Key header is required.")
        client = f"key:{hashlib.sha256(x_api_key.encode()).hexdigest()[:12]}"
    else:
        client = f"ip:{request.client.host if request.client else 'unknown'}"
    wait = request.app.state.limiter.retry_after(client)
    if wait:
        raise HTTPException(429, f"Rate limit reached; retry in {wait} s.", headers={"Retry-After": str(wait)})
    return client


def get_chat(request: Request) -> Any:  # RegIntelChat; Any keeps FastAPI from inspecting it
    chat: RegIntelChat | None = getattr(request.app.state, "chat", None)
    if chat is None:
        raise HTTPException(503, "The answering service is starting.")
    return chat


def create_app(chat: RegIntelChat | None = None) -> FastAPI:
    """`chat` is injected in tests; otherwise the Azure-backed service is built at startup."""
    settings = get_settings()
    from regintel.runtime import configure_telemetry

    connection = settings.applicationinsights_connection_string
    configure_telemetry(connection.get_secret_value() if connection else None)
    keys = [
        k.strip() for k in (settings.api_keys.get_secret_value() if settings.api_keys else "").split(",") if k.strip()
    ]
    limiter = RateLimiter(settings.rate_limit_per_minute)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if not keys:
            log.warning("api_open", detail="API_KEYS is empty: the API accepts unauthenticated requests")
        app.state.start_error = None
        if chat is not None:
            app.state.chat = chat
        else:
            # Stay up if the service cannot be built (no credentials, no data yet): /healthz keeps
            # answering, /readyz reports why, and the platform does not restart-loop the container.
            try:
                app.state.chat = await asyncio.to_thread(build_chat)
            except Exception as error:
                log.exception("service_unavailable")
                app.state.chat, app.state.start_error = None, f"{type(error).__name__}: {error}"[:300]
        yield

    app = FastAPI(title="RegIntel", version="0.1.0", lifespan=lifespan)
    app.state.keys = keys
    app.state.limiter = limiter

    @app.middleware("http")
    async def request_context(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        structlog.contextvars.bind_contextvars(request_id=request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.clear_contextvars()
        response.headers["X-Request-ID"] = request_id
        log.info("request", method=request.method, path=request.url.path, status=response.status_code,
                 ms=round((time.perf_counter() - started) * 1000), request_id=request_id)  # fmt: skip
        return response

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(request: Request) -> JSONResponse:
        ready = getattr(request.app.state, "chat", None) is not None
        body: dict[str, object] = {"ready": ready}
        if not ready and getattr(request.app.state, "start_error", None):
            body["reason"] = request.app.state.start_error
        return JSONResponse(body, status_code=200 if ready else 503)

    @app.post("/v1/ask")
    async def ask(
        body: AskRequest, _: Annotated[str, Depends(guarded)], chat_service: Annotated[Any, Depends(get_chat)]
    ) -> AskResponse:
        answer = await chat_service.ask(body.question, body.session_id)
        log.info("answer", label=answer.answer.result.label, path=answer.answer.path)
        return AskResponse.from_chat(answer)

    @app.post("/v1/ask/stream")
    async def ask_stream(
        body: AskRequest, _: Annotated[str, Depends(guarded)], chat_service: Annotated[Any, Depends(get_chat)]
    ) -> EventSourceResponse:
        session_id = chat_service.session(body.session_id).id

        async def events() -> AsyncIterator[dict[str, Any]]:
            yield {"event": "session", "data": json.dumps({"session_id": session_id})}
            task = asyncio.create_task(chat_service.ask(body.question, session_id))
            started = time.perf_counter()
            while not task.done():
                await asyncio.wait({task}, timeout=PROGRESS_INTERVAL_SECONDS)
                if not task.done():
                    yield {
                        "event": "progress",
                        "data": json.dumps({"elapsed": round(time.perf_counter() - started, 1)}),
                    }
            try:
                answer = task.result()
            except Exception:
                log.exception("stream_failed")
                yield {"event": "error", "data": json.dumps({"detail": "The answer could not be produced."})}
                return
            yield {"event": "answer", "data": AskResponse.from_chat(answer).model_dump_json()}
            yield {"event": "done", "data": "{}"}

        return EventSourceResponse(events())

    @app.get("/v1/faers/profile")
    async def faers_profile(
        drug: Annotated[str, Query(min_length=2, max_length=200)], _: Annotated[str, Depends(guarded)]
    ) -> dict[str, Any]:
        """Counts and the most frequent reactions for a drug, each with a reporting odds ratio (95% CI)."""
        return await asyncio.to_thread(data_tools.faers_profile, data_tools.faers_store(), drug)

    @app.get("/v1/cdisc")
    async def cdisc(
        term: Annotated[str, Query(min_length=1, max_length=200)], _: Annotated[str, Depends(guarded)]
    ) -> list[dict[str, Any]]:
        """Exact lookup in CDISC Controlled Terminology."""
        return data_tools.cdisc_lookup(data_tools.terminology(), term)

    @app.get("/v1/sessions/{session_id}")
    async def get_session(
        session_id: str, _: Annotated[str, Depends(guarded)], chat_service: Annotated[Any, Depends(get_chat)]
    ) -> SessionOut:
        session = chat_service.store.get(session_id)
        if session is None:
            raise HTTPException(404, "No such session.")
        return SessionOut.from_session(session)

    return app

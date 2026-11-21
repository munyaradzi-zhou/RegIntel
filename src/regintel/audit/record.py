"""One audit record per answer: enough to attribute it, reproduce it and check it was not changed.

Attributable: the caller is the API key's id (a hash prefix, never the key), with the request id
and session. Reproducible: the question as typed and as answered, the route, what retrieval was
asked, every passage the writer saw (id and sha256; tool records with their full text, since they
cannot be looked up again), the models, the prompt versions, the model's raw draft and every
verifier and reviewer decision. Replaying the draft through the verifier must give the same
decisions (scripts/replay.py). Accountable cost: latency and tokens.

Answers that never reached a model (blocked by the guard) are recorded too.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from regintel.prompts import prompt_tags

if TYPE_CHECKING:
    from regintel.conversation import ChatAnswer

RECORD_VERSION = 1  # bump when the record's fields change meaning


def build_record(
    chat: ChatAnswer,
    *,
    caller: str,
    request_id: str,
    models: dict[str, str | None],
    latency_seconds: float,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    routed, result = chat.answer, chat.answer.result
    if routed.path == "agent":
        queries: list[dict[str, Any]] = [{"tool": c.name, "arguments": c.arguments} for c in routed.calls]
    elif routed.path is not None:
        queries = [{"tool": f"knowledge_base:{routed.path}", "arguments": {"query": chat.standalone}}]
    else:
        queries = []
    return {
        "id": uuid.uuid4().hex,
        "type": "answer",
        "record_version": RECORD_VERSION,
        "day": now.strftime("%Y-%m-%d"),
        "at": now.isoformat(timespec="milliseconds"),
        "caller": caller,
        "request_id": request_id,
        "session_id": chat.session_id,
        "question": chat.question,
        "standalone_question": chat.standalone,
        "route": {"path": routed.path, "reason": routed.route_reason},
        "retrieval": queries,
        "sources": [{"chunk_id": c, "sha256": result.passage_sha256.get(c, "")} for c in result.retrieved],
        "tool_records": result.record_texts,
        "screened_out": result.screened_out,
        "models": models,
        "prompts": prompt_tags(),
        "draft": result.draft.model_dump() if result.draft is not None else None,
        "statements": [
            {
                "text": s.text,
                "verified": s.verified,
                "problems": s.problems,
                "citations": [{"chunk_id": c.chunk_id, "quote": c.quote} for c in s.citations],
            }
            for s in result.statements
        ],
        "label": result.label,
        "decline_reason": result.decline_reason,
        "latency_seconds": round(latency_seconds, 3),
        "timings": {
            "route": round(routed.route_seconds, 3),
            "research": round(routed.research_seconds, 3),
            "writing": round(result.generation_seconds, 3),
            "web": round(routed.web_seconds, 3),
        },
        "usage": {
            "writer": asdict(result.usage),
            "reviewer": asdict(result.review_usage) if result.review_usage else None,
            "router": asdict(routed.route_usage) if routed.route_usage else None,
            "research": {k: v for k, v in routed.research_usage.items() if isinstance(v, int | float)},
        },
    }

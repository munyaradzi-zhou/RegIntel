"""HTTP client for the RegIntel API (the front end never calls models or Azure directly)."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from collections.abc import Iterator

API_URL = os.environ.get("REGINTEL_API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = httpx.Timeout(180.0, connect=5.0)


def _headers() -> dict[str, str]:
    key = os.environ.get("REGINTEL_API_KEY")
    return {"X-API-Key": key} if key else {}


def ready() -> bool:
    try:
        return bool(httpx.get(f"{API_URL}/readyz", timeout=3.0).json().get("ready"))
    except (httpx.HTTPError, ValueError):
        return False


def parse_sse(lines: Iterator[str]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Server-sent events as (event, data) pairs; an event ends at a blank line."""
    event, data = "message", list[str]()
    for line in lines:
        if not line.strip():
            if data:
                yield event, json.loads("\n".join(data))
            event, data = "message", []
        elif line.startswith("event:"):
            event = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            data.append(line.removeprefix("data:").strip())
    if data:
        yield event, json.loads("\n".join(data))


def ask_stream(question: str, session_id: str | None) -> Iterator[tuple[str, dict[str, Any]]]:
    body = {"question": question, "session_id": session_id}
    with httpx.stream("POST", f"{API_URL}/v1/ask/stream", json=body, headers=_headers(), timeout=TIMEOUT) as response:
        response.raise_for_status()
        yield from parse_sse(response.iter_lines())


def faers_profile(drug: str) -> dict[str, Any]:
    response = httpx.get(f"{API_URL}/v1/faers/profile", params={"drug": drug}, headers=_headers(), timeout=TIMEOUT)
    response.raise_for_status()
    result: dict[str, Any] = response.json()
    return result


def cdisc(term: str) -> list[dict[str, Any]]:
    response = httpx.get(f"{API_URL}/v1/cdisc", params={"term": term}, headers=_headers(), timeout=TIMEOUT)
    response.raise_for_status()
    result: list[dict[str, Any]] = response.json()
    return result

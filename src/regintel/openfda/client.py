"""openFDA over HTTPS: timeouts, retries, a TTL cache, and a typed failure.

The label and enforcement endpoints are called at answer time, so this module handles the parts a
static corpus never meets:

- timeouts: 5 s to connect, 10 s per read, so a slow API cannot hold a question for long;
- retries: 429 and 5xx, and transport errors (timeouts, resets), with exponential backoff and
  jitter, 3 attempts; a Retry-After of a few seconds is respected;
- "No matches found!" is a 404 with a JSON body: an answer (nothing matched), not a failure;
- cache: results (including "no match") are kept for a day, keyed by endpoint and query, so
  repeated questions do not spend the rate limit (240 requests a minute, 1,000 a day per IP
  without a key; an API key raises the daily limit);
- anything else raises `OpenFDAUnavailableError`. The tools turn that into "openFDA could not be
  reached", and nothing is added to the evidence, so the writer cannot cite a label it never saw.
"""

from __future__ import annotations

import ssl
import threading
import time
from dataclasses import dataclass, field
from functools import cache
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

BASE_URL = "https://api.fda.gov"
TIMEOUT = httpx.Timeout(10.0, connect=5.0)
ATTEMPTS = 3
CACHE_SECONDS = 24 * 3600
CACHE_ENTRIES = 512
MAX_RETRY_AFTER = 5.0  # seconds; a longer wait is not worth holding a question for


class OpenFDAUnavailableError(Exception):
    """openFDA did not answer (timeouts, 429 or 5xx after retries, an unexpected status or body)."""


class _RetryableError(Exception):
    def __init__(self, status: int, retry_after: float = 0.0) -> None:
        super().__init__(f"HTTP {status}")
        self.status = status
        self.retry_after = retry_after


def _retryable(exc: BaseException) -> bool:
    return isinstance(exc, _RetryableError | httpx.TransportError)


@dataclass(frozen=True)
class Page:
    results: list[dict[str, Any]]
    total: int  # matches in openFDA (results holds at most `limit` of them)


@dataclass
class TTLCache:
    seconds: float = CACHE_SECONDS
    entries: int = CACHE_ENTRIES
    _items: dict[str, tuple[float, Page]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def get(self, key: str) -> Page | None:
        with self._lock:
            hit = self._items.get(key)
            if hit is None or hit[0] < time.monotonic():
                self._items.pop(key, None)
                return None
            return hit[1]

    def put(self, key: str, value: Page) -> None:
        with self._lock:
            self._items[key] = (time.monotonic() + self.seconds, value)
            while len(self._items) > self.entries:  # evict the oldest
                self._items.pop(next(iter(self._items)))


class OpenFDAClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = BASE_URL,
        transport: httpx.AsyncBaseTransport | None = None,
        cache: TTLCache | None = None,
        attempts: int = ATTEMPTS,
        max_wait: float = 4.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.transport = transport
        self.cache = cache if cache is not None else TTLCache()
        self.attempts = attempts
        self.max_wait = max_wait
        self.requests = 0  # HTTP requests sent (cache hits send none)

    async def search(self, endpoint: str, search: str, *, limit: int, sort: str | None = None) -> Page:
        """Results for an openFDA query (none when nothing matches)."""
        params: dict[str, str | int] = {"search": search, "limit": limit}
        if sort:
            params["sort"] = sort
        key = f"{endpoint}?{sorted(params.items())}"  # the API key is not part of the cache key
        if (cached := self.cache.get(key)) is not None:
            return cached
        if self.api_key:
            params["api_key"] = self.api_key
        try:
            page = await self._get(endpoint, params)
        except (_RetryableError, httpx.TransportError) as exc:
            raise OpenFDAUnavailableError(f"openFDA did not respond after {self.attempts} attempts ({exc})") from exc
        self.cache.put(key, page)
        return page

    async def _get(self, endpoint: str, params: dict[str, str | int]) -> Page:
        # A client per lookup, not per process: callers may run each question in its own event loop
        # (the evaluations do), and a pooled AsyncClient is bound to the loop that created it.
        async with httpx.AsyncClient(
            base_url=self.base_url, timeout=TIMEOUT, transport=self.transport, verify=_tls()
        ) as client:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception(_retryable),
                stop=stop_after_attempt(self.attempts),
                wait=_wait(self.max_wait),
                reraise=True,
            ):
                with attempt:
                    self.requests += 1
                    response = await client.get(endpoint, params=params)
                    return _page(response)
        raise AssertionError("unreachable")  # pragma: no cover


@cache
def _tls() -> ssl.SSLContext:
    """Built once: loading the certificate store is the slow part of opening a client."""
    return ssl.create_default_context()


def _wait(max_wait: float) -> Any:
    backoff = wait_exponential_jitter(multiplier=0.5, max=max_wait, jitter=min(1.0, max_wait))

    def wait(state: Any) -> float:
        exc = state.outcome.exception() if state.outcome else None
        if isinstance(exc, _RetryableError) and exc.retry_after:
            return min(exc.retry_after, MAX_RETRY_AFTER)
        return float(backoff(state))

    return wait


def _page(response: httpx.Response) -> Page:
    if response.status_code == 429 or response.status_code >= 500:
        raise _RetryableError(response.status_code, _retry_after(response))
    if response.status_code == 404 and _error_code(response) == "NOT_FOUND":
        return Page([], 0)  # "No matches found!"
    if response.status_code != 200:
        raise OpenFDAUnavailableError(f"openFDA returned HTTP {response.status_code}")
    try:
        body = response.json()
        results, total = body["results"], int(body["meta"]["results"]["total"])
    except (ValueError, KeyError, TypeError) as exc:
        raise OpenFDAUnavailableError("openFDA returned an unexpected body") from exc
    if not isinstance(results, list):
        raise OpenFDAUnavailableError("openFDA returned an unexpected body")
    return Page(results, total)


def _error_code(response: httpx.Response) -> str:
    try:
        return str(response.json()["error"]["code"])
    except (ValueError, KeyError, TypeError):
        return ""


def _retry_after(response: httpx.Response) -> float:
    try:
        return max(0.0, float(response.headers.get("Retry-After", "0")))
    except ValueError:
        return 0.0

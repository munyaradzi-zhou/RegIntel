"""Download the corpus listed in the manifest, verifying what can be verified.

Guidance PDFs are pinned by sha256: a mismatch means the publisher changed the document,
and the download is rejected rather than silently indexed. CDISC terminology and FAERS
are re-issued by their publishers, so their checksums are recorded in a LOCK.json instead.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import httpx
import structlog

if TYPE_CHECKING:
    from pathlib import Path

    from regintel.corpus.manifest import Manifest

log = structlog.get_logger()

USER_AGENT = "RegIntel corpus fetcher (research; https://github.com/munyaradzi-zhou)"


class BlockedDownloadError(Exception):
    """The publisher's CDN refused an automated download (fda.gov does this at times)."""


class ChecksumMismatchError(Exception):
    """A pinned document no longer matches the manifest."""


@dataclass(frozen=True)
class FetchResult:
    downloaded: int
    skipped: int


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _get(client: httpx.Client, url: str) -> bytes:
    response = client.get(url)
    response.raise_for_status()
    return response.content


def fetch_guidance(manifest: Manifest, dest: Path, client: httpx.Client) -> FetchResult:
    """Download each pinned PDF. Files already in `dest` with the right checksum are kept, so a
    document downloaded by hand in a browser (when a CDN blocks scripts) is accepted as is."""
    dest.mkdir(parents=True, exist_ok=True)
    downloaded = skipped = 0
    blocked: list[str] = []
    for doc in manifest.guidance:
        path = dest / f"{doc.id}.pdf"
        if path.exists() and sha256_hex(path.read_bytes()) == doc.sha256:
            skipped += 1
            continue
        try:
            body = _get(client, doc.url)
        except httpx.HTTPStatusError:
            blocked.append(f"  {doc.url}\n    -> save as {path}")
            continue
        if not body.startswith(b"%PDF"):
            blocked.append(f"  {doc.url}\n    -> save as {path}")
            continue
        actual = sha256_hex(body)
        if actual != doc.sha256:
            raise ChecksumMismatchError(
                f"{doc.id}: sha256 {actual[:12]}… does not match the manifest ({doc.sha256[:12]}…). "
                "The publisher may have revised the document: review it, then update data/manifest.json."
            )
        path.write_bytes(body)
        downloaded += 1
        log.info("fetched", doc=doc.id, kb=len(body) // 1024)
    if blocked:
        raise BlockedDownloadError(
            f"{len(blocked)} document(s) could not be downloaded automatically. Download them in a browser, "
            "save them under the names below, then run this again (checksums are still verified):\n"
            + "\n".join(blocked)
        )
    return FetchResult(downloaded, skipped)


def _write_lock(dest: Path, entries: dict[str, dict[str, str | int]]) -> None:
    lock = {"fetched_at": datetime.now(UTC).isoformat(timespec="seconds"), "files": entries}
    (dest / "LOCK.json").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")


def fetch_cdisc(manifest: Manifest, dest: Path, client: httpx.Client) -> FetchResult:
    dest.mkdir(parents=True, exist_ok=True)
    entries: dict[str, dict[str, str | int]] = {}
    for name, url in manifest.cdisc.files.items():
        body = _get(client, url)
        if not body.startswith(b"Code\t"):
            raise ValueError(f"CDISC {name}: {url} is not a tab-delimited terminology file")
        (dest / f"{name}.tsv").write_bytes(body)
        entries[name] = {"url": url, "sha256": sha256_hex(body), "bytes": len(body)}
        log.info("fetched", cdisc=name, kb=len(body) // 1024)
    _write_lock(dest, entries)
    return FetchResult(len(entries), 0)


def faers_part_urls(manifest: Manifest, index: dict[str, object]) -> list[str]:
    """Resolve the manifest's quarter and part numbers to download URLs from openFDA's index."""
    quarter = manifest.faers.quarter
    partitions = index["results"]["drug"]["event"]["partitions"]  # type: ignore[index]
    in_quarter = sorted(p["file"] for p in partitions if f"/{quarter}/" in p["file"])
    if not in_quarter:
        raise ValueError(f"openFDA lists no FAERS partitions for {quarter}")
    wanted = {f"drug-event-{n:04d}-of-" for n in manifest.faers.parts}
    urls = [u for u in in_quarter if any(w in u for w in wanted)]
    if len(urls) != len(manifest.faers.parts):
        raise ValueError(f"expected parts {manifest.faers.parts} of {quarter}, found {len(urls)}")
    return urls


def fetch_faers(manifest: Manifest, dest: Path, client: httpx.Client) -> FetchResult:
    dest.mkdir(parents=True, exist_ok=True)
    index = json.loads(_get(client, manifest.faers.index))
    entries: dict[str, dict[str, str | int]] = {}
    skipped = 0
    for url in faers_part_urls(manifest, index):
        name = url.rsplit("/", 1)[-1].removesuffix(".zip")
        path = dest / name
        if path.exists():
            skipped += 1
            entries[name] = {"url": url, "sha256": sha256_hex(path.read_bytes())}
            continue
        archive = _get(client, url)
        with zipfile.ZipFile(io.BytesIO(archive)) as zf:
            body = zf.read(name)
        path.write_bytes(body)
        entries[name] = {"url": url, "sha256": sha256_hex(body)}
        log.info("fetched", faers=name, mb=len(body) // 2**20)
    _write_lock(dest, entries)
    return FetchResult(len(entries) - skipped, skipped)


def make_client() -> httpx.Client:
    return httpx.Client(follow_redirects=True, timeout=300, headers={"User-Agent": USER_AGENT})

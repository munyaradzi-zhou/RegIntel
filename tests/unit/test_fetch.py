import hashlib
import io
import json
import zipfile
from pathlib import Path

import httpx
import pytest

from regintel.corpus.fetch import (
    BlockedDownloadError,
    ChecksumMismatchError,
    faers_part_urls,
    fetch_cdisc,
    fetch_faers,
    fetch_guidance,
)
from regintel.corpus.manifest import Manifest

PDF = b"%PDF-1.7 test document"


def _manifest(sha: str = hashlib.sha256(PDF).hexdigest()) -> Manifest:
    return Manifest.model_validate(
        {
            "description": "test",
            "verified": "2026-11-13",
            "guidance": [
                {
                    "id": "doc-a",
                    "issuer": "ICH",
                    "jurisdiction": "ICH",
                    "title": "A",
                    "reference": "E6(R3)",
                    "date": "2025-01-06",
                    "status": "final",
                    "topics": ["gcp"],
                    "url": "https://x.test/a.pdf",
                    "sha256": sha,
                    "pages": 1,
                }
            ],
            "cdisc": {"source": "s", "note": "n", "files": {"sdtm": "https://x.test/sdtm.txt"}},
            "faers": {
                "source": "s",
                "index": "https://x.test/download.json",
                "quarter": "2025q1",
                "parts": [1],
                "note": "n",
            },
        }
    )


def _client(routes: dict[str, httpx.Response]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return routes.get(str(request.url), httpx.Response(404))

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_guidance_downloads_and_verifies(tmp_path: Path) -> None:
    result = fetch_guidance(_manifest(), tmp_path, _client({"https://x.test/a.pdf": httpx.Response(200, content=PDF)}))
    assert result.downloaded == 1
    assert (tmp_path / "doc-a.pdf").read_bytes() == PDF


def test_guidance_already_present_is_kept(tmp_path: Path) -> None:
    (tmp_path / "doc-a.pdf").write_bytes(PDF)
    result = fetch_guidance(_manifest(), tmp_path, _client({}))
    assert (result.downloaded, result.skipped) == (0, 1)


def test_changed_document_is_rejected(tmp_path: Path) -> None:
    client = _client({"https://x.test/a.pdf": httpx.Response(200, content=b"%PDF-1.7 revised")})
    with pytest.raises(ChecksumMismatchError, match="does not match the manifest"):
        fetch_guidance(_manifest(), tmp_path, client)
    assert not (tmp_path / "doc-a.pdf").exists()


def test_blocked_download_explains_manual_route(tmp_path: Path) -> None:
    client = _client({"https://x.test/a.pdf": httpx.Response(200, content=b"<html>blocked</html>")})
    with pytest.raises(BlockedDownloadError, match=r"save as .*doc-a\.pdf"):
        fetch_guidance(_manifest(), tmp_path, client)


def test_cdisc_writes_lock(tmp_path: Path) -> None:
    body = b"Code\tCodelist Code\nC1\t\n"
    fetch_cdisc(_manifest(), tmp_path, _client({"https://x.test/sdtm.txt": httpx.Response(200, content=body)}))
    lock = json.loads((tmp_path / "LOCK.json").read_text())
    assert lock["files"]["sdtm"]["sha256"] == hashlib.sha256(body).hexdigest()


def test_cdisc_rejects_non_terminology(tmp_path: Path) -> None:
    client = _client({"https://x.test/sdtm.txt": httpx.Response(200, content=b"<html>")})
    with pytest.raises(ValueError, match="not a tab-delimited"):
        fetch_cdisc(_manifest(), tmp_path, client)


def _index(*files: str) -> dict[str, object]:
    return {"results": {"drug": {"event": {"partitions": [{"file": f} for f in files]}}}}


def test_faers_parts_resolve_from_index() -> None:
    base = "https://dl.test/drug/event"
    index = _index(f"{base}/2025q1/drug-event-0001-of-0028.json.zip", f"{base}/2025q1/drug-event-0002-of-0028.json.zip",
                   f"{base}/2024q4/drug-event-0001-of-0030.json.zip")  # fmt: skip
    assert faers_part_urls(_manifest(), index) == [f"{base}/2025q1/drug-event-0001-of-0028.json.zip"]


def test_faers_missing_quarter_fails() -> None:
    with pytest.raises(ValueError, match="no FAERS partitions"):
        faers_part_urls(_manifest(), _index("https://dl.test/drug/event/2024q4/drug-event-0001-of-0030.json.zip"))


def test_faers_download_unzips(tmp_path: Path) -> None:
    url = "https://dl.test/drug/event/2025q1/drug-event-0001-of-0028.json.zip"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("drug-event-0001-of-0028.json", '{"results": []}')
    client = _client(
        {
            "https://x.test/download.json": httpx.Response(200, json=_index(url)),
            url: httpx.Response(200, content=buffer.getvalue()),
        }
    )
    result = fetch_faers(_manifest(), tmp_path, client)
    assert result.downloaded == 1
    assert (tmp_path / "drug-event-0001-of-0028.json").read_text() == '{"results": []}'

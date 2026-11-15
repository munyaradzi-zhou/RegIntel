"""Azure AI Search object definitions and the index upload/cleanup logic, with a fake search client."""

import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from regintel import runtime
from regintel.azure_search.knowledge import (
    AzureNames,
    knowledge_base,
    storage_resource_id,
    web_knowledge_source,
)
from regintel.azure_search.sections_index import (
    build_index,
    delete_stale,
    sections_knowledge_source,
    to_document,
    upload_chunks,
)
from regintel.config import Settings
from regintel.ingestion.chunking import Chunk
from regintel.ingestion.pipeline import read_chunks, write_chunks

SETTINGS = Settings(_env_file=None, azure_search_endpoint="https://s", azure_openai_endpoint="https://o/")  # type: ignore[call-arg]
NAMES = AzureNames("regintel")
CHUNKS = [
    Chunk(f"ema-be#{i:04d}", "ema-be", "EU", "4.1.8", "Evaluation", 15, 16, f"text {i}", "[EMA]") for i in range(5)
]


def test_names_and_resource_id() -> None:
    assert (NAMES.sections_index, NAMES.web_ks, NAMES.knowledge_base("web")) == (
        "regintel-sections", "regintel-web-ks", "regintel-kb-web",
    )  # fmt: skip
    assert storage_resource_id("sub", "rg", "acct") == (
        "ResourceId=/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Storage/storageAccounts/acct;"
    )


def test_index_keeps_locators_filterable_and_vectors_3072_wide() -> None:
    index = build_index(SETTINGS, NAMES)
    fields = {f.name: f for f in index.fields}
    assert fields["id"].key and fields["doc_id"].filterable and fields["section"].filterable
    assert fields["content_vector"].vector_search_dimensions == 3072
    assert sections_knowledge_source(NAMES).name == "regintel-sections-ks"


def test_web_source_is_limited_to_regulator_domains_and_adds_a_routing_instruction() -> None:
    web = web_knowledge_source(NAMES)
    domains = [d.address for d in web.web_parameters.domains.allowed_domains]
    assert "fda.gov" in domains and "ema.europa.eu" in domains and all("." in d for d in domains)
    combined = knowledge_base(SETTINGS, NAMES, "sections-web", [NAMES.sections_ks, NAMES.web_ks])
    assert "regintel-web-ks" in combined.retrieval_instructions
    assert (
        "regintel-web-ks" not in knowledge_base(SETTINGS, NAMES, "sections", [NAMES.sections_ks]).retrieval_instructions
    )


class FakeSearch:
    def __init__(self, existing: list[str], fail: str | None = None) -> None:
        self.existing, self.fail = existing, fail
        self.uploaded: list[dict[str, Any]] = []
        self.deleted: list[dict[str, Any]] = []

    def upload_documents(self, docs: list[dict[str, Any]]) -> list[Any]:
        self.uploaded += docs
        return [SimpleNamespace(key=d["id"], succeeded=d["id"] != self.fail) for d in docs]

    def search(self, **_: Any) -> list[dict[str, str]]:
        return [{"id": key} for key in self.existing]

    def delete_documents(self, docs: list[dict[str, Any]]) -> None:
        self.deleted += docs


def test_upload_in_batches_with_index_safe_keys() -> None:
    search = FakeSearch([])
    batches: list[int] = []

    def embed(texts: list[str]) -> list[list[float]]:
        batches.append(len(texts))
        return [[0.0] * 3 for _ in texts]

    assert upload_chunks(search, CHUNKS, embed, batch_size=2) == 5  # type: ignore[arg-type]
    assert batches == [2, 2, 1] and search.uploaded[0]["id"] == "ema-be_0000"
    assert to_document(CHUNKS[0], [1.0])["content"] == "text 0"


def test_failed_upload_raises() -> None:
    with pytest.raises(RuntimeError, match="ema-be_0001"):
        upload_chunks(FakeSearch([], fail="ema-be_0001"), CHUNKS, lambda t: [[0.0]] * len(t))  # type: ignore[arg-type]


def test_stale_documents_are_removed_after_a_rechunk() -> None:
    search = FakeSearch(["ema-be_0000", "ema-be_0099", "old_0001"])
    assert delete_stale(search, CHUNKS) == 2  # type: ignore[arg-type]
    assert search.deleted == [{"id": "ema-be_0099"}, {"id": "old_0001"}]


def test_chunks_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    write_chunks(CHUNKS, path)
    assert read_chunks(path) == CHUNKS


def test_stdio_logging_goes_to_stderr() -> None:
    runtime.log_to_stderr()
    handlers = logging.getLogger().handlers
    assert handlers and all(getattr(h, "stream", None) is sys.stderr for h in handlers)
    runtime.use_os_certificates()  # no-op or injection; must not raise either way

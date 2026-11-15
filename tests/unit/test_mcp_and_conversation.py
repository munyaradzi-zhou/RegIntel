import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest

from regintel import mcp_server
from regintel.conversation import QuestionRewriter, Rewrite, Turn
from regintel.faers.query import Disproportionality, DrugSummary
from regintel.terminology.cdisc import Term, Terminology

SEVERITY = Term(
    "sdtm", "C66769", "", "Severity/Intensity Scale for Adverse Events", "AESEV", (), "A scale.", "Severity"
)
MILD = Term("sdtm", "C41338", "C66769", "Severity/Intensity Scale for Adverse Events", "MILD", (), "Mild.", "Mild")


class FakeFaers:
    def drug_summary(self, drug: str, top: int = 5) -> DrugSummary:
        return DrugSummary(drug, ["DRUGA"], 100, 10, 4, 1, [("Rash", 3)])

    def disproportionality(self, drug: str, reaction: str) -> Disproportionality:
        return Disproportionality(drug, reaction, a=20, b=80, c=100, d=9800)


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(mcp_server, "_faers", FakeFaers)
    monkeypatch.setattr(mcp_server, "_terminology", lambda: Terminology([SEVERITY, MILD]))
    return mcp_server.build_server()


def _payload(result: Any) -> Any:
    return json.loads(result.content[0].text)


def test_tools_are_listed_read_only(server: Any) -> None:
    tools = asyncio.run(server.list_tools())
    assert [t.name for t in tools] == [
        "ask_regulatory_question", "lookup_cdisc_term", "faers_drug_summary", "faers_disproportionality",
    ]  # fmt: skip
    assert all(t.annotations.read_only_hint for t in tools)


def test_cdisc_lookup_returns_codelist_members(server: Any) -> None:
    body = _payload(asyncio.run(server.call_tool("lookup_cdisc_term", {"term": "AESEV"})))
    assert body["code"] == "C66769" and body["terms"] == [{"submission_value": "MILD", "code": "C41338"}]


def test_faers_tools_return_numbers_and_their_text(server: Any) -> None:
    summary = _payload(asyncio.run(server.call_tool("faers_drug_summary", {"drug": "druga"})))
    assert (summary["reports"], summary["serious"], summary["deaths"]) == (10, 4, 1)
    ror = _payload(asyncio.run(server.call_tool("faers_disproportionality", {"drug": "x", "reaction": "y"})))
    assert ror["ror"] == round((20 / 80) / (100 / 9800), 2) and len(ror["ci95"]) == 2
    assert "does not show that the drug causes" in ror["text"]


class FakeResponses:
    def __init__(self) -> None:
        self.inputs: list[str] = []

    def parse(self, **kwargs: Any) -> Any:
        self.inputs.append(kwargs["input"])
        return SimpleNamespace(output_parsed=Rewrite(standalone_question="rewritten", is_followup=True))


def test_rewriter_skips_the_model_without_history_and_never_sends_answers() -> None:
    responses = FakeResponses()
    rewriter = QuestionRewriter(SimpleNamespace(responses=responses), "m")  # type: ignore[arg-type]
    assert rewriter.rewrite([], "first question").standalone_question == "first question"
    assert responses.inputs == []  # no call for a first question
    turns = [Turn("q1", "What is the EU BE range?", "SECRET ANSWER TEXT 80.00-125.00%", "grounded")]
    assert rewriter.rewrite(turns, "and for NTI drugs?").standalone_question == "rewritten"
    assert "What is the EU BE range?" in responses.inputs[0] and "SECRET ANSWER" not in responses.inputs[0]


class FakeContainer:
    def __init__(self) -> None:
        self.items: dict[str, dict[str, Any]] = {}

    def read_item(self, item: str, partition_key: str) -> dict[str, Any]:
        from azure.cosmos.exceptions import CosmosResourceNotFoundError

        if item not in self.items:
            raise CosmosResourceNotFoundError(message="missing")
        return self.items[item]

    def upsert_item(self, body: dict[str, Any]) -> None:
        self.items[body["id"]] = json.loads(json.dumps(body))  # what Cosmos would store


def test_cosmos_store_round_trips_sessions_with_an_expiry() -> None:
    from regintel.conversation import CosmosSessionStore, Session

    container = FakeContainer()
    store = CosmosSessionStore(container)
    assert store.get("s1") is None
    store.save(Session("s1", [Turn("q", "standalone q", "answer", "grounded", at=1.0)]))
    assert container.items["s1"]["ttl"] == 7 * 24 * 3600
    loaded = store.get("s1")
    assert loaded is not None and loaded.turns[0].standalone == "standalone q" and loaded.turns[0].at == 1.0


def test_telemetry_is_off_without_a_connection_string() -> None:
    from regintel.runtime import configure_telemetry

    assert configure_telemetry(None) is False

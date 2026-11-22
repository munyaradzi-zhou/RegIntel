"""The research agent on Foundry Agent Service: the published definition and the hosted run path."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from regintel.agent.foundry import HostedAgent, all_tools, strict_schema
from regintel.agent.research import RegIntelAgent
from regintel.answering.schema import DraftAnswer, DraftCitation, DraftStatement, Usage
from regintel.ingestion.chunking import Chunk
from regintel.retrieval.base import Hit

PASSAGE = Chunk("ich-e2a#0001", "ich-e2a", "ICH", "III.A.1", "Fatal or life-threatening", 7, 7,
                "Fatal or life-threatening unexpected ADRs should be reported as soon as possible but no later than "
                "7 calendar days after first knowledge.", "[ICH E2A]")  # fmt: skip


def test_strict_schema_requires_every_property_and_allows_null_for_optional_ones() -> None:
    recalls = next(t for t in all_tools() if t.name == "recent_recalls")
    schema = strict_schema(recalls.parameters())
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["drug", "since", "until"]
    since = schema["properties"]["since"]
    assert "default" not in since and {"type": "null"} in since["anyOf"]  # optional = nullable, not omitted


def test_the_published_definition_declares_every_tool() -> None:
    names = {t.name for t in all_tools()}
    assert names == {"search_guidance", "lookup_cdisc_term", "faers_drug_summary", "faers_disproportionality",
                     "drug_label_section", "recent_recalls"}  # fmt: skip


class FakeRetriever:
    name = "fake"

    def search(self, query: str, top_k: int = 10) -> list[Hit]:
        return [Hit(PASSAGE, 1.0)]


class FakeWriter:
    def draft(self, question: str, hits: list[Hit]) -> tuple[DraftAnswer, Usage]:
        quote = "no later than 7 calendar days after first knowledge"
        statement = DraftStatement(
            text="Report within 7 calendar days.", citations=[DraftCitation(passage=1, quote=quote)]
        )
        return DraftAnswer(answerable=True, statements=[statement], decline_reason=None), Usage(calls=1)


class FakeHosted:
    """Stands in for the Foundry service: plans by calling the local tools it was handed."""

    label = "foundry:regintel-research@7"

    def __init__(self) -> None:
        self.tools: dict[str, Any] = {}

    def agent(self, tools: list[Any], max_tool_calls: int = 8) -> Any:
        self.tools = {t.name: t for t in tools}
        hosted = self

        class Run:
            async def run(self, question: str) -> Any:
                hosted.tools["search_guidance"].func("ICH E2A fatal unexpected ADR timeline")
                return SimpleNamespace(text="E1 answers it.", usage_details={"input_token_count": 640})

        return Run()


def test_hosted_planning_runs_local_tools_then_the_local_writer_and_verifier() -> None:
    hosted = FakeHosted()
    agent = RegIntelAgent(None, FakeWriter(), FakeRetriever(), hosted=hosted)  # type: ignore[arg-type]
    answer = asyncio.run(agent.answer("How fast must fatal unexpected ADRs be reported?"))
    assert answer.result.label == "grounded" and [c.name for c in answer.calls] == ["search_guidance"]
    assert answer.result.shown[0].citations[0].chunk_id == "ich-e2a#0001"  # verified against the local evidence
    assert answer.research_usage["input_token_count"] == 640


def test_hosted_agent_needs_an_endpoint_and_a_version() -> None:
    from regintel.agent.factory import AzureComponents

    settings = SimpleNamespace(agent_hosting="foundry", foundry_project_endpoint=None, foundry_agent_version=None,
                               foundry_agent_name="regintel-research")  # fmt: skip
    with pytest.raises(ValueError, match="FOUNDRY_PROJECT_ENDPOINT"):
        AzureComponents.hosted_agent(SimpleNamespace(settings=settings))  # type: ignore[arg-type]
    settings.agent_hosting = "in_process"
    assert AzureComponents.hosted_agent(SimpleNamespace(settings=settings)) is None  # type: ignore[arg-type]
    assert (
        HostedAgent("https://x/api/projects/p", "regintel-research", "3", None).label == "foundry:regintel-research@3"
    )

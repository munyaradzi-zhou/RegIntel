"""RegIntel as an MCP server, so MCP clients (Claude Desktop, IDEs, agents) can use it as tools.

    python -m regintel.mcp_server              # stdio (Claude Desktop config: command + args)

Tools, all read-only:
- ask_regulatory_question: the full pipeline (guard, router, verifier, reviewer). With
  REGINTEL_API_URL set it calls the deployed API (thin client, API key from REGINTEL_API_KEY);
  otherwise it runs in-process with the local Azure credentials.
- lookup_cdisc_term, faers_drug_summary, faers_disproportionality: local data, no model involved.
- drug_label_section, recent_recalls: live openFDA lookups (no model involved).

Answers are returned as structured JSON (label, statements, numbered sources with quotes), so the
calling model can show citations instead of re-stating facts from memory.
"""

from __future__ import annotations

import json
import os
from functools import cache
from typing import TYPE_CHECKING, Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations
from pydantic import Field

from regintel import data_tools
from regintel.openfda.drugs import Section  # noqa: TC001  (resolved at runtime for the MCP tool schema)

if TYPE_CHECKING:
    from regintel.conversation import RegIntelChat
    from regintel.faers.query import FaersStore
    from regintel.terminology.cdisc import Terminology

INSTRUCTIONS = """\
RegIntel answers questions about FDA, EMA/EU and ICH regulatory guidance with citations that are \
verified in code against the source text, and computes FAERS adverse-event statistics and CDISC \
terminology lookups from data. When you use an answer, keep its citations and its label: \
'declined' means the indexed guidance does not answer the question. Not legal, medical or \
regulatory advice.\
"""

READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False)
LIVE = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=True)


@cache
def _chat() -> RegIntelChat:
    from regintel.api.app import build_chat

    return build_chat()


def _terminology() -> Terminology:
    return data_tools.terminology()


def _faers() -> FaersStore:
    return data_tools.faers_store()


async def _ask_remote(url: str, question: str, session_id: str | None) -> dict[str, Any]:
    import httpx

    headers = {"X-API-Key": key} if (key := os.environ.get("REGINTEL_API_KEY")) else {}
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            url.rstrip("/") + "/v1/ask", json={"question": question, "session_id": session_id}, headers=headers
        )
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result


def build_server() -> MCPServer:
    server = MCPServer(name="regintel", title="RegIntel", instructions=INSTRUCTIONS, version="0.1.0")

    @server.tool(
        title="Ask about regulatory guidance",
        description="Answer a question about FDA, EMA/EU or ICH guidance (or FAERS / CDISC data) with verified "
        "citations. Pass the returned session_id to ask a follow-up question.",
        annotations=READ_ONLY,
    )
    async def ask_regulatory_question(
        question: Annotated[str, Field(description="The question, in plain language.", max_length=2000)],
        session_id: Annotated[str | None, Field(description="From an earlier answer, for follow-ups.")] = None,
    ) -> dict[str, Any]:
        if url := os.environ.get("REGINTEL_API_URL"):
            return await _ask_remote(url, question, session_id)
        from regintel.api.schemas import AskResponse

        answer = await _chat().ask(question, session_id)
        return json.loads(AskResponse.from_chat(answer).model_dump_json())  # type: ignore[no-any-return]

    @server.tool(
        title="Look up CDISC terminology",
        description="Exact lookup in CDISC Controlled Terminology (SDTM, ADaM, SEND, glossary) by submission value, "
        "codelist name or term. Codelists include their member terms.",
        annotations=READ_ONLY,
    )
    def lookup_cdisc_term(
        term: Annotated[str, Field(description="Submission value or name, e.g. 'AESEV' or 'Sex'.", max_length=200)],
    ) -> list[dict[str, Any]]:
        return data_tools.cdisc_lookup(_terminology(), term)

    @server.tool(
        title="FAERS drug summary",
        description="Reports listing a drug as suspect in the loaded FAERS extract: total, serious, deaths and the "
        "most frequent reactions. Computed from data.",
        annotations=READ_ONLY,
    )
    def faers_drug_summary(
        drug: Annotated[str, Field(description="Active substance or generic name.", max_length=200)],
    ) -> dict[str, Any]:
        return data_tools.faers_summary(_faers(), drug)

    @server.tool(
        title="FAERS disproportionality",
        description="Reporting odds ratio with 95% CI for a drug and a MedDRA reaction term in the loaded FAERS "
        "extract. A signal-screening statistic, not evidence of causation.",
        annotations=READ_ONLY,
    )
    def faers_disproportionality(
        drug: Annotated[str, Field(description="Active substance or generic name.", max_length=200)],
        reaction: Annotated[str, Field(description="MedDRA preferred term, e.g. 'Nausea'.", max_length=200)],
    ) -> dict[str, Any]:
        return data_tools.faers_signal(_faers(), drug, reaction)

    @server.tool(
        title="FDA drug label section",
        description="One section of a drug's current US label (boxed warning, indications, contraindications, "
        "warnings and precautions, ...), live from openFDA, with the label set id, version and effective date.",
        annotations=LIVE,
    )
    async def drug_label_section(
        drug: Annotated[str, Field(description="Generic or brand name.", max_length=200)],
        section: Annotated[Section, Field(description="The label section.")],
    ) -> dict[str, Any]:
        return await data_tools.label_section(data_tools.openfda(), drug, section)

    @server.tool(
        title="FDA drug recalls",
        description="Drug recalls (FDA enforcement reports) for a drug, live from openFDA, most recent first: recall "
        "number, class, status, firm, reason and product.",
        annotations=LIVE,
    )
    async def recent_recalls(
        drug: Annotated[str, Field(description="Generic or brand name.", max_length=200)],
        since: Annotated[str | None, Field(description="Earliest report date, YYYY-MM-DD.")] = None,
        until: Annotated[str | None, Field(description="Latest report date, YYYY-MM-DD.")] = None,
    ) -> dict[str, Any]:
        return await data_tools.drug_recalls(data_tools.openfda(), drug, since, until)

    return server


def main() -> None:
    from regintel.runtime import log_to_stderr, use_os_certificates

    log_to_stderr()
    use_os_certificates()
    build_server().run("stdio")


if __name__ == "__main__":
    main()

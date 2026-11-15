"""The research agent's tools. Each records its result in the evidence pool and returns it,
numbered, so the agent can see what it has; nothing here generates text with a model.

- search_guidance: one Azure AI Search knowledge base retrieval (minimal effort: the agent does
  the decomposition, one call per sub-question, which is what the multi-hop evaluation favoured).
- lookup_cdisc_term: exact lookup in CDISC Controlled Terminology (NCI EVS).
- faers_drug_summary / faers_disproportionality: DuckDB computations over the FAERS extract.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated

from agent_framework import FunctionTool, tool
from pydantic import Field

from regintel.agent.evidence import EvidencePool
from regintel.faers.query import SCOPE

if TYPE_CHECKING:
    from collections.abc import Callable

    from regintel.faers.query import FaersStore
    from regintel.retrieval.base import Retriever
    from regintel.terminology.cdisc import Term, Terminology

PASSAGES_PER_SEARCH = 4
MAX_CODELIST_MEMBERS = 20


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, str]
    evidence: list[int]
    seconds: float


@dataclass
class ResearchTools:
    retriever: Retriever
    terminology: Terminology | None = None
    faers: FaersStore | None = None
    terminology_version: str = ""
    pool: EvidencePool = field(default_factory=EvidencePool)
    calls: list[ToolCall] = field(default_factory=list)

    def _record(self, name: str, arguments: dict[str, str], started: float, numbers: list[int]) -> None:
        self.calls.append(ToolCall(name, arguments, numbers, round(time.perf_counter() - started, 2)))

    def search_guidance(
        self,
        query: Annotated[
            str, Field(description="One focused sub-question, naming the regulator or guideline if known.")
        ],
    ) -> str:
        """Search the indexed FDA, EMA/EU and ICH guidance. Call once per sub-question; for a comparison,
        search each jurisdiction separately."""
        started = time.perf_counter()
        hits = self.retriever.search(query, top_k=PASSAGES_PER_SEARCH)
        items = [self.pool.add("guidance", h.chunk) for h in hits]
        self._record("search_guidance", {"query": query}, started, [i.number for i in items])
        if not items:
            return "No guidance passages found for this query."
        return "\n\n".join(item.render() for item in items)

    def lookup_cdisc_term(
        self,
        term: Annotated[str, Field(description="A CDISC submission value, codelist name or term, e.g. 'AESEV'.")],
    ) -> str:
        """Look up a term or codelist in CDISC Controlled Terminology (SDTM, ADaM, SEND, glossary). Exact
        matches only: use the submission value or the full name."""
        started = time.perf_counter()
        if self.terminology is None:
            return "CDISC terminology is not available."
        matches = self.terminology.lookup(term, limit=3)
        items = [
            self.pool.add_record(
                "terminology",
                f"cdisc:{t.standard}:{t.codelist_code}:{t.code}",
                f"CDISC Controlled Terminology · {t.standard.upper()} · {self.terminology_version}",
                self._describe(t),
            )
            for t in matches
        ]
        self._record("lookup_cdisc_term", {"term": term}, started, [i.number for i in items])
        if not items:
            return f"No CDISC Controlled Terminology entry matches '{term}' exactly."
        return "\n\n".join(item.render() for item in items)

    def _describe(self, t: Term) -> str:
        if t.is_codelist:
            members = self.terminology.codelist(t.code) if self.terminology is not None else []
            listed = ", ".join(f"{m.submission_value} ({m.code})" for m in members[:MAX_CODELIST_MEMBERS])
            more = f", and {len(members) - MAX_CODELIST_MEMBERS} more" if len(members) > MAX_CODELIST_MEMBERS else ""
            head = f"Codelist {t.codelist_name or t.preferred_term} ({t.code}), submission value {t.submission_value}."
            body = f" It has {len(members)} terms: {listed}{more}." if members else ""
            return f"{head}{body} Definition: {t.definition}"
        return (
            f"Term {t.submission_value} ({t.code}) in codelist {t.codelist_name} ({t.codelist_code}). "
            f"Preferred term: {t.preferred_term}. Synonyms: {', '.join(t.synonyms) or 'none'}. "
            f"Definition: {t.definition}"
        )

    def faers_drug_summary(
        self, drug: Annotated[str, Field(description="Active substance or generic name, e.g. 'dupilumab'.")]
    ) -> str:
        """Count FAERS reports listing a drug as suspect: total, serious, deaths, and the most frequent reactions.
        Computed from the FAERS 2025 Q1 extract, not from model knowledge."""
        started = time.perf_counter()
        if self.faers is None:
            return "FAERS data is not available."
        text = self.faers.drug_summary(drug).to_text()
        item = self.pool.add_record("faers", f"faers:summary:{drug.lower()}", f"{SCOPE} · computed", text)
        self._record("faers_drug_summary", {"drug": drug}, started, [item.number])
        return item.render()

    def faers_disproportionality(
        self,
        drug: Annotated[str, Field(description="Active substance or generic name.")],
        reaction: Annotated[str, Field(description="MedDRA preferred term as reported, e.g. 'Conjunctivitis'.")],
    ) -> str:
        """Reporting odds ratio (with 95% CI) for a drug and a reaction in FAERS: a signal-screening statistic,
        not proof of causation. Computed from the FAERS 2025 Q1 extract."""
        started = time.perf_counter()
        if self.faers is None:
            return "FAERS data is not available."
        text = self.faers.disproportionality(drug, reaction).to_text()
        record_id = f"faers:ror:{drug.lower()}:{reaction.lower()}"
        item = self.pool.add_record("faers", record_id, f"{SCOPE} · computed", text)
        self._record("faers_disproportionality", {"drug": drug, "reaction": reaction}, started, [item.number])
        return item.render()

    def as_tools(self) -> list[FunctionTool]:
        available: list[Callable[..., str]] = [self.search_guidance]
        if self.terminology is not None:
            available.append(self.lookup_cdisc_term)
        if self.faers is not None:
            available += [self.faers_drug_summary, self.faers_disproportionality]
        return [tool(f) for f in available]

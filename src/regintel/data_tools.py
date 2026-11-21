"""Structured payloads for the data tools, shared by the HTTP API and the MCP server.

Everything here is computed from data (FAERS Parquet, CDISC Controlled Terminology, live openFDA
labels and recalls); no model is involved, so the numbers can be shown as they are.
"""

from __future__ import annotations

from dataclasses import asdict
from functools import cache
from typing import TYPE_CHECKING, Any

from regintel.config import get_settings
from regintel.faers.query import CAVEAT, SCOPE
from regintel.openfda import drugs
from regintel.openfda.client import OpenFDAClient

if TYPE_CHECKING:
    from regintel.faers.query import FaersStore
    from regintel.terminology.cdisc import Terminology

SIGNAL_REACTIONS = 8


@cache
def faers_store() -> FaersStore:
    from regintel.faers.query import FaersStore

    return FaersStore(get_settings().data_dir / "processed" / "faers")


@cache
def openfda() -> OpenFDAClient:
    key = get_settings().openfda_api_key
    return OpenFDAClient(key.get_secret_value() if key else None)


async def label_section(client: OpenFDAClient, drug: str, section: drugs.Section) -> dict[str, Any]:
    result = await drugs.label_section(client, drug, section)
    return asdict(result) | {"source": result.source_label if result.found else None, "text": result.to_text()}


async def drug_recalls(client: OpenFDAClient, drug: str, since: str | None, until: str | None) -> dict[str, Any]:
    result = await drugs.recalls(client, drug, since, until)
    return asdict(result) | {"source": result.source_label, "text": result.to_text()}


@cache
def terminology() -> Terminology:
    from regintel.terminology.cdisc import load_terminology

    return load_terminology(get_settings().data_dir / "raw" / "cdisc")


def faers_summary(store: FaersStore, drug: str) -> dict[str, Any]:
    summary = store.drug_summary(drug, top=SIGNAL_REACTIONS)
    return {
        "drug": drug, "scope": SCOPE, "matched_names": summary.matched_names, "reports": summary.reports,
        "serious": summary.serious, "deaths": summary.deaths, "total_reports_in_extract": summary.total_reports,
        "top_reactions": [{"reaction": r, "reports": n} for r, n in summary.top_reactions],
        "text": summary.to_text(),
    }  # fmt: skip


def faers_signal(store: FaersStore, drug: str, reaction: str) -> dict[str, Any]:
    result = store.disproportionality(drug, reaction)
    ci = result.ci95
    return {
        "drug": drug, "reaction": reaction, "a": result.a, "b": result.b, "c": result.c, "d": result.d,
        "ror": round(result.ror, 2) if result.ror is not None else None,
        "ci95": [round(ci[0], 2), round(ci[1], 2)] if ci else None,
        "text": result.to_text(),
    }  # fmt: skip


def faers_profile(store: FaersStore, drug: str) -> dict[str, Any]:
    """Summary plus a reporting odds ratio for each of the drug's most frequent reactions."""
    summary = faers_summary(store, drug)
    summary["signals"] = [faers_signal(store, drug, r["reaction"]) for r in summary["top_reactions"]]
    summary["caveat"] = CAVEAT
    return summary


def cdisc_lookup(terms: Terminology, term: str, limit: int = 5) -> list[dict[str, Any]]:
    results = []
    for t in terms.lookup(term, limit=limit):
        members = terms.codelist(t.code) if t.is_codelist else []
        results.append({
            "standard": t.standard, "code": t.code, "codelist_code": t.codelist_code or None,
            "codelist_name": t.codelist_name, "submission_value": t.submission_value,
            "preferred_term": t.preferred_term, "synonyms": list(t.synonyms), "definition": t.definition,
            "terms": [{"submission_value": m.submission_value, "code": m.code} for m in members[:50]],
        })  # fmt: skip
    return results

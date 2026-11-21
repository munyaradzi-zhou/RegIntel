"""openFDA client and tools against recorded responses (respx): retries, cache, failures, evidence."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from regintel.agent.tools import ResearchTools
from regintel.openfda import drugs
from regintel.openfda.client import BASE_URL, OpenFDAClient, OpenFDAUnavailableError

LABEL_URL = BASE_URL + drugs.LABEL_ENDPOINT
RECALL_URL = BASE_URL + drugs.ENFORCEMENT_ENDPOINT
NOT_FOUND = {"error": {"code": "NOT_FOUND", "message": "No matches found!"}}

OZEMPIC = {
    "set_id": "42bdd912-2393-44c4-b7e0-47672ca28991",
    "version": "1",
    "effective_time": "20260730",
    "openfda": {"brand_name": ["Ozempic"], "generic_name": ["SEMAGLUTIDE"], "route": ["SUBCUTANEOUS"]},
    "boxed_warning": [
        "WARNING: RISK OF THYROID C-CELL TUMORS",
        "In rodents, semaglutide causes thyroid C-cell tumors.",
    ],
    "contraindications": ["OZEMPIC is contraindicated in patients with a personal or family history of MTC."],
}
NO_BOXED = {
    "set_id": "older",
    "version": "9",
    "effective_time": "20250101",
    "openfda": {},
    "indications_and_usage": ["x"],
}
RECALL = {
    "recall_number": "D-0328-2025",
    "report_date": "20250416",
    "recall_initiation_date": "20250313",
    "classification": "Class II",
    "status": "Ongoing",
    "recalling_firm": "Glenmark Pharmaceuticals Inc., USA",
    "reason_for_recall": "CGMP Deviations",
    "product_description": "Metformin Hydrochloride Extended-Release Tablets 1000mg, 90-count bottle.",
}


def _page(*results: dict[str, Any], total: int | None = None) -> dict[str, Any]:
    return {"meta": {"results": {"total": len(results) if total is None else total}}, "results": list(results)}


def _client(**kwargs: Any) -> OpenFDAClient:
    return OpenFDAClient(max_wait=0.01, **kwargs)  # near-instant backoff in tests


@respx.mock
async def test_label_section_reads_the_newest_label_that_has_the_section() -> None:
    route = respx.get(LABEL_URL).respond(200, json=_page(NO_BOXED, OZEMPIC))
    result = await drugs.label_section(_client(), "semaglutide", "boxed_warning")
    assert result.found and result.set_id == OZEMPIC["set_id"] and result.effective == "2026-07-30"
    assert "RISK OF THYROID C-CELL TUMORS In rodents" in result.to_text()
    assert result.record_id == f"openfda:label:{OZEMPIC['set_id']}:v1:boxed_warning"
    params = route.calls[0].request.url.params
    assert params["search"] == '(openfda.generic_name:"semaglutide" OR openfda.brand_name:"semaglutide")'
    assert params["sort"] == "effective_time:desc"


@respx.mock
async def test_no_match_is_an_answer_not_a_failure() -> None:
    respx.get(LABEL_URL).respond(404, json=NOT_FOUND)
    result = await drugs.label_section(_client(), "notadrug", "boxed_warning")
    assert not result.found and "no drug label matching" in result.to_text()

    respx.get(LABEL_URL).respond(200, json=_page(NO_BOXED))
    result = await drugs.label_section(_client(), "other", "boxed_warning")
    assert "1 label(s) for 'other', but none has a boxed warning section" in result.to_text()


@respx.mock
async def test_drug_names_cannot_add_query_clauses() -> None:
    route = respx.get(LABEL_URL).respond(404, json=NOT_FOUND)
    await drugs.label_section(_client(), 'x" OR openfda.route:"ORAL', "boxed_warning")
    search = route.calls[0].request.url.params["search"]
    assert search.count('"') == 4  # only the quotes this module adds
    assert drugs.clean_name('a"b)(c:*') == "a b c"


@respx.mock
async def test_429_and_5xx_are_retried() -> None:
    route = respx.get(LABEL_URL)
    route.side_effect = [
        httpx.Response(429, headers={"Retry-After": "0"}),
        httpx.Response(503),
        httpx.Response(200, json=_page(OZEMPIC)),
    ]
    client = _client()
    assert (await drugs.label_section(client, "semaglutide", "boxed_warning")).found
    assert route.call_count == 3 and client.requests == 3


@respx.mock
async def test_persistent_failure_raises_after_the_attempts() -> None:
    respx.get(LABEL_URL).respond(503)
    client = _client(attempts=2)
    with pytest.raises(OpenFDAUnavailableError, match="after 2 attempts"):
        await drugs.label_section(client, "semaglutide", "boxed_warning")
    assert client.requests == 2


@respx.mock
async def test_timeouts_are_retried_then_reported() -> None:
    route = respx.get(LABEL_URL)
    route.side_effect = [httpx.ReadTimeout("slow"), httpx.Response(200, json=_page(OZEMPIC))]
    assert (await drugs.label_section(_client(), "semaglutide", "boxed_warning")).found

    route.side_effect = httpx.ConnectTimeout("down")
    with pytest.raises(OpenFDAUnavailableError):
        await drugs.label_section(_client(), "semaglutide", "boxed_warning")


@respx.mock
async def test_other_statuses_and_bodies_fail_without_retry() -> None:
    route = respx.get(LABEL_URL).respond(400, json={"error": {"code": "BAD_REQUEST"}})
    with pytest.raises(OpenFDAUnavailableError, match="HTTP 400"):
        await drugs.label_section(_client(), "semaglutide", "boxed_warning")
    assert route.call_count == 1
    respx.get(LABEL_URL).respond(200, text="<html>maintenance</html>")
    with pytest.raises(OpenFDAUnavailableError, match="unexpected body"):
        await drugs.label_section(_client(), "semaglutide", "boxed_warning")


@respx.mock
async def test_results_are_cached_and_the_api_key_is_sent_but_not_cached_on() -> None:
    route = respx.get(LABEL_URL).respond(200, json=_page(OZEMPIC))
    client = _client(api_key="k")
    for _ in range(3):
        await drugs.label_section(client, "semaglutide", "boxed_warning")
    assert route.call_count == 1 and route.calls[0].request.url.params["api_key"] == "k"
    assert all("api_key" not in key for key in client.cache._items)
    await drugs.label_section(client, "semaglutide", "contraindications")  # same labels, another section
    assert route.call_count == 1
    await drugs.label_section(client, "metformin", "boxed_warning")  # different query: new request
    assert route.call_count == 2


@respx.mock
async def test_recalls_query_the_report_date_window() -> None:
    route = respx.get(RECALL_URL).respond(200, json=_page(RECALL, total=3))
    result = await drugs.recalls(_client(), "metformin", "2025-01-01", "2025-06-30")
    search = route.calls[0].request.url.params["search"]
    assert search.endswith("AND report_date:[20250101 TO 20250630]") and 'product_description:"metformin"' in search
    text = result.to_text()
    assert "from 2025-01-01 to 2025-06-30" in text and ": 3 recall(s). The 1 most recent are listed." in text
    assert "Recall D-0328-2025: Class II, status Ongoing, reported 2025-04-16" in text
    with pytest.raises(ValueError):
        await drugs.recalls(_client(), "metformin", "last year")


@respx.mock
async def test_tools_add_evidence_only_for_what_openfda_returned() -> None:
    respx.get(LABEL_URL).respond(200, json=_page(OZEMPIC))
    respx.get(RECALL_URL).respond(503)
    tools = ResearchTools(retriever=None, openfda=_client(attempts=1))  # type: ignore[arg-type]
    assert {t.name for t in tools.as_tools()} >= {"drug_label_section", "recent_recalls"}

    label = await tools.drug_label_section("Ozempic", "contraindications")
    assert label.startswith("[E1] [FDA drug label via openFDA · Ozempic (SEMAGLUTIDE) · effective 2026-07-30]")
    down = await tools.recent_recalls("Ozempic")
    assert "do not state whether the drug was recalled" in down
    assert [e.kind for e in tools.pool.items] == ["label"]  # the failed call added nothing citable
    assert [c.name for c in tools.calls] == ["drug_label_section", "recent_recalls"]
    assert tools.calls[1].evidence == []
    assert await tools.recent_recalls("Ozempic", since="soon") == "Dates must be YYYY-MM-DD."


def test_without_a_client_the_tools_are_not_offered() -> None:
    tools = ResearchTools(retriever=None)  # type: ignore[arg-type]
    assert "drug_label_section" not in {t.name for t in tools.as_tools()}

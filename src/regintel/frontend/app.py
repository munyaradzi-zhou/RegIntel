"""RegIntel front end (Streamlit, Paper & Ember).

    regintel                                  # start the API first (http://127.0.0.1:8000)
    streamlit run src/regintel/frontend/app.py

Talks to the API only (REGINTEL_API_URL, REGINTEL_API_KEY). Model text is rendered escaped, never
as HTML.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import pandas as pd
import streamlit as st

from regintel.frontend import charts, client
from regintel.frontend.theme import (
    answer_body,
    apply_theme,
    card_trio,
    footer,
    page_header,
    pointers,
    section_heading,
    source_card,
    stat_tiles,
    status_row,
)

EXAMPLES = [
    "What acceptance range applies to the 90% confidence interval in an EU bioequivalence study?",
    "How quickly must a fatal unexpected suspected adverse reaction be reported to FDA, and under ICH E2A?",
    "Is there a conjunctivitis signal for dupilumab in FAERS?",
    "Which CDISC codelist holds adverse event severity values?",
]
PATHS = {
    "guidance_single": "path · guidance",
    "guidance_multi": "path · guidance, multi-part",
    "agent": "path · research agent",
}

st.set_page_config(page_title="RegIntel", page_icon="📑", layout="wide")


@st.cache_data(ttl=3600, show_spinner=False)
def faers_profile(drug: str) -> dict[str, Any]:
    """Cached: Streamlit reruns every tab on each interaction, and the data does not change."""
    return client.faers_profile(drug)


@st.cache_data(ttl=3600, show_spinner=False)
def cdisc(term: str) -> list[dict[str, Any]]:
    return client.cdisc(term)


@st.cache_data(ttl=30, show_spinner=False)
def api_ready() -> bool:
    return client.ready()


apply_theme()
st.session_state.setdefault("messages", [])  # [{"role": "user"|"assistant", "body": ...}]
st.session_state.setdefault("session_id", None)
st.session_state.setdefault("pending", None)


def render_answer(body: dict[str, Any]) -> None:
    chips = [PATHS.get(body.get("path") or "", "")] if body.get("path") else []
    if body.get("is_followup"):
        chips.append("follow-up")
    if body.get("dropped"):
        chips.append(f"{body['dropped']} removed by checks")
    total = sum(body.get("timings", {}).values())
    if total:
        chips.append(f"{total:.1f} s")
    status_row(body["label"], [c for c in chips if c])
    if body.get("is_followup"):
        st.caption(f"Answered as: {body['standalone_question']}")
    if body["statements"]:
        answer_body(body["statements"])
    else:
        st.markdown(body["answer"].split("\n\nPages on regulator websites")[0])  # plain text, escaped by Streamlit
    if body.get("web_pointers"):
        pointers(body["web_pointers"])
    if body["sources"]:
        with st.expander(f"Sources ({len(body['sources'])}) · verbatim quotes, checked against the text"):
            for source in body["sources"]:
                source_card(source["number"], source["label"], source["quote"])
    if body.get("tool_calls"):
        with st.expander("What the research agent did"):
            for call in body["tool_calls"]:
                args = ", ".join(f"{k}={v!r}" for k, v in call["arguments"].items())
                st.markdown(f"`{call['name']}({args})` · {call['seconds']} s")


def ask(question: str) -> None:
    st.session_state.messages.append({"role": "user", "body": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        status = st.status("Checking the guidance…", expanded=False)
        body: dict[str, Any] | None = None
        try:
            for event, data in client.ask_stream(question, st.session_state.session_id):
                if event == "session":
                    st.session_state.session_id = data["session_id"]
                elif event == "progress":
                    status.update(label=f"Retrieving, writing and checking… {data['elapsed']:.0f} s")
                elif event == "answer":
                    body = data
                elif event == "error":
                    status.update(label="The answer could not be produced", state="error")
        except httpx.HTTPError as error:
            status.update(label="The API could not be reached", state="error")
            st.error(f"Could not reach the RegIntel API at {client.API_URL} ({error.__class__.__name__}).")
            return
        if body is None:
            return
        status.update(label="Answered", state="complete")
        render_answer(body)
        st.session_state.messages.append({"role": "assistant", "body": body})


with st.sidebar:
    st.markdown('<p class="pe-eyebrow">Conversation</p>', unsafe_allow_html=True)
    if st.button("New conversation", use_container_width=True):
        st.session_state.messages, st.session_state.session_id = [], None
        st.rerun()
    st.caption(f"Session: `{st.session_state.session_id or '—'}`")
    st.markdown('<p class="pe-eyebrow" style="margin-top:24px">Service</p>', unsafe_allow_html=True)
    st.caption("API ready" if api_ready() else f"API not reachable at {client.API_URL}")
    st.markdown('<p class="pe-eyebrow" style="margin-top:24px">Corpus</p>', unsafe_allow_html=True)
    st.caption("23 final FDA, EMA/EU and ICH guidances (checksum-pinned), CDISC Controlled Terminology, "
               "openFDA FAERS 2025 Q1 extract (36,000 reports).")  # fmt: skip

page_header(
    "Regulatory intelligence · FDA · EMA · ICH",
    "Ask the guidance.",
    "Check every quote.",
    "Answers cite the guidance word for word. Code checks every quote and number against the source, a "
    "second model checks that no statement says more than its quote, and questions the guidance does not "
    "cover are declined.",
)

tab_ask, tab_signals, tab_terms, tab_how = st.tabs(["Ask", "Safety signals", "CDISC terms", "How it works"])

with tab_ask:
    # The conversation sits above the input: inside a tab, st.chat_input renders where it is called.
    conversation = st.container()
    typed = st.chat_input("Ask about FDA, EMA or ICH guidance, FAERS reports or CDISC terms")
    question = typed or st.session_state.pending
    st.session_state.pending = None
    with conversation:
        if not st.session_state.messages and not question:
            st.markdown('<p class="pe-eyebrow" style="margin-top:12px">Try</p>', unsafe_allow_html=True)
            columns = st.columns(2)
            for i, example in enumerate(EXAMPLES):
                if columns[i % 2].button(example, key=f"example-{i}", use_container_width=True):
                    st.session_state.pending = example
                    st.rerun()
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.markdown(message["body"])
                else:
                    render_answer(message["body"])
        if question:
            ask(question)

with tab_signals:
    section_heading(
        "FAERS · computed from data",
        "Safety signals",
        "Reports listing a drug as suspect in the openFDA FAERS 2025 Q1 extract, and a reporting odds ratio "
        "for each of its most frequent reactions. Disproportionality screens for signals; it does not show "
        "that the drug causes the reaction.",
    )
    drug = st.text_input("Drug (active substance or generic name)", value="pembrolizumab", key="drug")
    if drug.strip():
        try:
            started = time.perf_counter()
            profile: dict[str, Any] | None = faers_profile(drug.strip())
        except httpx.HTTPError:
            st.error(f"Could not reach the RegIntel API at {client.API_URL}.")
            profile = None
        if profile and profile["reports"] == 0:
            st.info(f"No reports in the extract list {drug.upper()} as a suspect drug.")
        elif profile:
            serious_pct = 100 * profile["serious"] / profile["reports"]
            stat_tiles([
                (f"{profile['reports']:,}", "Reports, suspect drug"),
                (f"{serious_pct:.1f}%", "Serious"),
                (f"{profile['deaths']:,}", "Death reported"),
                (f"{profile['total_reports_in_extract']:,}", "Reports in extract"),
            ])  # fmt: skip
            left, right = st.columns(2, gap="large")
            with left:
                st.markdown("#### Most frequent reactions")
                st.altair_chart(charts.reactions_bar(profile), use_container_width=True)
            with right:
                st.markdown("#### Reporting odds ratio")
                st.altair_chart(charts.ror_forest(profile), use_container_width=True)
                st.caption("Right of the dashed line: reported more often with this drug than with others in "
                           "the extract. An interval crossing the line is not a clear signal.")  # fmt: skip
            with st.expander("Table view"):
                table = pd.DataFrame([
                    {"Reaction": s["reaction"], "Reports with drug": s["a"], "ROR": s["ror"],
                     "95% CI": f"{s['ci95'][0]} to {s['ci95'][1]}" if s["ci95"] else "n/a"}
                    for s in profile["signals"]
                ])  # fmt: skip
                st.dataframe(table, hide_index=True, use_container_width=True)
            st.caption(f"Matched names: {', '.join(profile['matched_names'])} · {profile['scope']} · "
                       f"computed in {time.perf_counter() - started:.1f} s")  # fmt: skip

with tab_terms:
    section_heading(
        "CDISC Controlled Terminology",
        "Look up a term",
        "Exact lookup by submission value, codelist name or term (SDTM, ADaM, SEND, glossary).",
    )
    term = st.text_input("Term or submission value", value="AESEV", key="term")
    if term.strip():
        try:
            matches: list[dict[str, Any]] = cdisc(term.strip())
        except httpx.HTTPError:
            st.error(f"Could not reach the RegIntel API at {client.API_URL}.")
            matches = []
        if not matches:
            st.info(f"No exact match for '{term}'. Try the submission value (e.g. AESEV) or the full codelist name.")
        for match in matches:
            kind = "Codelist" if match["codelist_code"] is None else f"Term in {match['codelist_name']}"
            st.markdown(f"#### {match['submission_value']} · {match['code']}")
            st.caption(f"{kind} · {match['standard'].upper()} · {match['preferred_term']}")
            st.markdown(match["definition"])
            if match["terms"]:
                st.dataframe(pd.DataFrame(match["terms"]).rename(columns={"submission_value": "Submission value",
                                                                          "code": "NCI code"}),
                             hide_index=True, use_container_width=True)  # fmt: skip

with tab_how:
    section_heading("How it works", "Retrieve, write, check")
    card_trio([
        ("Retrieve", "Azure AI Search knowledge base over section-aware chunks of 23 guidances; an agent with "
         "FAERS and CDISC tools for data questions.", "router → guidance | agent"),
        ("Verify", "Every quote must appear in the cited passage and every number in its quotes. Statements "
         "that fail are removed; nothing verified means no answer.", "regintel.answering.verify"),
        ("Review", "A second model family checks that no statement says more than its quote, and the answer "
         "declines when the guidance never names the subject asked about.", "gpt-oss-120b reviewer"),
    ])  # fmt: skip
    st.markdown(
        "Questions are screened for prompt injection before any model sees them. Follow-ups are rewritten "
        "into standalone questions from earlier questions, never from earlier answers. When the guidance "
        "cannot answer, regulator websites are searched for pages to check; those links are not verified."
    )

footer("RegIntel · regulatory guidance Q&A", "Information about guidance, not legal, medical or regulatory advice")
